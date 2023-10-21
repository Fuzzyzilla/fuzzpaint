#![feature(portable_simd)]
#![feature(once_cell_try)]
use std::sync::Arc;
pub mod commands;
mod egui_impl;
pub mod repositories;
pub mod vulkano_prelude;
pub mod window;
use vulkano_prelude::*;
pub mod actions;
pub mod blend;
pub mod brush;
pub mod document_viewport_proxy;
pub mod gizmos;
pub mod gpu_tess;
pub mod id;
pub mod pen_tools;
pub mod render_device;
pub mod state;
pub mod stylus_events;
pub mod tess;
pub mod ui;
pub mod view_transform;
use blend::Blend;

pub use id::FuzzID;
pub use tess::StrokeTessellator;

#[cfg(feature = "dhat_heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Obviously will be user specified on a per-document basis, but for now...
const DOCUMENT_DIMENSION: u32 = 1080;
/// Premultiplied RGBA16F for interesting effects (negative + overbright colors and alpha) with
/// more than 11bit per channel precision in the \[0,1\] range.
/// Will it be user specified in the future?
const DOCUMENT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

use anyhow::Result as AnyResult;

pub fn preferences_dir() -> Option<std::path::PathBuf> {
    let mut base_dir = dirs::preference_dir()?;
    base_dir.push(env!("CARGO_PKG_NAME"));
    Some(base_dir)
}

pub struct GlobalHotkeys {
    failed_to_load: bool,
    actions_to_keys: actions::hotkeys::ActionsToKeys,
    keys_to_actions: actions::hotkeys::KeysToActions,
}
impl GlobalHotkeys {
    const FILENAME: &'static str = "hotkeys.ron";
    /// Shared global hotkeys, saved and loaded from user preferences.
    /// (Or defaulted, if unavailable for some reason)
    pub fn get() -> &'static Self {
        static GLOBAL_HOTKEYS: std::sync::OnceLock<GlobalHotkeys> = std::sync::OnceLock::new();

        GLOBAL_HOTKEYS.get_or_init(|| {
            let mut dir = preferences_dir();
            match dir.as_mut() {
                None => Self::no_path(),
                Some(dir) => {
                    dir.push(Self::FILENAME);
                    Self::load_or_default(&dir)
                }
            }
        })
    }
    pub fn no_path() -> Self {
        log::warn!("Hotkeys weren't available, defaulting.");
        use actions::hotkeys::*;
        let default = ActionsToKeys::default();
        // Default action map is reversable - this is assured by the default impl when debugging.
        let reverse = (&default).try_into().unwrap();

        Self {
            failed_to_load: true,
            keys_to_actions: reverse,
            actions_to_keys: default,
        }
    }
    fn load_or_default(path: &std::path::Path) -> Self {
        use actions::hotkeys::*;
        let mappings: anyhow::Result<(ActionsToKeys, KeysToActions)> = try_block::try_block! {
            let string = std::fs::read_to_string(path)?;
            let actions_to_keys : ActionsToKeys = ron::from_str(&string)?;
            let keys_to_actions : KeysToActions = (&actions_to_keys).try_into()?;

            Ok((actions_to_keys,keys_to_actions))
        };

        match mappings {
            Ok((actions_to_keys, keys_to_actions)) => Self {
                failed_to_load: false,
                actions_to_keys,
                keys_to_actions,
            },
            Err(_) => Self::no_path(),
        }
    }
    /// Return true if loading user's settings failed. This can be useful for
    /// displaying a warning.
    pub fn did_fail_to_load(&self) -> bool {
        self.failed_to_load
    }
    pub fn save(&self) -> anyhow::Result<()> {
        let mut preferences =
            preferences_dir().ok_or_else(|| anyhow::anyhow!("No preferences dir found"))?;
        // Explicity do *not* create recursively. If not found, the user probably has a good reason.
        // Ignore errors (could already exist). Any real errors will be emitted by file access below.
        let _ = std::fs::DirBuilder::new().create(&preferences);

        preferences.push(Self::FILENAME);
        let writer = std::io::BufWriter::new(std::fs::File::create(&preferences)?);
        Ok(ron::ser::to_writer_pretty(
            writer,
            &self.actions_to_keys,
            Default::default(),
        )?)
    }
}

pub struct StrokeLayer {
    pub id: FuzzID<Self>,
    pub name: String,
    pub blend: blend::Blend,
}
pub struct Document {
    /// The path from which the file was loaded, or None if opened as new.
    pub path: Option<std::path::PathBuf>,
    /// Name of the document, from its path or generated.
    pub name: String,

    // In structure, a document is rather similar to a GroupLayer :O
    /// Layers that make up this document
    pub layer_top_level: Vec<StrokeLayer>,

    /// ID that is unique within this execution of the program
    pub id: FuzzID<Document>,
}
mod stroke_renderer {
    /// The data managed by the renderer.
    /// For now, in persuit of actually getting a working product one day,
    /// this is a very coarse caching sceme. In the future, perhaps a bit more granular
    /// control can occur, should performance become an issue:
    ///  * Caching images of incrementally older states, reducing work to get to any given state (performant undo)
    ///  * Caching tesselation output
    pub struct RenderData {
        image: Arc<vk::StorageImage>,
        pub view: Arc<vk::ImageView<vk::StorageImage>>,
    }

    use crate::vk;
    use anyhow::Result as AnyResult;
    use std::sync::Arc;
    use vulkano::{pipeline::graphics::vertex_input::Vertex, pipeline::Pipeline, sync::GpuFuture};
    mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/stamp.vert",
        }
    }
    mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/stamp.frag",
        }
    }

    pub struct StrokeLayerRenderer {
        context: Arc<crate::render_device::RenderContext>,
        texture_descriptor: Arc<vk::PersistentDescriptorSet>,
        gpu_tess: super::gpu_tess::GpuStampTess,
        pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl StrokeLayerRenderer {
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> AnyResult<Self> {
            let image = image::open("brushes/splotch.png")
                .unwrap()
                .into_luma_alpha8();

            //Iter over transparencies.
            let image_grey = image.iter().skip(1).step_by(2).cloned();

            let mut cb = vk::AutoCommandBufferBuilder::primary(
                context.allocators().command_buffer(),
                context.queues().transfer().idx(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
            )?;
            let (image, sampler) = {
                let image = vk::ImmutableImage::from_iter(
                    context.allocators().memory(),
                    image_grey,
                    vk::ImageDimensions::Dim2d {
                        width: image.width(),
                        height: image.height(),
                        array_layers: 1,
                    },
                    vulkano::image::MipmapsCount::One,
                    vk::Format::R8_UNORM,
                    &mut cb,
                )?;
                context
                    .now()
                    .then_execute(context.queues().transfer().queue().clone(), cb.build()?)?
                    .then_signal_fence_and_flush()?
                    .wait(None)?;

                let view = vk::ImageView::new(
                    image.clone(),
                    vk::ImageViewCreateInfo {
                        component_mapping: vk::ComponentMapping {
                            //Red is coverage of white, with premul.
                            a: vk::ComponentSwizzle::Red,
                            r: vk::ComponentSwizzle::Red,
                            b: vk::ComponentSwizzle::Red,
                            g: vk::ComponentSwizzle::Red,
                        },
                        ..vk::ImageViewCreateInfo::from_image(&image)
                    },
                )?;

                let sampler = vk::Sampler::new(
                    context.device().clone(),
                    vk::SamplerCreateInfo {
                        min_filter: vk::Filter::Linear,
                        mag_filter: vk::Filter::Linear,
                        ..Default::default()
                    },
                )?;

                (view, sampler)
            };

            let frag = frag::load(context.device().clone())?;
            let vert = vert::load(context.device().clone())?;
            // Unwraps ok here, using GLSL where "main" is the only allowed entry point.
            let frag = frag.entry_point("main").unwrap();
            let vert = vert.entry_point("main").unwrap();

            // DualSrcBlend (~75% coverage) is used to control whether to erase or draw on a per-fragment basis
            // [1.0; 4] = draw, [0.0; 4] = erase.
            let mut premul_dyn_constants = vk::ColorBlendState::new(1);
            premul_dyn_constants.blend_constants = vk::StateMode::Fixed([1.0; 4]);
            premul_dyn_constants.attachments[0].blend = Some(vk::AttachmentBlend {
                alpha_source: vulkano::pipeline::graphics::color_blend::BlendFactor::Src1Alpha,
                color_source: vulkano::pipeline::graphics::color_blend::BlendFactor::Src1Color,
                alpha_destination:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                color_destination:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                alpha_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
                color_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
            });

            let pipeline = vk::GraphicsPipeline::start()
                .fragment_shader(frag, ())
                .vertex_shader(vert, ())
                .vertex_input_state(super::gpu_tess::interface::OutputStrokeVertex::per_vertex())
                .input_assembly_state(vk::InputAssemblyState::new()) //Triangle list, no prim restart
                .color_blend_state(premul_dyn_constants)
                .rasterization_state(vk::RasterizationState::new()) // No cull
                .viewport_state(vk::ViewportState::viewport_fixed_scissor_irrelevant([
                    vk::Viewport {
                        depth_range: 0.0..1.0,
                        dimensions: [super::DOCUMENT_DIMENSION as f32; 2],
                        origin: [0.0; 2],
                    },
                ]))
                .render_pass(
                    vulkano::pipeline::graphics::render_pass::PipelineRenderPassType::BeginRendering(
                        vulkano::pipeline::graphics::render_pass::PipelineRenderingCreateInfo {
                            view_mask: 0,
                            color_attachment_formats: vec![Some(super::DOCUMENT_FORMAT)],
                            depth_attachment_format: None,
                            stencil_attachment_format: None,
                            ..Default::default()
                        }
                    )
                )
                .build(context.device().clone())?;

            let descriptor_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0, image, sampler,
                )],
            )?;

            let tess = super::gpu_tess::GpuStampTess::new(context.clone())?;

            Ok(Self {
                context,
                pipeline,
                gpu_tess: tess,
                texture_descriptor: descriptor_set,
            })
        }
        /// Allocate a new RenderData object. Initial contents are undefined!
        pub fn uninit_render_data(&self) -> anyhow::Result<RenderData> {
            let image = vk::StorageImage::with_usage(
                self.context.allocators().memory(),
                vulkano::image::ImageDimensions::Dim2d {
                    width: super::DOCUMENT_DIMENSION,
                    height: super::DOCUMENT_DIMENSION,
                    array_layers: 1,
                },
                super::DOCUMENT_FORMAT,
                vk::ImageUsage::COLOR_ATTACHMENT | vk::ImageUsage::STORAGE,
                vk::ImageCreateFlags::empty(),
                [
                    // Todo: if these are the same queue, what happen?
                    self.context.queues().graphics().idx(),
                    self.context.queues().compute().idx(),
                ]
                .into_iter(),
            )?;
            let view = vk::ImageView::new_default(image.clone())?;

            use vulkano::VulkanObject;
            log::info!("Made render data at id{:?}", view.handle());

            Ok(RenderData { image, view })
        }
        pub fn draw(
            &self,
            strokes: &[super::ImmutableStroke],
            renderbuf: &RenderData,
            clear: bool,
        ) -> AnyResult<vk::sync::future::SemaphoreSignalFuture<impl vk::sync::GpuFuture>> {
            let (future, vertices, indirects) = self.gpu_tess.tess(strokes)?;
            let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
            )?;

            let mut matrix = cgmath::Matrix4::from_scale(2.0 / super::DOCUMENT_DIMENSION as f32);
            matrix.y *= -1.0;
            matrix.w.x -= 1.0;
            matrix.w.y += 1.0;

            command_buffer
                .begin_rendering(vulkano::command_buffer::RenderingInfo {
                    color_attachments: vec![Some(
                        vulkano::command_buffer::RenderingAttachmentInfo {
                            clear_value: if clear {
                                Some([0.0, 0.0, 0.0, 0.0].into())
                            } else {
                                None
                            },
                            load_op: if clear {
                                vulkano::render_pass::LoadOp::Clear
                            } else {
                                vulkano::render_pass::LoadOp::Load
                            },
                            store_op: vulkano::render_pass::StoreOp::Store,
                            ..vulkano::command_buffer::RenderingAttachmentInfo::image_view(
                                renderbuf.view.clone(),
                            )
                        },
                    )],
                    contents: vulkano::command_buffer::SubpassContents::Inline,
                    depth_attachment: None,
                    ..Default::default()
                })?
                .bind_pipeline_graphics(self.pipeline.clone())
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    Into::<[[f32; 4]; 4]>::into(matrix),
                )
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    self.texture_descriptor.clone(),
                )
                .bind_vertex_buffers(0, vertices)
                .draw_indirect(indirects)?
                .end_rendering()?;

            let command_buffer = command_buffer.build()?;

            // After tessellation finishes, render.
            Ok(future
                .then_execute(
                    self.context.queues().graphics().queue().clone(),
                    command_buffer,
                )?
                .then_signal_semaphore_and_flush()?)
        }
    }
}

#[derive(Clone, Debug)]
pub struct StrokeBrushSettings {
    brush: brush::BrushID,
    /// `a` is flow, NOT opacity, since the stroke is blended continuously not blended as a group.
    color_modulate: [f32; 4],
    spacing_px: f32,
    size_mul: f32,
    /// If true, the blend constants must be set to generate an erasing effect.
    is_eraser: bool,
}
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct StrokePoint {
    pos: [f32; 2],
    pressure: f32,
    /// Arc length of stroke from beginning to this point
    dist: f32,
}

impl StrokePoint {
    pub fn lerp(&self, other: &Self, factor: f32) -> Self {
        let inv_factor = 1.0 - factor;

        let s = std::simd::f32x4::from_array([self.pos[0], self.pos[1], self.pressure, self.dist]);
        let o =
            std::simd::f32x4::from_array([other.pos[0], other.pos[1], other.pressure, other.dist]);
        // FMA is planned but unimplemented ;w;
        let n = s * std::simd::f32x4::splat(inv_factor) + (o * std::simd::f32x4::splat(factor));
        Self {
            pos: [n[0], n[1]],
            pressure: n[2],
            dist: n[3],
        }
    }
}
pub struct Stroke {
    /// Unique id during this execution of the program.
    /// We have 64 bits, this won't get exhausted anytime soon! :P
    id: FuzzID<Stroke>,
    brush: StrokeBrushSettings,
    points: Vec<StrokePoint>,
}
/// Decoupled data from header, stored in separate manager. Header managed by UI.
/// Stores the strokes generated from pen input, with optional render data inserted by renderer.
pub struct StrokeLayerData {
    strokes: Vec<ImmutableStroke>,
    undo_cursor_position: Option<usize>,
}
/// Collection of layer data (stroke contents and render data) mapped from ID
pub struct StrokeLayerManager {
    layers: std::collections::HashMap<FuzzID<StrokeLayer>, StrokeLayerData>,
}
impl Default for StrokeLayerManager {
    fn default() -> Self {
        Self {
            layers: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct ImmutableStroke {
    id: FuzzID<Stroke>,
    brush: StrokeBrushSettings,
    point_collection: repositories::points::PointCollectionID,
}
impl From<Stroke> for ImmutableStroke {
    fn from(value: Stroke) -> Self {
        let point_collection = repositories::points::global()
            .insert(&value.points)
            .unwrap();
        Self {
            id: value.id,
            brush: value.brush,
            point_collection,
        }
    }
}

struct DocumentSelections {
    pub cur_layer: Option<FuzzID<StrokeLayer>>,
}
impl Default for DocumentSelections {
    fn default() -> Self {
        Self { cur_layer: None }
    }
}
struct Selections {
    pub cur_document: Option<FuzzID<Document>>,
    pub document_selections: std::collections::HashMap<FuzzID<Document>, DocumentSelections>,
    pub cur_brush: Option<brush::BrushID>,
    pub brush_settings: StrokeBrushSettings,
    pub undos: std::sync::atomic::AtomicU32,
}
impl Default for Selections {
    fn default() -> Self {
        Self {
            cur_document: None,
            document_selections: Default::default(),
            cur_brush: None,
            brush_settings: StrokeBrushSettings {
                brush: *brush::todo_brush().id(),
                color_modulate: [0.0, 0.0, 0.0, 1.0],
                size_mul: 15.0,
                spacing_px: 0.75,
                is_eraser: false,
            },
            undos: 0.into(),
        }
    }
}

// Icky. with a planned client-server architecture, we won't have as many globals -w-;;
// (well, a server is still a global, but the interface will be much less hacked-)
struct Globals {
    stroke_layers: tokio::sync::RwLock<StrokeLayerManager>,
    documents: tokio::sync::RwLock<Vec<Document>>,
    selections: tokio::sync::RwLock<Selections>,
}
impl Globals {
    fn new() -> Self {
        Self {
            stroke_layers: tokio::sync::RwLock::new(Default::default()),
            documents: tokio::sync::RwLock::new(Vec::new()),
            selections: Default::default(),
        }
    }
    fn strokes(&'_ self) -> &'_ tokio::sync::RwLock<StrokeLayerManager> {
        &self.stroke_layers
    }
    fn documents(&'_ self) -> &'_ tokio::sync::RwLock<Vec<Document>> {
        &self.documents
    }
    fn selections(&'_ self) -> &'_ tokio::sync::RwLock<Selections> {
        &self.selections
    }
}
static GLOBALS: std::sync::OnceLock<Globals> = std::sync::OnceLock::new();
pub enum RenderMessage {
    SwitchDocument(FuzzID<Document>),
    StrokeLayer {
        layer: FuzzID<StrokeLayer>,
        kind: StrokeLayerRenderMessageKind,
    },
}
pub enum StrokeLayerRenderMessageKind {
    /// The blend settings (opacity, clip, mode) for the layer were modified.
    BlendChanged(Blend),
    /// A stroke was appended to the given layer.
    /// Could be sourced from redos or new data entirely.
    Append(ImmutableStroke),
    /// This number of strokes were trucated (undone or replaced)
    Truncate(usize),
}
async fn render_worker(
    renderer: Arc<render_device::RenderContext>,
    document_preview: Arc<document_viewport_proxy::DocumentViewportPreviewProxy>,
    mut render_recv: tokio::sync::mpsc::UnboundedReceiver<RenderMessage>,
) -> AnyResult<()> {
    let layer_render = stroke_renderer::StrokeLayerRenderer::new(renderer.clone())?;
    let blend_engine = blend::BlendEngine::new(renderer.device().clone())?;

    let globals = GLOBALS.get_or_init(Globals::new);

    let mut cached_renders =
        std::collections::HashMap::<FuzzID<StrokeLayer>, stroke_renderer::RenderData>::new();
    // Keep track of layer states locally, as globals can be mutated out-of-step with render commands
    // We expect eventual consistency though!
    let mut weak_layer_strokes =
        std::collections::HashMap::<FuzzID<StrokeLayer>, Vec<ImmutableStroke>>::new();

    // An event that was peeked and rejected during aggregation.
    // Take it the next time around.
    let mut peeked = None;

    let mut document_to_draw = None;

    // Take the peeked value, or try to receive new value.
    while let Some(message) = match peeked.take() {
        Some(v) => {
            // We skip awaiting the reciever, and have more work immediately.
            // yield so that we don't starve the runtime.
            tokio::task::yield_now().await;
            Some(v)
        }
        None => render_recv.recv().await,
    } {
        // Try to aggregate more events, to batch work effectively.
        let mut events = vec![message];
        let mut skip_blend = false;
        loop {
            match render_recv.try_recv() {
                Ok(v) => {
                    // Accept value, or put into `peeked` and break if it
                    // is incompatible with aggregated work

                    // unwrap ok - vec always has at least one element.
                    let can_aggregate = match (events.last().unwrap(), &v) {
                        // Multiple edits to the same layer can be aggregated
                        (
                            RenderMessage::StrokeLayer { layer: target, .. },
                            RenderMessage::StrokeLayer { layer: new, .. },
                        ) if new == target => true,
                        // Multiple document switches can be aggregated
                        // (all are ignored but the last one)
                        (RenderMessage::SwitchDocument(..), RenderMessage::SwitchDocument(..)) => {
                            true
                        }
                        // Defer switch to next pass, but the switching document means we can skip
                        // re-rendering the document after the stroke layer commands.
                        (RenderMessage::StrokeLayer { .. }, RenderMessage::SwitchDocument(..)) => {
                            skip_blend = true;
                            false
                        }
                        _ => {
                            // Incompatible events. Defer event to next pass, and break
                            false
                        }
                    };

                    if can_aggregate {
                        events.push(v);
                    } else {
                        // peeked is always None here, as we took it above. no events lost :3
                        peeked = Some(v);
                        break;
                    }
                }
                Err(_) => {
                    // Empty or closed, we don't care - break and work
                    // on what events we managed to collect (always at least one)
                    break;
                }
            }
        }
        match events.first().unwrap() {
            // All our commands are layer commands relating to this target:
            RenderMessage::StrokeLayer { layer: target, .. } => {
                // clear the layer image before rendering? true if undone past the cached version.
                let mut clear = false;
                // keep track of rendering work to do. May include old strokes too,
                // if the truncate commands put us back in time before the cached image.
                let mut strokes_to_render = Vec::<ImmutableStroke>::new();
                let layer_strokes = weak_layer_strokes.entry(target.clone()).or_default();

                // Keep track of the blend changes. Only the last will apply.
                // None at the end of iteration means it hasn't changed - read from
                // document data directly.
                let mut new_blend = None;
                for event in events.iter() {
                    match event {
                        RenderMessage::StrokeLayer { layer, kind } => {
                            assert!(layer == target);

                            match kind {
                                StrokeLayerRenderMessageKind::Append(s) => {
                                    strokes_to_render.push(s.clone());
                                    layer_strokes.push(s.clone());
                                }
                                StrokeLayerRenderMessageKind::Truncate(num) => {
                                    let num_to_render = strokes_to_render.len();

                                    // Remove strokes from render queue
                                    strokes_to_render.drain(num_to_render.saturating_sub(*num)..);

                                    let global_strokes_truncate = layer_strokes
                                        .len()
                                        .checked_sub(*num)
                                        .expect("Cant truncate past empty");

                                    layer_strokes.drain(global_strokes_truncate..);

                                    // we took as many as possible from new commands - how many left over to remove?
                                    // if more than zero, we have to fetch strokes from globals.
                                    let num = num.saturating_sub(num_to_render);
                                    if num > 0 {
                                        // We're no longer strictly adding new content, we're removing what's
                                        // already been drawn. Clear the image.
                                        clear = true;
                                        strokes_to_render.extend(layer_strokes.iter().cloned());
                                    }
                                }
                                StrokeLayerRenderMessageKind::BlendChanged(blend) => {
                                    new_blend = Some(blend);
                                }
                            }
                        }
                        _ => panic!("Incorrect render command aggregation!"),
                    }
                }

                if !clear && strokes_to_render.is_empty() && new_blend.is_none() {
                    // No work to do. Go listen for new events.
                    continue;
                }

                let mut draw_semaphore_future = None;

                if clear && strokes_to_render.len() == 0 {
                    // Clear without remaking - just delete the data.
                    cached_renders.remove(&target);
                } else {
                    // get or try insert with - create buffer if it doesn't exist, and get a ref to it.
                    let cache_entry = match cached_renders.entry(target.clone()) {
                        std::collections::hash_map::Entry::Occupied(occupied) => {
                            &*occupied.into_mut()
                        }
                        std::collections::hash_map::Entry::Vacant(vacant) => {
                            &*vacant.insert(layer_render.uninit_render_data()?)
                        }
                    };
                    match layer_render.draw(&strokes_to_render, cache_entry, clear) {
                        Ok(semaphore) => draw_semaphore_future = Some(semaphore),
                        Err(e) => {
                            log::warn!("{e:?}");
                        }
                    };
                }

                // implicit sync: if skip_blend or document_to_draw.is_none the draw_semaphore_future will not be awaited.
                // next draw to this layer may need to be synchronized. Vulkano handles this in the background, but
                // it's an important subtlety so... note for future self!

                if !skip_blend {
                    if let Some(document_to_draw) = document_to_draw {
                        // wawawawawaaaw
                        // let blend = new_blend.unwrap();
                        // collect blend info from realtime document.
                        // need to track this locally, this info could be from wayyy in the future.
                        let blend_info: Vec<_> = {
                            let read = globals.documents.read().await;
                            let Some(doc) = read.iter().find(|doc| doc.id == document_to_draw)
                            else {
                                // Document not found, nothin to draw.
                                continue;
                            };

                            doc.layer_top_level
                                .iter()
                                .filter_map(|layer| {
                                    Some((
                                        layer.blend.clone(),
                                        cached_renders.get(&layer.id)?.view.clone(),
                                    ))
                                })
                                .collect()
                        };

                        let proxy = document_preview.write().await;
                        let commands = blend_engine.blend(
                            &renderer,
                            proxy.clone(),
                            true,
                            &blend_info,
                            [0; 2],
                            [0; 2],
                        )?;

                        let fence = match draw_semaphore_future {
                            Some(semaphore) => semaphore
                                .then_execute(
                                    renderer.queues().compute().queue().clone(),
                                    commands,
                                )?
                                .boxed_send(),
                            None => renderer
                                .now()
                                .then_execute(
                                    renderer.queues().compute().queue().clone(),
                                    commands,
                                )?
                                .boxed_send(),
                        }
                        .then_signal_fence_and_flush()?;
                        proxy.submit_with_fence(fence);
                    }
                }
            }
            // All our commands are document switch commands:
            // (we only care about the final one)
            RenderMessage::SwitchDocument(..) => {
                let RenderMessage::SwitchDocument(new_document) = events.last().unwrap() else {
                    panic!("Incorrect render command aggregation!")
                };
                document_to_draw = Some(new_document.clone());

                // <rerender viewport>
                let blend_info: Vec<_> = {
                    let read = globals.documents.read().await;
                    let Some(doc) = read.iter().find(|doc| doc.id == *new_document) else {
                        // Document not found, nothin to draw.
                        continue;
                    };

                    doc.layer_top_level
                        .iter()
                        .filter_map(|layer| {
                            Some((
                                layer.blend.clone(),
                                cached_renders.get(&layer.id)?.view.clone(),
                            ))
                        })
                        .collect()
                };

                let proxy = document_preview.write().await;
                let commands = blend_engine.blend(
                    &renderer,
                    proxy.clone(),
                    true,
                    &blend_info,
                    [0; 2],
                    [0; 2],
                )?;

                let fence = renderer
                    .now()
                    .then_execute(renderer.queues().compute().queue().clone(), commands)?
                    .boxed_send()
                    .then_signal_fence_and_flush()?;
                proxy.submit_with_fence(fence);
            }
        }
    }
    Ok(())
}
async fn stylus_event_collector(
    mut event_stream: tokio::sync::broadcast::Receiver<stylus_events::StylusEventFrame>,
    mut action_listener: actions::ActionListener,
    mut tools: pen_tools::ToolState,
    document_preview: Arc<document_viewport_proxy::DocumentViewportPreviewProxy>,
    render_send: tokio::sync::mpsc::UnboundedSender<RenderMessage>,
) -> AnyResult<()> {
    // Create a document and a few layers and select them, to speed up testing iterations :P
    /*
    let globals = GLOBALS.get_or_init(Globals::new);
    {
        let (default_document, default_layer) = {
            let mut document = Document::default();
            let document_id = document.id.weak();

            let layer = StrokeLayer::default();
            let layer_id = layer.id.weak();
            document.layer_top_level.push(layer);

            globals.documents.write().await.push(document);
            (document_id, layer_id)
        };

        let mut selections = globals.selections().write().await;
        selections.cur_document = Some(default_document);
        let document_selections = selections
            .document_selections
            .entry(default_document)
            .or_default();
        document_selections.cur_layer = Some(default_layer);
    }*/

    loop {
        match event_stream.recv().await {
            Ok(stylus_frame) => {
                // We need a transform in order to do any of our work!
                let Some(transform) = document_preview.get_view_transform().await else {
                    continue;
                };

                // Get the actions, returning if stream closed.
                let action_frame = match action_listener.frame() {
                    Ok(frame) => frame,
                    Err(e) => match e {
                        actions::ListenError::Closed => return Ok(()),
                        // Todo: this is recoverable!
                        actions::ListenError::Poisoned => todo!(),
                    },
                };

                let render = tools
                    .process(&transform, stylus_frame, &action_frame, &render_send)
                    .await;

                if let Some(transform) = render.set_view {
                    document_preview.insert_document_transform(transform).await;
                }
                document_preview.insert_cursor(render.cursor);
                document_preview.insert_tool_render(render.render_as);
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(num)) => {
                log::warn!("Lost {num} stylus frames!");
            }
            // Stream closed, no more data to handle - we're done here!
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    #[cfg(feature = "dhat_heap")]
    let _profiler = {
        log::trace!("Installed dhat");
        dhat::Profiler::new_heap()
    };
    env_logger::builder()
        .filter_level(log::LevelFilter::max())
        .init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    if let Err(e) = GlobalHotkeys::get().save() {
        log::warn!("Failed to save hotkey config:\n{e:?}");
    };

    GLOBALS.get_or_init(Globals::new);

    let (send, recv) = std::sync::mpsc::channel();
    let ui = ui::MainUI::new(send);
    std::thread::spawn(move || {
        while let Ok(recv) = recv.recv() {
            log::trace!("UI Message: {recv:?}");
        }
    });

    // Test image generators.
    //let (image, future) = make_test_image(render_context.clone())?;
    //let (image, future) = load_document_image(render_context.clone(), &std::path::PathBuf::from("/home/aspen/Pictures/thesharc.png"))?;
    //future.wait(None)?;

    let document_view = Arc::new(document_viewport_proxy::DocumentViewportPreviewProxy::new(
        &render_surface,
    )?);
    let window_renderer = window_surface.with_render_surface(
        render_surface,
        render_context.clone(),
        document_view.clone(),
        ui,
    )?;

    let event_stream = window_renderer.stylus_events();
    let action_listener = window_renderer.action_listener();

    std::thread::spawn(move || {
        #[cfg(feature = "dhat_heap")]
        // Keep alive. Winit takes ownership of main, and will never
        // drop this unless we steal it.
        let _profiler = _profiler;

        let result: Result<((), ()), anyhow::Error> = 'block: {
            let mut tools = match pen_tools::ToolState::new_from_renderer(&render_context) {
                Ok(tools) => tools,
                Err(e) => break 'block Err(e),
            };
            // We don't expect this channel to get very large, but it's important
            // that messages don't get lost under any circumstance, lest an expensive
            // document rebuild be needed :P
            let (render_sender, render_reciever) =
                tokio::sync::mpsc::unbounded_channel::<RenderMessage>();

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            // between current_thread runtime and try_join, these tasks are
            // not actually run in parallel, just interleaved. This is preferable
            // for now, just a note for future self UwU
            runtime.block_on(async {
                tokio::try_join!(
                    render_worker(render_context, document_view.clone(), render_reciever,),
                    stylus_event_collector(
                        event_stream,
                        action_listener,
                        tools,
                        document_view,
                        render_sender,
                    ),
                )
            })
        };
        if let Err(e) = result {
            log::error!("Helper task exited with err, runtime terminated:\n{e:?}")
        }
    });

    window_renderer.run();
}

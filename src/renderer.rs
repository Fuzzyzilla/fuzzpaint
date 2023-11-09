use std::sync::Arc;

use crate::vulkano_prelude::*;

type AnySemaphoreFuture = vk::sync::future::SemaphoreSignalFuture<Box<dyn GpuFuture>>;

struct PerDocumentData {
    listener: crate::commands::queue::DocumentCommandListener,
    /// Cached images of each of the nodes of the graph.
    graph_render_data: hashbrown::HashMap<crate::state::graph::AnyID, RenderData>,
}
#[derive(thiserror::Error, Debug)]
enum IncrementalDrawErr {
    #[error("{0}")]
    Anyhow(anyhow::Error),
    /// State was not usable for incremental draw.
    /// Draw from scratch instead!
    #[error("State mismatch")]
    StateMismatch,
}
enum CachedImage<'data> {
    /// The data is ready for use immediately.
    Ready(&'data RenderData),
    /// The data is currently in-flight on the graphics device, and will become ready once the provided
    /// `fence` becomes signalled.
    ReadyAfter {
        image: &'data RenderData,
        fence: &'data vk::sync::future::FenceSignalFuture<Box<dyn vk::sync::GpuFuture>>,
    },
}
impl<'data> CachedImage<'data> {
    fn data(&self) -> &'data RenderData {
        match self {
            CachedImage::Ready(data) => data,
            CachedImage::ReadyAfter { image, .. } => image,
        }
    }
}
struct Renderer {
    context: Arc<crate::render_device::RenderContext>,
    stroke_renderer: stroke_renderer::StrokeLayerRenderer,
    blend_engine: crate::blend::BlendEngine,
    data: hashbrown::HashMap<crate::state::DocumentID, PerDocumentData>,
}
impl Renderer {
    fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        Ok(Self {
            context: context.clone(),
            blend_engine: crate::blend::BlendEngine::new(context.device().clone())?,
            stroke_renderer: stroke_renderer::StrokeLayerRenderer::new(context)?,
            data: Default::default(),
        })
    }
    fn render_one(
        &mut self,
        id: crate::state::DocumentID,
        into: Arc<vk::ImageView>,
    ) -> anyhow::Result<()> {
        let data = self.data.entry(id);
        // Get the document data, and a flag for if we need to initialize that data.
        let (is_new, data) = match data {
            hashbrown::hash_map::Entry::Occupied(o) => (false, o.into_mut()),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let Some(listener) =
                    crate::default_provider().inspect(id, |queue| queue.listen_from_now())
                else {
                    // Deleted before we could do anything.
                    anyhow::bail!("Document deleted before render worker reached it");
                };
                (
                    true,
                    v.insert(PerDocumentData {
                        listener,
                        graph_render_data: Default::default(),
                    }),
                )
            }
        };
        // Forward the listener state.
        let changes = match data.listener.forward_clone_state() {
            Ok(changes) => changes,
            Err(e) => {
                // Destroy the render data, report the error.
                // Could be closed, or a thrashed document state D:
                self.data.remove(&id);
                return Err(e.into());
            }
        };
        // Render from scratch if we just created the data,
        // otherwise update from previous state.
        if is_new {
            Self::draw_from_scratch(
                &self.context,
                &self.blend_engine,
                &self.stroke_renderer,
                data,
                &changes,
                into,
            )
        } else {
            // Try to draw incrementally. If that reports it's impossible, try
            // to draw from scratch.
            match Self::draw_incremental(
                &self.context,
                &self.blend_engine,
                &self.stroke_renderer,
                data,
                &changes,
                into.clone(),
            ) {
                Err(IncrementalDrawErr::StateMismatch) => {
                    log::info!("Incremental draw failed! Retrying from scratch...");
                    Self::draw_from_scratch(
                        &self.context,
                        &self.blend_engine,
                        &self.stroke_renderer,
                        data,
                        &changes,
                        into,
                    )
                }
                Err(IncrementalDrawErr::Anyhow(anyhow)) => Err(anyhow),
                Ok(()) => Ok(()),
            }
        }
    }
    /// Draws the entire state from the beginning, ignoring the diff.
    /// Reuses allocated images, but ignores their contents!
    fn draw_from_scratch(
        context: &Arc<crate::render_device::RenderContext>,
        blend_engine: &crate::blend::BlendEngine,
        renderer: &stroke_renderer::StrokeLayerRenderer,
        document_data: &mut PerDocumentData,
        state: &impl crate::commands::queue::state_reader::CommandQueueStateReader,
        into: Arc<vk::ImageView>,
    ) -> anyhow::Result<()> {
        // Create/discard images
        Self::allocate_prune_graph(
            &renderer,
            &mut document_data.graph_render_data,
            state.graph(),
        )?;
        // Collect nodes renders
        let mut semaphores =
            Vec::<vk::sync::future::SemaphoreSignalFuture<Box<dyn GpuFuture>>>::new();
        let mut blend_infos = Vec::new();

        for (node_id, node_data) in state.graph().iter_top_level() {
            match (node_data.leaf(), node_data.node()) {
                // Render solid color image:
                (Some(crate::state::graph::LeafType::SolidColor { source, blend }), None) => {
                    let image = document_data.graph_render_data
                        .get(&node_id)
                        .ok_or_else(|| anyhow::anyhow!("Expected image to be created by allocate_prune_graph for {node_id:?}"))?;
                    // Fill image, store semaphore.
                    semaphores.push(Self::render_color(context.as_ref(), image, *source)?);
                    blend_infos.push((*blend, image.view.clone()));
                }
                // Render stroke image
                (Some(crate::state::graph::LeafType::StrokeLayer { collection, blend }), None) => {
                    let image = document_data.graph_render_data
                        .get(&node_id)
                        .ok_or_else(|| anyhow::anyhow!("Expected image to be created by allocate_prune_graph for {node_id:?}"))?;
                    let strokes = state.stroke_collections().get(*collection).ok_or_else(|| {
                        anyhow::anyhow!("Missing stroke collection {collection:?}")
                    })?;
                    let strokes: Vec<_> = strokes.iter_active().collect();
                    // Todo: strokes.strokes shouldn't be public x3
                    // and the renderer should respect stroke deletion state.
                    if strokes.is_empty() {
                        //FIXME: Renderer doesn't know how to handle zero strokes.
                        semaphores.push(Self::render_color(
                            context.as_ref(),
                            image,
                            [0.0, 0.0, 0.0, 0.0],
                        )?);
                    } else {
                        semaphores.push(renderer.draw(strokes.as_ref(), image, true)?);
                    }
                    blend_infos.push((*blend, image.view.clone()));
                }
                // Groups todo lol
                (None, Some(..)) => unimplemented!(),
                // impossible states :V
                (None, None) | (Some(..), Some(..)) => unreachable!(),
                // No other types have render work to do
                _ => (),
            }
        }
        // At this point, every layer with render data is rendering in the background, signaling the items in `semaphores`
        // Fold them all into a single semaphore future, or None if there is no rendering happenening.
        let composite = if let Some(front) = semaphores.pop() {
            // So many boxes Q^Q
            // Could group these into chunks of static size, but that's a detail uwu
            Some(
                semaphores
                    .into_iter()
                    .fold(front.boxed(), |prev, next| prev.join(next).boxed()),
            )
        } else {
            // No work to be done, no future to await!
            None
        };

        // Build blend tree
        todo!();

        // Host wait for all semaphores to finish.
        // Dirty dirty!
        if let Some(composite) = composite {
            composite.then_signal_fence_and_flush()?.wait(None)?;
        };

        Ok(())
    }
    /// Assumes the existence of a previous draw_from_scratch, applying only the diff.
    fn draw_incremental(
        _context: &Arc<crate::render_device::RenderContext>,
        _blend_engine: &crate::blend::BlendEngine,
        _renderer: &stroke_renderer::StrokeLayerRenderer,
        _document_data: &mut PerDocumentData,
        state: &impl crate::commands::queue::state_reader::CommandQueueStateReader,
        _into: Arc<vk::ImageView>,
    ) -> Result<(), IncrementalDrawErr> {
        if state.has_changes() {
            // State is dirty!
            // Lol, just defer to draw_from_scratch until that works.
            Err(IncrementalDrawErr::StateMismatch)
        } else {
            // Nothing to do
            Ok(())
        }
    }
    fn render_color(
        context: &crate::render_device::RenderContext,
        image: &RenderData,
        color: [f32; 4],
    ) -> anyhow::Result<AnySemaphoreFuture> {
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            context.allocators().command_buffer(),
            // Unfortunately transfer queue cannot clear images TwT
            // Here, graphics is more likely to be idle than compute.
            context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.clear_color_image(vk::ClearColorImageInfo {
            clear_value: color.into(),
            regions: smallvec::smallvec![image.view.subresource_range().clone()],
            image_layout: vk::ImageLayout::General,
            ..vk::ClearColorImageInfo::image(image.image.clone())
        })?;

        let command_buffer = command_buffer.build()?;

        Ok(vk::sync::now(context.device().clone())
            .then_execute(context.queues().graphics().queue().clone(), command_buffer)?
            .boxed()
            .then_signal_semaphore())
    }
    /// Creates images for all nodes which require rendering, drops node images that are deleted, etc.
    /// Only fails when graphics device is out-of-memory
    fn allocate_prune_graph(
        renderer: &stroke_renderer::StrokeLayerRenderer,
        graph_render_data: &mut hashbrown::HashMap<crate::state::graph::AnyID, RenderData>,
        graph: &crate::state::graph::BlendGraph,
    ) -> anyhow::Result<()> {
        let mut retain_data = hashbrown::HashSet::new();
        for (id, node) in graph.iter() {
            let has_graphics = match (node.leaf(), node.node()) {
                // We expect it to be a node xor leaf!
                // This is an api issue ;w;
                (Some(..), Some(..)) | (None, None) => unreachable!(),
                // Color and Stroke have images.
                // Color needing a whole image is a big ol inefficiency but that's todo :P
                (
                    Some(
                        crate::state::graph::LeafType::SolidColor { .. }
                        | crate::state::graph::LeafType::StrokeLayer { .. },
                    ),
                    None,
                ) => true,
                // Blend groups need an image.
                (None, Some(crate::state::graph::NodeType::GroupedBlend(..))) => true,
                // Every other type has no graphic.
                _ => false,
            };
            if has_graphics {
                // Mark this data as needed
                retain_data.insert(id);
                // Allocate new image, if none allocated already.
                match graph_render_data.entry(id) {
                    hashbrown::hash_map::Entry::Vacant(v) => {
                        v.insert(renderer.uninit_render_data()?);
                    }
                    _ => (),
                }
            }
        }

        // Drop all images that are no longer needed
        graph_render_data.retain(|id, _| retain_data.contains(id));

        Ok(())
    }
}
pub async fn render_worker(
    renderer: Arc<crate::render_device::RenderContext>,
    document_preview: Arc<crate::document_viewport_proxy::DocumentViewportPreviewProxy>,
    _: tokio::sync::mpsc::UnboundedReceiver<()>,
) -> anyhow::Result<()> {
    let mut change_notifier = crate::default_provider().change_notifier();
    let mut changed: Vec<_> = crate::default_provider().document_iter().collect();
    let mut renderer = Renderer::new(renderer)?;
    // Initialize renderer with all documents.
    // let _ = renderer.render(&changed);
    loop {
        use tokio::sync::broadcast::error::RecvError;
        let first_msg = change_notifier.recv().await;
        match first_msg {
            // Got message. Collect as many as are available, then go render.
            Ok(msg) => {
                changed.clear();
                changed.push(msg.id());
                while let Ok(msg) = change_notifier.try_recv() {
                    // Handle lagged? That'd be a weird failure case...
                    changed.push(msg.id());
                }
                // Implicitly handles deletion - when the renderer goes to fetch changes,
                // it will see that the document has closed.
                //renderer.render(&changed)?;
                // No current doc, skip rendering.
                let Some(selections) = crate::AdHocGlobals::read_clone() else {
                    continue;
                };
                // Rerender, if requested
                if changed.contains(&selections.document) {
                    let write = document_preview.write().await;
                    renderer.render_one(selections.document, (*write).clone())?;
                    // We sync with renderer, oofs :V
                    write.submit_now();
                }
            }
            // Messages lost. Resubscrive and check all documents for changes, to be safe.
            Err(RecvError::Lagged(..)) => {
                // Discard messages.
                change_notifier = change_notifier.resubscribe();
                // Replace with every document ID. Doing this after the
                // resubscribe is important, such that no new docs are missed!
                changed.clear();
                changed.extend(crate::default_provider().document_iter());
                // Retain here. This is a list of all docs, so any not listed
                // are therefore deleted.
                tokio::task::yield_now().await;
                //renderer.render_retain(&changed)?;
            }
            // Work here is done!
            Err(RecvError::Closed) => return Ok(()),
        }
    }
}

/// The data managed by the renderer.
/// For now, in persuit of actually getting a working product one day,
/// this is a very coarse caching sceme. In the future, perhaps a bit more granular
/// control can occur, should performance become an issue:
///  * Caching images of incrementally older states, reducing work to get to any given state (performant undo)
///  * Caching tesselation output
pub struct RenderData {
    image: Arc<vk::Image>,
    pub view: Arc<vk::ImageView>,
}
mod stroke_renderer {

    use crate::vulkano_prelude::*;
    use anyhow::Result as AnyResult;
    use std::sync::Arc;
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
        gpu_tess: crate::gpu_tess::GpuStampTess,
        pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl StrokeLayerRenderer {
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> AnyResult<Self> {
            // Begin uploading a brush image in the background while we continue setup
            let (image, sampler, _defer) = {
                let brush_image = image::open("brushes/splotch.png")?.into_luma_alpha8();

                // Iter over opacities. Weird/inefficient way to do this hehe
                // Idealy it'd just be an Alpha image but nothing suports that file layout :V
                let brush_image_alphas = brush_image.iter().skip(1).step_by(2);

                let device_image = vk::Image::new(
                    context.allocators().memory().clone(),
                    vk::ImageCreateInfo {
                        extent: [brush_image.width(), brush_image.height(), 1],
                        array_layers: 1,
                        format: vk::Format::R8_UNORM,
                        usage: vk::ImageUsage::SAMPLED | vk::ImageUsage::TRANSFER_DST,

                        ..Default::default()
                    },
                    vk::AllocationCreateInfo {
                        memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                )?;
                let image_stage = vk::Buffer::new_slice::<u8>(
                    context.allocators().memory().clone(),
                    vk::BufferCreateInfo {
                        usage: vk::BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    vk::AllocationCreateInfo {
                        memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    brush_image.width() as vk::DeviceSize * brush_image.height() as vk::DeviceSize,
                )?;
                // Write image into buffer
                {
                    // Unwrap ok - the device can't possibly be using it,
                    // and we don't read from it from host.
                    let mut write = image_stage.write().unwrap();
                    // Copy pixel-by-pixel
                    write
                        .iter_mut()
                        .zip(brush_image_alphas)
                        .for_each(|(into, from)| *into = *from);
                }
                let mut cb = vk::AutoCommandBufferBuilder::primary(
                    context.allocators().command_buffer(),
                    context.queues().transfer().idx(),
                    vk::CommandBufferUsage::OneTimeSubmit,
                )?;
                let region = vk::BufferImageCopy {
                    image_extent: device_image.extent(),
                    image_subresource: vk::ImageSubresourceLayers {
                        array_layers: 0..1,
                        aspects: vk::ImageAspects::COLOR,
                        mip_level: 0,
                    },
                    // Buffer is tightly-packed, same size as device_image.
                    ..Default::default()
                };
                cb.copy_buffer_to_image(vk::CopyBufferToImageInfo {
                    regions: smallvec::smallvec![region],
                    ..vk::CopyBufferToImageInfo::buffer_image(image_stage, device_image.clone())
                })?;
                let fence = context
                    .now()
                    .then_execute(context.queues().transfer().queue().clone(), cb.build()?)?
                    .then_signal_fence_and_flush()?;

                let view = vk::ImageView::new(
                    device_image.clone(),
                    vk::ImageViewCreateInfo {
                        component_mapping: vk::ComponentMapping {
                            //Red is coverage of white, with premul.
                            a: vk::ComponentSwizzle::Red,
                            r: vk::ComponentSwizzle::Red,
                            b: vk::ComponentSwizzle::Red,
                            g: vk::ComponentSwizzle::Red,
                        },
                        ..vk::ImageViewCreateInfo::from_image(&device_image)
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

                (
                    view,
                    sampler,
                    // synchronizing at the end of init so other setup can happen in parallel.
                    defer::defer(move || fence.wait(None).unwrap()),
                )
            };

            let frag = frag::load(context.device().clone())?;
            let vert = vert::load(context.device().clone())?;
            // Unwraps ok here, using GLSL where "main" is the only allowed entry point.
            let frag = frag.entry_point("main").unwrap();
            let vert = vert.entry_point("main").unwrap();

            let frag_stage = vk::PipelineShaderStageCreateInfo::new(frag);
            let vert_stage = vk::PipelineShaderStageCreateInfo::new(vert.clone());
            // DualSrcBlend (~75% coverage) is used to control whether to erase or draw on a per-fragment basis
            // [1.0; 4] = draw, [0.0; 4] = erase.
            let premul_dyn_constants = {
                let blend = vk::AttachmentBlend {
                    src_alpha_blend_factor: vk::BlendFactor::Src1Alpha,
                    src_color_blend_factor: vk::BlendFactor::Src1Color,
                    dst_alpha_blend_factor: vk::BlendFactor::OneMinusSrcAlpha,
                    dst_color_blend_factor: vk::BlendFactor::OneMinusSrcAlpha,
                    alpha_blend_op: vk::BlendOp::Add,
                    color_blend_op: vk::BlendOp::Add,
                };
                let blend_states = vk::ColorBlendAttachmentState {
                    blend: Some(blend),
                    ..Default::default()
                };
                vk::ColorBlendState::with_attachment_states(1, blend_states)
            };

            let matrix_push_constant = vk::PushConstantRange {
                offset: 0,
                stages: vk::ShaderStages::VERTEX,
                size: std::mem::size_of::<vert::Matrix>() as u32,
            };

            let image_sampler_layout = vk::DescriptorSetLayout::new(
                context.device().clone(),
                vk::DescriptorSetLayoutCreateInfo {
                    bindings: [(
                        0,
                        vk::DescriptorSetLayoutBinding {
                            descriptor_count: 1,
                            stages: vk::ShaderStages::FRAGMENT,
                            ..vk::DescriptorSetLayoutBinding::descriptor_type(
                                vk::DescriptorType::CombinedImageSampler,
                            )
                        },
                    )]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                },
            )?;

            let layout = vk::PipelineLayout::new(
                context.device().clone(),
                vk::PipelineLayoutCreateInfo {
                    push_constant_ranges: vec![matrix_push_constant],
                    set_layouts: vec![image_sampler_layout],
                    ..Default::default()
                },
            )?;

            let pipeline = vk::GraphicsPipeline::new(
                context.device().clone(),
                None,
                vk::GraphicsPipelineCreateInfo {
                    color_blend_state: Some(premul_dyn_constants),
                    input_assembly_state: Some(vk::InputAssemblyState {
                        topology: vk::PrimitiveTopology::TriangleList,
                        primitive_restart_enable: false,
                        ..Default::default()
                    }),
                    multisample_state: Some(Default::default()),
                    rasterization_state: Some(vk::RasterizationState {
                        cull_mode: vk::CullMode::None,
                        ..Default::default()
                    }),
                    vertex_input_state: Some(
                        crate::gpu_tess::interface::OutputStrokeVertex::per_vertex()
                            .definition(&vert.info().input_interface)?,
                    ),
                    viewport_state: Some(vk::ViewportState {
                        viewports: smallvec::smallvec![vk::Viewport {
                            depth_range: 0.0..=1.0,
                            extent: [crate::DOCUMENT_DIMENSION as f32; 2],
                            offset: [0.0; 2],
                        }],
                        ..Default::default()
                    }),
                    subpass: Some(vk::PipelineSubpassType::BeginRendering(
                        vk::PipelineRenderingCreateInfo {
                            color_attachment_formats: vec![Some(crate::DOCUMENT_FORMAT)],
                            ..Default::default()
                        },
                    )),
                    stages: smallvec::smallvec![vert_stage, frag_stage,],
                    ..vk::GraphicsPipelineCreateInfo::layout(layout)
                },
            )?;
            /*start()
            .fragment_shader(frag, ())
            .vertex_shader(vert, ())
            .vertex_input_state(crate::gpu_tess::interface::OutputStrokeVertex::per_vertex())
            .input_assembly_state(vk::InputAssemblyState::new()) //Triangle list, no prim restart
            .color_blend_state(premul_dyn_constants)
            .rasterization_state(vk::RasterizationState::new()) // No cull
            .viewport_state(vk::ViewportState::viewport_fixed_scissor_irrelevant([
                vk::Viewport {
                    depth_range: 0.0..1.0,
                    dimensions: [crate::DOCUMENT_DIMENSION as f32; 2],
                    origin: [0.0; 2],
                },
            ]))
            .render_pass(
            )
            .build(context.device().clone())?;*/

            let descriptor_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0, image, sampler,
                )],
                [],
            )?;

            let tess = crate::gpu_tess::GpuStampTess::new(context.clone())?;

            Ok(Self {
                context,
                pipeline,
                gpu_tess: tess,
                texture_descriptor: descriptor_set,
            })
        }
        /// Allocate a new RenderData object. Initial contents are undefined!
        pub fn uninit_render_data(&self) -> anyhow::Result<super::RenderData> {
            let image = vk::Image::new(
                self.context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    usage: vk::ImageUsage::COLOR_ATTACHMENT
                        | vk::ImageUsage::STORAGE
                        // For color clearing
                        | vk::ImageUsage::TRANSFER_DST,
                    extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                    array_layers: 1,
                    mip_levels: 1,
                    sharing: vk::Sharing::Concurrent(smallvec::smallvec![
                        self.context.queues().graphics().idx(),
                        self.context.queues().compute().idx()
                    ]),
                    format: crate::DOCUMENT_FORMAT,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?;
            let view = vk::ImageView::new_default(image.clone())?;

            use vulkano::VulkanObject;
            log::info!("Made render data at id{:?}", view.handle());

            Ok(super::RenderData { image, view })
        }
        pub fn draw(
            &self,
            strokes: &[&crate::state::stroke_collection::ImmutableStroke],
            renderbuf: &super::RenderData,
            clear: bool,
        ) -> AnyResult<vk::sync::future::SemaphoreSignalFuture<Box<dyn vk::sync::GpuFuture>>>
        {
            let (future, vertices, indirects) = self.gpu_tess.tess(strokes)?;

            let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;

            let mut matrix = cgmath::Matrix4::from_scale(2.0 / crate::DOCUMENT_DIMENSION as f32);
            matrix.y *= -1.0;
            matrix.w.x -= 1.0;
            matrix.w.y += 1.0;

            log::trace!(
                "Drawing {} vertices from {} indirects",
                vertices.len(),
                indirects.len()
            );

            command_buffer
                .begin_rendering(vk::RenderingInfo {
                    color_attachments: vec![Some(vk::RenderingAttachmentInfo {
                        clear_value: if clear {
                            Some([0.0, 0.0, 0.0, 0.0].into())
                        } else {
                            None
                        },
                        load_op: if clear {
                            vk::AttachmentLoadOp::Clear
                        } else {
                            vk::AttachmentLoadOp::Load
                        },
                        store_op: vk::AttachmentStoreOp::Store,
                        ..vk::RenderingAttachmentInfo::image_view(renderbuf.view.clone())
                    })],
                    contents: vk::SubpassContents::Inline,
                    depth_attachment: None,
                    ..Default::default()
                })?
                .bind_pipeline_graphics(self.pipeline.clone())?
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    Into::<[[f32; 4]; 4]>::into(matrix),
                )?
                .bind_descriptor_sets(
                    vk::PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    self.texture_descriptor.clone(),
                )?
                .bind_vertex_buffers(0, vertices)?
                .draw_indirect(indirects)?
                .end_rendering()?;

            let command_buffer = command_buffer.build()?;

            // After tessellation finishes, render.
            Ok(future
                .then_execute(
                    self.context.queues().graphics().queue().clone(),
                    command_buffer,
                )?
                .boxed()
                .then_signal_semaphore())
        }
    }
}

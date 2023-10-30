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
    /// Get the cached document image for the given ID, if found.
    fn get_cached(&self, id: crate::state::DocumentID) -> Option<CachedImage> {
        None
        /*self.data
        .get(&id)
        .map(|data| CachedImage::Ready(&data.root_image))*/
    }
    /*
    fn display(
        &self,
        id: crate::state::DocumentID,
        into: Arc<vk::StorageImage>,
    ) -> anyhow::Result<vk::sync::future::FenceSignalFuture<Box<dyn GpuFuture + Send>>> {
        let image = self
            .get_cached(id)
            .ok_or_else(|| anyhow::anyhow!("No image available for {id:?}"))?;

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().compute().idx(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        command_buffer.copy_image(vulkano::command_buffer::CopyImageInfo::images(
            image.data().image.clone(),
            into,
        ))?;
        vk::sync::now(self.context.device().clone()).then_execute(queue, command_buffer);

        todo!()
    }*/
    /// Same as `render`, but drops any associated data not included in `retain` before rendering the included ids.
    fn render_retain(&mut self, retain: &[crate::state::DocumentID]) -> anyhow::Result<()> {
        self.data.retain(|k, _| retain.contains(k));
        self.render(retain)
    }
    /// Checks the given document IDs for changes, rendering those changes.
    /// Will try all changes, ignoring errors. Returns the first error that occured,
    /// if any.
    fn render(&mut self, changes: &[crate::state::DocumentID]) -> anyhow::Result<()> {
        /*
        let mut err = None;
        for change in changes {
            err = err.or(self.render_one(*change).err());
        }
        match err {
            None => Ok(()),
            Some(err) => Err(err),
        }*/
        todo!()
    }
    fn render_one(
        &mut self,
        id: crate::state::DocumentID,
        into: Arc<vk::ImageView<vk::StorageImage>>,
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
        into: Arc<vk::ImageView<vk::StorageImage>>,
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
                    // Todo: strokes.strokes shouldn't be public x3
                    // and the renderer should respect stroke deletion state.
                    semaphores.push(renderer.draw(&strokes.strokes, image, true)?);
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

        // Build blend commands
        let blend_commands =
            blend_engine.blend(&context, into, true, &blend_infos[..], [0; 2], [0; 2])?;

        // After all the semaphores finish, execute the blend and signal a fence.
        // Or, if no sempahores, execute immediately. (It will just be a clear.)
        let fence = if let Some(composite) = composite {
            composite.boxed()
        } else {
            vk::sync::now(context.device().clone()).boxed()
        }
        .then_execute(context.queues().compute().queue().clone(), blend_commands)?
        .then_signal_fence_and_flush()?;

        fence.wait(None)?;

        Ok(())
    }
    /// Assumes the existence of a previous draw_from_scratch, applying only the diff.
    fn draw_incremental(
        context: &Arc<crate::render_device::RenderContext>,
        blend_engine: &crate::blend::BlendEngine,
        renderer: &stroke_renderer::StrokeLayerRenderer,
        document_data: &mut PerDocumentData,
        state: &impl crate::commands::queue::state_reader::CommandQueueStateReader,
        into: Arc<vk::ImageView<vk::StorageImage>>,
    ) -> Result<(), IncrementalDrawErr> {
        if state.has_changes() {
            // State is dirty!
            // Lol, just defer to draw_from_scratch until that works.
            Self::draw_from_scratch(context, blend_engine, renderer, document_data, state, into)
                .map_err(|err| IncrementalDrawErr::Anyhow(err))
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
        use vulkano::image::ImageViewAbstract;
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            context.allocators().command_buffer(),
            // Unfortunately transfer queue cannot clear images TwT
            // Here, graphics is more likely to be idle than compute.
            context.queues().graphics().idx(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.clear_color_image(vulkano::command_buffer::ClearColorImageInfo {
            clear_value: color.into(),
            regions: smallvec::smallvec![image.view.subresource_range().clone()],
            image_layout: vulkano::image::ImageLayout::General,
            ..vulkano::command_buffer::ClearColorImageInfo::image(image.image.clone())
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
    let _ = renderer.render(&changed);
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
                tokio::task::yield_now().await;
                renderer.render(&changed)?;
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
                renderer.render_retain(&changed)?;
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
    image: Arc<vk::StorageImage>,
    pub view: Arc<vk::ImageView<vk::StorageImage>>,
}
mod stroke_renderer {

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
        gpu_tess: crate::gpu_tess::GpuStampTess,
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
                    vulkano::pipeline::graphics::render_pass::PipelineRenderPassType::BeginRendering(
                        vulkano::pipeline::graphics::render_pass::PipelineRenderingCreateInfo {
                            view_mask: 0,
                            color_attachment_formats: vec![Some(crate::DOCUMENT_FORMAT)],
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
            let image = vk::StorageImage::with_usage(
                self.context.allocators().memory(),
                vulkano::image::ImageDimensions::Dim2d {
                    width: crate::DOCUMENT_DIMENSION,
                    height: crate::DOCUMENT_DIMENSION,
                    array_layers: 1,
                },
                crate::DOCUMENT_FORMAT,
                vk::ImageUsage::COLOR_ATTACHMENT
                    | vk::ImageUsage::STORAGE
                    // For color clearing
                    | vk::ImageUsage::TRANSFER_DST,
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

            Ok(super::RenderData { image, view })
        }
        pub fn draw(
            &self,
            strokes: &[crate::state::stroke_collection::ImmutableStroke],
            renderbuf: &super::RenderData,
            clear: bool,
        ) -> AnyResult<vk::sync::future::SemaphoreSignalFuture<Box<dyn vk::sync::GpuFuture>>>
        {
            let (future, vertices, indirects) = self.gpu_tess.tess(strokes)?;
            let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
            )?;

            let mut matrix = cgmath::Matrix4::from_scale(2.0 / crate::DOCUMENT_DIMENSION as f32);
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
                .boxed()
                .then_signal_semaphore())
        }
    }
}

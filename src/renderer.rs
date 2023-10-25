struct PerDocumentData {
    listener: crate::commands::queue::DocumentCommandListener,
}
struct Renderer {
    data: hashbrown::HashMap<crate::state::DocumentID, PerDocumentData>,
}
impl Renderer {
    fn new(renderer: std::sync::Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        Ok(Self {
            data: Default::default(),
        })
    }
    /// Same as `render`, but drops any associated data not included in `retain` before rendering the included ids.
    fn render_retain(&mut self, retain: &[crate::state::DocumentID]) -> anyhow::Result<()> {
        self.data.retain(|k, _| retain.contains(k));
        self.render(retain)
    }
    /// Checks the given document IDs for changes, rendering those changes.
    fn render(&mut self, changes: &[crate::state::DocumentID]) -> anyhow::Result<()> {
        for change in changes {
            self.render_one(*change)?;
        }
        Ok(())
    }
    fn render_one(&mut self, id: crate::state::DocumentID) -> anyhow::Result<()> {
        let data = self.data.entry(id);
        let data = match data {
            hashbrown::hash_map::Entry::Occupied(o) => o.into_mut(),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let Some(listener) =
                    crate::default_provider().inspect(id, |queue| queue.listen_from_now())
                else {
                    // Deleted before we could do anything. Not an error!
                    return Ok(());
                };
                v.insert(PerDocumentData { listener })
            }
        };
        let Ok(changes) = data.listener.forward_clone_state() else {
            // Failed to read the state. Closed or broke!
            // Cleanup and bail.
            self.data.remove(&id);
            return Ok(());
        };

        todo!()
    }
}
pub async fn render_worker(
    renderer: std::sync::Arc<crate::render_device::RenderContext>,
    _document_preview: std::sync::Arc<crate::document_viewport_proxy::DocumentViewportPreviewProxy>,
    _: tokio::sync::mpsc::UnboundedReceiver<()>,
) -> anyhow::Result<()> {
    let mut change_notifier = crate::default_provider().change_notifier();
    let mut changed = vec![];
    let mut renderer = Renderer::new(renderer)?;
    loop {
        use tokio::sync::broadcast::error::RecvError;
        match change_notifier.recv().await {
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
        pub fn uninit_render_data(&self) -> anyhow::Result<RenderData> {
            let image = vk::StorageImage::with_usage(
                self.context.allocators().memory(),
                vulkano::image::ImageDimensions::Dim2d {
                    width: crate::DOCUMENT_DIMENSION,
                    height: crate::DOCUMENT_DIMENSION,
                    array_layers: 1,
                },
                crate::DOCUMENT_FORMAT,
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
            strokes: &[crate::state::stroke_collection::ImmutableStroke],
            renderbuf: &RenderData,
            clear: bool,
        ) -> AnyResult<vk::sync::future::SemaphoreSignalFuture<impl vk::sync::GpuFuture>> {
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
                .then_signal_semaphore_and_flush()?)
        }
    }
}

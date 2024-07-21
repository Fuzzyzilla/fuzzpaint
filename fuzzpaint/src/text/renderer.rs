use super::{interface, DrawOutput, OutputColor, SizeClass};
use crate::vulkano_prelude::*;

/// Upload instances and indirects. Ok(None) if either is empty.
fn predraw_upload(
    ctx: &crate::render_device::RenderContext,
    output: &super::DrawOutput,
) -> anyhow::Result<
    Option<(
        vk::Subbuffer<[interface::Instance]>,
        vk::Subbuffer<[vk::DrawIndexedIndirectCommand]>,
    )>,
> {
    if output.instances.is_empty() || output.indirects.is_empty() {
        Ok(None)
    } else {
        let instances_len_bytes = std::mem::size_of_val(output.instances.as_slice()) as u64;
        let indirects_len_bytes = std::mem::size_of_val(output.indirects.as_slice()) as u64;
        let scratch_size = instances_len_bytes + indirects_len_bytes;

        let align = std::mem::align_of_val(output.instances.as_slice());
        // make sure align requirements are sound between the two bufs
        assert!(align >= std::mem::align_of_val(output.indirects.as_slice()));

        let scratch_buffer = vk::Buffer::new(
            ctx.allocators().memory().clone(),
            vk::BufferCreateInfo {
                sharing: vulkano::sync::Sharing::Exclusive,
                // We make a host accessible instance buffer... this kinda sucks!
                usage: vk::BufferUsage::INDIRECT_BUFFER | vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vulkano::memory::allocator::DeviceLayout::new(
                scratch_size
                    .try_into()
                    // Guarded by if, not empty!
                    .unwrap(),
                // Std guarantees power of two
                vulkano::memory::DeviceAlignment::new(align as u64).unwrap(),
                // Unwrap ok - it fits in host mem just fine (<= device address space), the size won't overflow.
            )
            .unwrap(),
        )?;
        // Slices ok - both known non-zero by if guard
        let instance_buffer = vk::Subbuffer::new(scratch_buffer.clone())
            .slice(0..instances_len_bytes)
            .reinterpret::<[super::interface::Instance]>();
        {
            // We just made it - no concurrent access.
            let mut write = instance_buffer.write().unwrap();
            write.copy_from_slice(&output.instances);
        }
        let indirect_buffer = vk::Subbuffer::new(scratch_buffer)
            // Align is OK - we ickily checked that the aligns of the two types are compatible
            .slice(instances_len_bytes..)
            .reinterpret::<[vk::DrawIndexedIndirectCommand]>();
        {
            // We just made it - no concurrent access.
            let mut write = indirect_buffer.write().unwrap();
            write.copy_from_slice(&output.indirects);
        }

        Ok(Some((instance_buffer, indirect_buffer)))
    }
}
fn sample_count_to_size_class(count: vk::SampleCount) -> SizeClass {
    use vk::SampleCount;
    // Spacial resolution multiplier is sqrt(sample count).
    // To convert to SizeClass, take the log2 of this, ceil.
    let log2_sqrt_samples = match count {
        SampleCount::Sample1 => 0,
        SampleCount::Sample2 | SampleCount::Sample4 => 1,
        SampleCount::Sample8 | SampleCount::Sample16 => 2,
        SampleCount::Sample32 | SampleCount::Sample64 => 3,
        // We should never be able to choose a sample count we can't even observe!
        _ => unimplemented!("unknown sample count"),
    };

    super::SizeClass::from_exp_lossy(log2_sqrt_samples)
}

pub mod monochrome {
    use crate::vulkano_prelude::*;
    use std::sync::Arc;
    mod shaders {
        pub mod msaa {
            //! Write glyphs into monochrome MSAA buff
            pub mod vert {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 460
                        layout(push_constant) uniform Matrix {
                            mat4 mvp;
                        };
                        // Per-vertex
                        layout(location = 0) in vec2 position;        // arbitrary font units (not em)
                        // Per-instance (per-glyph)
                        layout(location = 1) in ivec2 glyph_position; // arbitrary font units (not em)
                        layout(location = 2) in vec4 glyph_color;     // monochrome pipeline - dontcare!

                        void main() {
                            gl_Position = mvp * vec4(position + vec2(glyph_position), 0.0, 1.0);
                        }
                    "
                }
            }
            pub mod frag {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 460
                        // Rendering into R8_UNORM
                        // Cleared to zero, frags set to one.
                        layout(location = 0) out float color;
                        void main() {
                            color = 1.0;
                        }
                    "
                }
            }
        }
        pub mod colorize {
            //! Take rendered fragments and apply color to them in a tiled-friendly manner
            pub mod vert {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                            #version 460
    
                            void main() {
                                // fullscreen tri
                                gl_Position = vec4(
                                    float((gl_VertexIndex & 1) * 4 - 1),
                                    float((gl_VertexIndex & 2) * 2 - 1),
                                    0.0,
                                    1.0
                                );
                            }
                        "
                }
            }
            pub mod frag {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                            #version 460
                            // R8_UNORM from resolved prepass
                            layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput in_coverage;
                            layout(push_constant) uniform Color {
                                /// Premultipled color, directly multiplied by coverage.
                                vec4 modulate;
                            };
    
                            layout(location = 0) out vec4 color;
    
                            void main() {
                                color = modulate * subpassLoad(in_coverage).rrrr;
                            }
                        "
                }
            }
        }
    }

    pub struct Renderer {
        context: Arc<crate::render_device::RenderContext>,
        renderpass: Arc<vk::RenderPass>,
        /// Draws the tris into MSAA buff
        msaa_pipeline: Arc<vk::GraphicsPipeline>,
        /// MSAA buffer for antialaised rendering, immediately resolved to a regular image.
        /// (thus, can be transient attatchment)
        multisample: Arc<vk::ImageView>,
        /// Resolve target for MSAA, also transient. Used as input for a coloring stage.
        resolve: Arc<vk::ImageView>,
        resolve_input_set: Arc<vk::PersistentDescriptorSet>,
        /// Colors resolved greyscale image into color image
        colorize_pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl Renderer {
        /// Get the internal scale factor due to multisampling.
        /// Should be multiplied by the tessellation factor prior to building for best looking results.
        #[must_use]
        pub fn internal_size_class(&self) -> super::SizeClass {
            super::sample_count_to_size_class(self.multisample.image().samples())
        }
        pub fn make_renderpass(
            device: Arc<vk::Device>,
            format: vk::Format,
            samples: vk::SampleCount,
        ) -> anyhow::Result<Arc<vk::RenderPass>> {
            use vulkano::render_pass::{
                AttachmentDescription, AttachmentReference, RenderPass, RenderPassCreateInfo,
                SubpassDependency, SubpassDescription,
            };
            use vulkano::sync::{AccessFlags, DependencyFlags, PipelineStages};

            // Vulkano doesn't support this usecase of transient images, so we do it ourselves >:3c
            let attachments = vec![
                // First render attach. Transient. Clear, dontcare for store.
                AttachmentDescription {
                    load_op: vk::AttachmentLoadOp::Clear,
                    store_op: vk::AttachmentStoreOp::DontCare,
                    format,
                    samples,
                    initial_layout: vk::ImageLayout::Undefined,
                    final_layout: vk::ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                },
                // Second attach, for resolving first. Transient input for final stage. dontcare for load/store.
                AttachmentDescription {
                    load_op: vk::AttachmentLoadOp::DontCare,
                    store_op: vk::AttachmentStoreOp::DontCare,
                    format,
                    samples: vk::SampleCount::Sample1,
                    initial_layout: vk::ImageLayout::Undefined,
                    final_layout: vk::ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                },
                // Output color, user provided.
                AttachmentDescription {
                    // We overwrite every texel, dontcare for load.
                    load_op: vk::AttachmentLoadOp::DontCare,
                    store_op: vk::AttachmentStoreOp::Store,
                    format: crate::DOCUMENT_FORMAT,
                    samples: vk::SampleCount::Sample1,
                    // We clear it anyway, don't mind the initial layout.
                    initial_layout: vk::ImageLayout::Undefined,
                    final_layout: vk::ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                },
            ];
            let attachment_references = [
                AttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::ColorAttachmentOptimal,
                    stencil_layout: None,
                    ..Default::default()
                },
                // Pre-resolve:
                AttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::ColorAttachmentOptimal,
                    stencil_layout: None,
                    ..Default::default()
                },
                // Post-resolve:
                AttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                    stencil_layout: None,
                    aspects: vk::ImageAspects::COLOR,
                    ..Default::default()
                },
                AttachmentReference {
                    attachment: 2,
                    layout: vk::ImageLayout::ColorAttachmentOptimal,
                    stencil_layout: None,
                    ..Default::default()
                },
            ];

            let subpasses = vec![
                SubpassDescription {
                    color_attachments: vec![Some(attachment_references[0].clone())],
                    // Afterwards, resolve into other transient attachment.
                    color_resolve_attachments: vec![Some(attachment_references[1].clone())],
                    ..Default::default()
                },
                SubpassDescription {
                    color_attachments: vec![Some(attachment_references[3].clone())],
                    input_attachments: vec![Some(attachment_references[2].clone())],
                    ..Default::default()
                },
            ];
            let dependencies = vec![SubpassDependency {
                src_subpass: Some(0),
                dst_subpass: Some(1),
                // Second subpass only cares about pixel-local info, so individual tiles can move on asynchronously.
                dependency_flags: DependencyFlags::BY_REGION,
                // After we resolve...
                src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                // Then we can read from fragment...
                dst_access: AccessFlags::INPUT_ATTACHMENT_READ,
                dst_stages: PipelineStages::FRAGMENT_SHADER,
                ..Default::default()
            }];

            RenderPass::new(
                device,
                RenderPassCreateInfo {
                    attachments,
                    subpasses,
                    dependencies,
                    ..Default::default()
                },
            )
            .map_err(Into::into)
        }
        fn make_images_for(
            context: &crate::render_device::RenderContext,
            format: vk::Format,
        ) -> anyhow::Result<(Arc<vk::ImageView>, Arc<vk::ImageView>)> {
            // Prevent absurd sample counts for deminishing returns.
            // Todo: polyfill this when not available on this hardware for consistent visuals across devices.
            let valid_samples = vk::SampleCounts::SAMPLE_2
                | vk::SampleCounts::SAMPLE_4
                | vk::SampleCounts::SAMPLE_8;
            let samples = context
                .physical_device()
                .image_format_properties(vulkano::image::ImageFormatInfo {
                    format,
                    tiling: vulkano::image::ImageTiling::Optimal,
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT | vk::ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                })?
                .ok_or_else(|| anyhow::anyhow!("unsupported image configuration"))?
                .sample_counts
                .intersection(valid_samples)
                .max_count();
            if samples == vk::SampleCount::Sample1 {
                // The failure path here involves a whole different pipeline :V
                anyhow::bail!("msaa unsupported")
            }
            let multisample = vk::Image::new(
                context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    format,
                    samples,
                    // Don't need long-lived data. Render, and resolve source.
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT | vk::ImageUsage::COLOR_ATTACHMENT,
                    // Todo: query whether this is supported
                    extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                    sharing: vk::Sharing::Exclusive,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?;
            let resolve = vk::Image::new(
                context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    format,
                    // Don't need long-lived data. resolve destination and input.
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT
                        | vk::ImageUsage::COLOR_ATTACHMENT
                        | vk::ImageUsage::INPUT_ATTACHMENT,
                    // Todo: query whether this is supported
                    extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                    sharing: vk::Sharing::Exclusive,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?;

            Ok((
                vk::ImageView::new_default(multisample)?,
                vk::ImageView::new_default(resolve)?,
            ))
        }
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
            let (multisample, resolve) =
                Self::make_images_for(context.as_ref(), vk::Format::R8_UNORM)?;
            let samples = multisample.image().samples();
            // Tiling renderer shenanigans, mostly for practice lol
            // Forms a per-fragment pipe from MSAA -> Resolve -> Color
            let renderpass =
                Self::make_renderpass(context.device().clone(), multisample.format(), samples)?;

            let mut pass = renderpass.clone().first_subpass();
            let msaa_pipeline =
                Self::make_msaa_pipe(context.device().clone(), pass.clone(), samples)?;

            pass.next_subpass();
            let (colorize_pipeline, resolve_input_set) =
                Self::make_colorize_pipe(context.as_ref(), pass, resolve.clone())?;

            Ok(Self {
                context,
                renderpass,
                msaa_pipeline,
                multisample,
                resolve,
                resolve_input_set,
                colorize_pipeline,
            })
        }
        /// Make a framebuffer compatible with `self.monochrome_renderpass`
        fn make_framebuffer(
            &self,
            render_into: Arc<vk::ImageView>,
        ) -> anyhow::Result<Arc<vk::Framebuffer>> {
            vk::Framebuffer::new(
                self.renderpass.clone(),
                vk::FramebufferCreateInfo {
                    attachments: vec![self.multisample.clone(), self.resolve.clone(), render_into],
                    ..Default::default()
                },
            )
            .map_err(Into::into)
        }
        fn make_msaa_pipe(
            device: Arc<vk::Device>,
            subpass: vk::Subpass,
            samples: vk::SampleCount,
        ) -> anyhow::Result<Arc<vk::GraphicsPipeline>> {
            let vert = shaders::msaa::vert::load(device.clone())?;
            let vert = vert.entry_point("main").unwrap();
            let frag = shaders::msaa::frag::load(device.clone())?;
            let frag = frag.entry_point("main").unwrap();

            let vertex_description = [
                super::interface::Vertex::per_vertex(),
                super::interface::Instance::per_instance(),
            ]
            .definition(&vert.info().input_interface)?;

            let stages = smallvec::smallvec![
                vk::PipelineShaderStageCreateInfo::new(vert),
                vk::PipelineShaderStageCreateInfo::new(frag),
            ];

            let matrix_range = vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<shaders::msaa::vert::Matrix>()
                    .try_into()
                    .unwrap(),
                stages: vk::ShaderStages::VERTEX,
            };

            let layout = vk::PipelineLayout::new(
                device.clone(),
                vk::PipelineLayoutCreateInfo {
                    push_constant_ranges: vec![matrix_range],
                    ..Default::default()
                },
            )?;

            vk::GraphicsPipeline::new(
                device,
                None,
                vk::GraphicsPipelineCreateInfo {
                    stages,
                    vertex_input_state: Some(vertex_description),
                    input_assembly_state: Some(vk::InputAssemblyState::default()),
                    color_blend_state: Some(vk::ColorBlendState::with_attachment_states(
                        1,
                        vk::ColorBlendAttachmentState::default(),
                    )),
                    rasterization_state: Some(vk::RasterizationState::default()),
                    subpass: Some(subpass.into()),
                    multisample_state: Some(vk::MultisampleState {
                        rasterization_samples: samples,
                        ..Default::default()
                    }),
                    // Viewport dynamic, scissor irrelevant.
                    viewport_state: Some(vk::ViewportState::default()),
                    dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                    ..vk::GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .map_err(Into::into)
        }
        fn make_colorize_pipe(
            context: &crate::render_device::RenderContext,
            subpass: vk::Subpass,
            resolve_input_image: Arc<vk::ImageView>,
        ) -> anyhow::Result<(Arc<vk::GraphicsPipeline>, Arc<vk::PersistentDescriptorSet>)> {
            use vulkano::descriptor_set::DescriptorSet;
            let device = context.device();
            let vert = shaders::colorize::vert::load(device.clone())?;
            let vert = vert.entry_point("main").unwrap();
            let frag = shaders::colorize::frag::load(device.clone())?;
            let frag = frag.entry_point("main").unwrap();

            // No input.
            let vertex_description = vk::VertexInputState::new();

            let stages = smallvec::smallvec![
                vk::PipelineShaderStageCreateInfo::new(vert),
                vk::PipelineShaderStageCreateInfo::new(frag),
            ];

            let color_range = vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<shaders::colorize::frag::Color>()
                    .try_into()
                    .unwrap(),
                stages: vk::ShaderStages::FRAGMENT,
            };

            let resolve_input_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                vk::DescriptorSetLayout::new(
                    device.clone(),
                    vk::DescriptorSetLayoutCreateInfo {
                        bindings: [(
                            0,
                            vk::DescriptorSetLayoutBinding {
                                stages: vk::ShaderStages::FRAGMENT,
                                ..vk::DescriptorSetLayoutBinding::descriptor_type(
                                    vk::DescriptorType::InputAttachment,
                                )
                            },
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )?,
                [vk::WriteDescriptorSet::image_view(0, resolve_input_image)],
                [],
            )?;

            let layout = vk::PipelineLayout::new(
                device.clone(),
                vk::PipelineLayoutCreateInfo {
                    set_layouts: vec![resolve_input_set.layout().clone()],
                    push_constant_ranges: vec![color_range],
                    ..Default::default()
                },
            )?;

            Ok((
                vk::GraphicsPipeline::new(
                    device.clone(),
                    None,
                    vk::GraphicsPipelineCreateInfo {
                        stages,
                        vertex_input_state: Some(vertex_description),
                        input_assembly_state: Some(vk::InputAssemblyState::default()),
                        color_blend_state: Some(vk::ColorBlendState::with_attachment_states(
                            1,
                            vk::ColorBlendAttachmentState::default(),
                        )),
                        rasterization_state: Some(vk::RasterizationState::default()),
                        multisample_state: Some(vk::MultisampleState::default()),
                        subpass: Some(subpass.into()),
                        // Viewport dynamic, scissor irrelevant.
                        viewport_state: Some(vk::ViewportState::default()),
                        dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                        ..vk::GraphicsPipelineCreateInfo::layout(layout)
                    },
                )?,
                resolve_input_set,
            ))
        }
        /// Create commands to render `DrawOutput` into the given image.
        ///
        /// `xform` should be a combined view+proj matrix, tranforming face's arbitrary units (NOT em) to NDC.
        /// See [`rustybuzz::ttf_parser::Face::units_per_em`].
        ///
        /// `self` is in exclusive use through the duration of the returned buffer's execution.
        pub fn draw(
            &self,
            xform: ultraviolet::Mat4,
            render_into: Arc<vk::ImageView>,
            output: &super::DrawOutput,
        ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
            let monochrome_color = match output.color_mode {
                super::OutputColor::Solid(s) => s,
                super::OutputColor::PerInstance => {
                    anyhow::bail!("cannot use monochrome pipeline for per-instance colored text")
                }
            };
            // Even if none to draw, we can still clear and return.
            let instances_indirects = super::predraw_upload(&self.context, output)?;

            let mut commands = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;

            if let Some((instances, indirects)) = instances_indirects {
                let framebuffer = self.make_framebuffer(render_into)?;
                let viewport = vk::Viewport {
                    depth_range: 0.0..=1.0,
                    offset: [0.0; 2],
                    extent: [
                        framebuffer.extent()[0] as f32,
                        framebuffer.extent()[1] as f32,
                    ],
                };

                commands
                    .begin_render_pass(
                        vk::RenderPassBeginInfo {
                            // Start is cleared, other two are dontcare (immediately overwritten)
                            clear_values: vec![Some([0.0; 4].into()), None, None],
                            ..vk::RenderPassBeginInfo::framebuffer(framebuffer)
                        },
                        vk::SubpassBeginInfo {
                            contents: vk::SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(self.msaa_pipeline.clone())?
                    .set_viewport(0, smallvec::smallvec![viewport.clone()])?
                    .bind_vertex_buffers(0, (output.vertices.clone(), instances))?
                    .bind_index_buffer(output.indices.clone())?
                    .push_constants(
                        self.msaa_pipeline.layout().clone(),
                        0,
                        shaders::msaa::vert::Matrix { mvp: xform.into() },
                    )?
                    .draw_indexed_indirect(indirects)?
                    .next_subpass(
                        vk::SubpassEndInfo::default(),
                        vk::SubpassBeginInfo {
                            contents: vk::SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(self.colorize_pipeline.clone())?
                    .set_viewport(0, smallvec::smallvec![viewport.clone()])?
                    .bind_descriptor_sets(
                        vk::PipelineBindPoint::Graphics,
                        self.colorize_pipeline.layout().clone(),
                        0,
                        self.resolve_input_set.clone(),
                    )?
                    .push_constants(
                        self.colorize_pipeline.layout().clone(),
                        0,
                        shaders::colorize::frag::Color {
                            modulate: monochrome_color,
                        },
                    )?
                    .draw(3, 1, 0, 0)?
                    .end_render_pass(vk::SubpassEndInfo::default())?;

                commands.build().map_err(Into::into)
            } else {
                // Nothing to draw. Just clear the output image for consistent behavior
                let clear = vk::ClearColorImageInfo {
                    regions: smallvec::smallvec![render_into.subresource_range().clone()],
                    ..vk::ClearColorImageInfo::image(render_into.image().clone())
                };

                commands.clear_color_image(clear)?;

                commands.build().map_err(Into::into)
            }
        }
    }
}
pub mod color {
    use crate::vulkano_prelude::*;
    use std::sync::Arc;

    mod shaders {
        pub mod msaa {
            //! Write glyphs into monochrome MSAA buff
            pub mod vert {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 460
                        layout(push_constant) uniform Matrix {
                            mat4 mvp;
                        };
                        // Per-vertex
                        layout(location = 0) in vec2 position;        // arbitrary font units (not em)
                        // Per-instance (per-glyph)
                        layout(location = 1) in ivec2 glyph_position; // arbitrary font units (not em)
                        layout(location = 2) in vec4 glyph_color;

                        // Flat since these aren't *vertex* colors and are constant accross whole mesh.
                        // Not sure if this has perf benifit :V
                        layout(location = 0) flat out vec4 instance_color;

                        void main() {
                            gl_Position = mvp * vec4(position + vec2(glyph_position), 0.0, 1.0);
                            instance_color = glyph_color;
                        }
                    "
                }
            }
            pub mod frag {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 460

                        layout(location = 0) flat in vec4 instance_color;
                        layout(location = 0) out vec4 color;
                        void main() {
                            color = instance_color;
                        }
                    "
                }
            }
        }
    }
    pub struct Renderer {
        context: Arc<crate::render_device::RenderContext>,
        renderpass: Arc<vk::RenderPass>,
        /// Draws the tris into MSAA buff
        msaa_pipeline: Arc<vk::GraphicsPipeline>,
        /// MSAA buffer for antialaised rendering, immediately resolved to a regular image.
        /// (thus, can be transient attatchment)
        multisample: Arc<vk::ImageView>,
    }
    impl Renderer {
        /// Get the internal scale factor due to multisampling.
        /// Should be multiplied by the tessellation factor prior to building for best looking results.
        #[must_use]
        pub fn internal_size_class(&self) -> super::SizeClass {
            use vk::SampleCount;
            // Spacial resolution multiplier is sqrt(sample count).
            // To convert to SizeClass, take the log2 of this, ceil.
            let log2_sqrt_samples = match self.multisample.image().samples() {
                SampleCount::Sample1 => 0,
                SampleCount::Sample2 | SampleCount::Sample4 => 1,
                SampleCount::Sample8 | SampleCount::Sample16 => 2,
                SampleCount::Sample32 | SampleCount::Sample64 => 3,
                // We should never be able to choose a sample count we can't even observe!
                _ => unimplemented!("unknown sample count"),
            };

            super::SizeClass::from_exp_lossy(log2_sqrt_samples)
        }
        pub fn make_renderpass(
            device: Arc<vk::Device>,
            format: vk::Format,
            samples: vk::SampleCount,
        ) -> anyhow::Result<Arc<vk::RenderPass>> {
            // We can use the macro here - not using transient image as resolve, which causes
            // the issue with this macro!
            vulkano::single_pass_renderpass!(
                device,
                attachments: {
                    msaa: {
                        format: format,
                        samples: samples,
                        load_op: Clear,
                        store_op: DontCare,
                    },
                    resolve: {
                        format: crate::DOCUMENT_FORMAT,
                        samples: 1,
                        load_op: DontCare,
                        store_op: Store,
                    }
                },
                pass: {
                    color: [msaa],
                    color_resolve: [resolve],
                    depth_stencil: {},
                },
            )
            .map_err(Into::into)
        }
        fn make_image_for(
            context: &crate::render_device::RenderContext,
            format: vk::Format,
        ) -> anyhow::Result<Arc<vk::ImageView>> {
            // Prevent absurd sample counts for deminishing returns.
            // Todo: polyfill this when not available on this hardware for consistent visuals across devices.
            let valid_samples = vk::SampleCounts::SAMPLE_2
                | vk::SampleCounts::SAMPLE_4
                | vk::SampleCounts::SAMPLE_8;
            let samples = context
                .physical_device()
                .image_format_properties(vulkano::image::ImageFormatInfo {
                    format,
                    tiling: vulkano::image::ImageTiling::Optimal,
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT | vk::ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                })?
                .ok_or_else(|| anyhow::anyhow!("unsupported image configuration"))?
                .sample_counts
                .intersection(valid_samples)
                .max_count();
            if samples == vk::SampleCount::Sample1 {
                // The failure path here involves a whole different pipeline :V
                anyhow::bail!("msaa unsupported")
            }
            let multisample = vk::Image::new(
                context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    format,
                    samples,
                    // Don't need long-lived data. Render, and resolve source.
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT | vk::ImageUsage::COLOR_ATTACHMENT,
                    // Todo: query whether this is supported
                    extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                    sharing: vk::Sharing::Exclusive,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?;

            Ok(vk::ImageView::new_default(multisample)?)
        }
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
            let multisample = Self::make_image_for(context.as_ref(), crate::DOCUMENT_FORMAT)?;
            let samples = multisample.image().samples();
            let renderpass =
                Self::make_renderpass(context.device().clone(), multisample.format(), samples)?;

            let msaa_pipeline = Self::make_msaa_pipe(
                context.device().clone(),
                renderpass.clone().first_subpass(),
                samples,
            )?;

            Ok(Self {
                context,
                renderpass,
                msaa_pipeline,
                multisample,
            })
        }
        /// Make a framebuffer compatible with `self.monochrome_renderpass`
        fn make_framebuffer(
            &self,
            render_into: Arc<vk::ImageView>,
        ) -> anyhow::Result<Arc<vk::Framebuffer>> {
            vk::Framebuffer::new(
                self.renderpass.clone(),
                vk::FramebufferCreateInfo {
                    attachments: vec![self.multisample.clone(), render_into],
                    ..Default::default()
                },
            )
            .map_err(Into::into)
        }
        fn make_msaa_pipe(
            device: Arc<vk::Device>,
            subpass: vk::Subpass,
            samples: vk::SampleCount,
        ) -> anyhow::Result<Arc<vk::GraphicsPipeline>> {
            let vert = shaders::msaa::vert::load(device.clone())?;
            let vert = vert.entry_point("main").unwrap();
            let frag = shaders::msaa::frag::load(device.clone())?;
            let frag = frag.entry_point("main").unwrap();

            let vertex_description = [
                super::interface::Vertex::per_vertex(),
                super::interface::Instance::per_instance(),
            ]
            .definition(&vert.info().input_interface)?;

            let stages = smallvec::smallvec![
                vk::PipelineShaderStageCreateInfo::new(vert),
                vk::PipelineShaderStageCreateInfo::new(frag),
            ];

            let matrix_range = vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<shaders::msaa::vert::Matrix>()
                    .try_into()
                    .unwrap(),
                stages: vk::ShaderStages::VERTEX,
            };

            let layout = vk::PipelineLayout::new(
                device.clone(),
                vk::PipelineLayoutCreateInfo {
                    push_constant_ranges: vec![matrix_range],
                    ..Default::default()
                },
            )?;

            vk::GraphicsPipeline::new(
                device,
                None,
                vk::GraphicsPipelineCreateInfo {
                    stages,
                    vertex_input_state: Some(vertex_description),
                    input_assembly_state: Some(vk::InputAssemblyState::default()),
                    color_blend_state: Some(vk::ColorBlendState::with_attachment_states(
                        1,
                        vk::ColorBlendAttachmentState::default(),
                    )),
                    rasterization_state: Some(vk::RasterizationState::default()),
                    subpass: Some(subpass.into()),
                    multisample_state: Some(vk::MultisampleState {
                        rasterization_samples: samples,
                        ..Default::default()
                    }),
                    // Viewport dynamic, scissor irrelevant.
                    viewport_state: Some(vk::ViewportState::default()),
                    dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                    ..vk::GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .map_err(Into::into)
        }
        /// Create commands to render `DrawOutput` into the given image.
        ///
        /// `xform` should be a combined view+proj matrix, tranforming face's arbitrary units (NOT em) to NDC.
        /// See [`rustybuzz::ttf_parser::Face::units_per_em`].
        ///
        /// `self` is in exclusive use through the duration of the returned buffer's execution.
        pub fn draw(
            &self,
            xform: ultraviolet::Mat4,
            render_into: Arc<vk::ImageView>,
            output: &super::DrawOutput,
        ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
            // We accept either color mode. Would be more efficient if the monochrome pipe was used, tho.
            if matches!(output.color_mode, super::OutputColor::Solid(_)) {
                log::debug!("using expensive color pipe for monochrome text!");
            }
            // Even if none to draw, we can still clear and return.
            let instances_indirects = super::predraw_upload(&self.context, output)?;

            let mut commands = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;

            if let Some((instances, indirects)) = instances_indirects {
                let framebuffer = self.make_framebuffer(render_into)?;
                let viewport = vk::Viewport {
                    depth_range: 0.0..=1.0,
                    offset: [0.0; 2],
                    extent: [
                        framebuffer.extent()[0] as f32,
                        framebuffer.extent()[1] as f32,
                    ],
                };

                commands
                    .begin_render_pass(
                        vk::RenderPassBeginInfo {
                            // Start is cleared, other two are dontcare (immediately overwritten)
                            clear_values: vec![Some([0.0; 4].into()), None, None],
                            ..vk::RenderPassBeginInfo::framebuffer(framebuffer)
                        },
                        vk::SubpassBeginInfo {
                            contents: vk::SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(self.msaa_pipeline.clone())?
                    .set_viewport(0, smallvec::smallvec![viewport.clone()])?
                    .bind_vertex_buffers(0, (output.vertices.clone(), instances))?
                    .bind_index_buffer(output.indices.clone())?
                    .push_constants(
                        self.msaa_pipeline.layout().clone(),
                        0,
                        shaders::msaa::vert::Matrix { mvp: xform.into() },
                    )?
                    .draw_indexed_indirect(indirects)?
                    .next_subpass(
                        vk::SubpassEndInfo::default(),
                        vk::SubpassBeginInfo {
                            contents: vk::SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .end_render_pass(vk::SubpassEndInfo::default())?;

                commands.build().map_err(Into::into)
            } else {
                // Nothing to draw. Just clear the output image for consistent behavior
                let clear = vk::ClearColorImageInfo {
                    regions: smallvec::smallvec![render_into.subresource_range().clone()],
                    ..vk::ClearColorImageInfo::image(render_into.image().clone())
                };

                commands.clear_color_image(clear)?;

                commands.build().map_err(Into::into)
            }
        }
    }
}

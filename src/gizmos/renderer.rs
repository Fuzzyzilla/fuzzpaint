use crate::vulkano_prelude::*;
use std::sync::Arc;

#[derive(vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct GizmoVertex {
    /// Position, in the gizmo's local coordinates.
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    /// Straight, linear RGBA color
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
}

mod shaders {
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    pub struct PushConstants {
        /// Should be an affine2 transform, but glsl std430 align and packing of such is unclear.
        pub transform: [[f32; 4]; 4],
        /// The color the whole object is multiplied by.
        pub color: [f32; 4],
    }
    pub mod vertex {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r#"#version 460
            
            layout(std430, push_constant) uniform Push {
                mat4 transform;
                vec4 gizmo_color;
            };

            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec4 color;
            layout(location = 2) in vec2 uv;

            layout(location = 0) out vec4 outColor;
            layout(location = 1) out vec2 outUV;

            void main() {
                outColor = color * gizmo_color;
                outUV = uv;

                gl_Position = transform * vec4(pos, 0.0, 1.0);
            }
            "#
        }
    }
    pub mod fragment_textured {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460
            
            layout(set = 0, binding = 0) uniform sampler2D tex;

            layout(location = 0) in vec4 inColor;
            layout(location = 1) in vec2 inUV;

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = texture(tex, inUV) * inColor;
            }
            "#
        }
    }
    pub mod fragment_untextured {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460

            layout(location = 0) in vec4 inColor;
            layout(location = 1) in vec2 _;

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = inColor;
            }
            "#
        }
    }
    pub mod fragment_ant_trail {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460

            layout(location = 0) in vec4 _0;
            layout(location = 1) in vec2 _1;

            layout(location = 0) out vec4 outColor;

            void main() {
                uint pos = uint(gl_FragCoord.x + gl_FragCoord.y);

                float bright = ((pos & 0xF) < 8) ? 1.0: 0.0;
                outColor = vec4(vec3(bright), 1.0) * outColor;
            }
            "#
        }
    }
    pub mod thick_polyline {
        pub mod vert {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r#"#version 460
                
                layout(std430, push_constant) uniform Push {
                    mat4 transform;
                    vec4 gizmo_color;
                };
    
                layout(location = 0) in vec2 pos;
                layout(location = 1) in vec4 color;
                // Polyline has a single-dimension UV, or just U (that would be confusing tho lol)
                // Expands into UV in the widening geometry shader
                layout(location = 2) in float tex_coord;
                // Width, in transform units.
                layout(location = 3) in float width;
    
                layout(location = 0) out vec4 out_color;
                layout(location = 1) out float out_texcoord;
                layout(location = 2) out float out_width;
    
                void main() {
                    out_color = color * gizmo_color;
                    out_texcoord = tex_coord;
                    out_width = width;

                    gl_Position = transform * vec4(pos, 0.0, 1.0);
                }
                "#
            }
        }
        // Takes lines adjacency and turns each segment into a
        // wide rectangle.
        pub mod geom {
            vulkano_shaders::shader! {
                ty: "geometry",
                src: r#"#version 460
                layout(std430, push_constant) uniform Push {
                    mat4 transform;
                };

                layout(lines_adjacency) in;
                layout(triangle_strip, max_vertices = 4) out;
    
                layout(location = 0) in vec4 in_color[4];
                layout(location = 1) in float in_texcoord[4];
                layout(location = 2) in float in_width[4];
    
                // Output compatible with the other fragment shaders
                layout(location = 0) out vec4 out_color;
                layout(location = 1) out vec2 out_uv;

                #define A 0
                #define B 1
                #define C 2
                #define D 3
                #define A_in gl_in[A]
                #define B_in gl_in[B]
                #define C_in gl_in[C]
                #define D_in gl_in[D]

                // positive if B is "to the left of" A.
                // alternatively, sin of the counterclockwise angle between A and B, times A and B's lengths.
                float cross2(in vec2 a, in vec2 b) {
                    return a.x * b.y - a.y * b.x;
                }

                void main() {
                    //         2\--------4
                    //         |  \      |
                    // A - - - B - -\- - C - - - D
                    //         |      \  |
                    //         1--------\3

                    // --------- Calculate.....
                    // Todo: what if smol delta?
                    vec2 ba = normalize(A_in.gl_Position.xy - B_in.gl_Position.xy);
                    vec2 bc = normalize(C_in.gl_Position.xy - B_in.gl_Position.xy);
                    vec2 cd = normalize(D_in.gl_Position.xy - C_in.gl_Position.xy);
                    vec2 cb = normalize(B_in.gl_Position.xy - C_in.gl_Position.xy);

                    // If inner angle is acute, mirror it!
                    // Makes sharp corners have a squared off appearance instead of a needle lol
                    ba = dot(ba, bc) < 0 ? -ba : ba;
                    cd = dot(cd, cb) < 0 ? -cd : cd;

                    // Points from B to 1 or 2
                    // Todo: what if smol normal?
                    vec2 b_normal = normalize((bc - ba) / 2.0);
                    // Make the cross positive, now points from B to 2
                    b_normal = cross2(ba, b_normal) < 0.0 ? -b_normal : b_normal;

                    // Points from C to 3 or 4
                    vec2 c_normal = normalize((cb - cd) / 2.0);
                    // Make the cross positive, now points from C to 4
                    c_normal = cross2(cd, c_normal) < 0.0 ? -c_normal : c_normal;

                    // Project normals (todo: is this right?)
                    b_normal = (transform * vec4(b_normal, 0.0, 0.0)).xy;
                    c_normal = (transform * vec4(c_normal, 0.0, 0.0)).xy;

                    // --------- Do vertices!
                    float b_half_width = abs(in_width[B] / 2.0);
                    float c_half_width = abs(in_width[C] / 2.0);

                    // 1
                    gl_Position = B_in.gl_Position + vec4(b_normal * b_half_width, 0.0, 0.0);
                    out_color = in_color[B];
                    out_uv = vec2(in_texcoord[B], 0.0);
                    EmitVertex();

                    // 2
                    gl_Position = B_in.gl_Position - vec4(b_normal * b_half_width, 0.0, 0.0);
                    out_color = in_color[B];
                    out_uv = vec2(in_texcoord[B], 1.0);
                    EmitVertex();

                    // 3
                    gl_Position = C_in.gl_Position - vec4(c_normal * c_half_width, 0.0, 0.0);
                    out_color = in_color[C];
                    out_uv = vec2(in_texcoord[C], 1.0);
                    EmitVertex();

                    // 4
                    gl_Position = C_in.gl_Position + vec4(c_normal * c_half_width, 0.0, 0.0);
                    out_color = in_color[C];
                    out_uv = vec2(in_texcoord[C], 0.0);
                    EmitVertex();
                }
                "#
            }
        }
    }
}
pub struct GizmoRenderer {
    context: Arc<crate::render_device::RenderContext>,
    textured_pipeline: Arc<vk::GraphicsPipeline>,
    untextured_pipeline: Arc<vk::GraphicsPipeline>,

    // Premade, static vertex buffers for common shapes.
    triangulated_shapes: vk::Subbuffer<[GizmoVertex]>,
    triangulated_square: vk::Subbuffer<[GizmoVertex]>,
    _triangulated_circle: vk::Subbuffer<[GizmoVertex]>,
}
impl GizmoRenderer {
    const CIRCLE_RES: usize = 32;
    /// Create the layouts for both the textured and untextured pipelines, sharing the same push constant layout.
    fn layout(
        context: &crate::render_device::RenderContext,
    ) -> anyhow::Result<(Arc<vk::PipelineLayout>, Arc<vk::PipelineLayout>)> {
        let push_constant_ranges = {
            let matrix_color_range = vk::PushConstantRange {
                offset: 0,
                stages: vk::ShaderStages::VERTEX,
                size: 4 * 4 * 4 + 4 * 4, //4x4 matrix of f32, + vec4 of f32
            };
            vec![matrix_color_range]
        };
        let mut texture_descriptor_set = std::collections::BTreeMap::new();
        texture_descriptor_set.insert(
            0,
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                stages: vk::ShaderStages::FRAGMENT,
                ..vk::DescriptorSetLayoutBinding::descriptor_type(
                    vk::DescriptorType::CombinedImageSampler,
                )
            },
        );

        let texture_descriptor_set = vk::DescriptorSetLayout::new(
            context.device().clone(),
            vk::DescriptorSetLayoutCreateInfo {
                bindings: texture_descriptor_set,
                ..Default::default()
            },
        )?;
        let textured = vk::PipelineLayout::new(
            context.device().clone(),
            vk::PipelineLayoutCreateInfo {
                set_layouts: vec![texture_descriptor_set],
                push_constant_ranges: push_constant_ranges.clone(),
                ..Default::default()
            },
        )?;
        let untextured = vk::PipelineLayout::new(
            context.device().clone(),
            vk::PipelineLayoutCreateInfo {
                set_layouts: Vec::new(),
                push_constant_ranges,
                ..Default::default()
            },
        )?;
        Ok((textured, untextured))
    }
    /// Make static shape buffers. (unit square origin at 0.0, unit circle origin at 0.0)
    fn make_shapes(
        context: &crate::render_device::RenderContext,
    ) -> anyhow::Result<(
        vk::Subbuffer<[GizmoVertex]>,
        vk::Subbuffer<[GizmoVertex]>,
        vk::Subbuffer<[GizmoVertex]>,
    )> {
        let mut vertices = Vec::with_capacity(6 + Self::CIRCLE_RES * 3);
        // Construct square
        {
            let top_right = GizmoVertex {
                pos: [1.0, 0.0],
                uv: [1.0, 1.0],
                color: [255; 4],
            };
            let bottom_left = GizmoVertex {
                pos: [0.0, 1.0],
                uv: [0.0, 0.0],
                color: [255; 4],
            };
            vertices.extend_from_slice(&[
                GizmoVertex {
                    pos: [0.0, 0.0],
                    uv: [0.0, 1.0],
                    color: [255; 4],
                },
                top_right,
                bottom_left,
                bottom_left,
                top_right,
                GizmoVertex {
                    pos: [1.0, 1.0],
                    uv: [1.0, 0.0],
                    color: [255; 4],
                },
            ]);
        };
        // construct circle
        let circle = (0..Self::CIRCLE_RES).flat_map(|idx| {
            let proportion = std::f32::consts::TAU * (idx as f32) / (Self::CIRCLE_RES as f32);
            let proportion_next =
                std::f32::consts::TAU * ((idx + 1) as f32) / (Self::CIRCLE_RES as f32);

            let (first_y, first_x) = proportion.sin_cos();
            let (next_y, next_x) = proportion_next.sin_cos();

            [
                // Center
                GizmoVertex {
                    pos: [0.0, 0.0],
                    uv: [0.5, 0.5],
                    color: [255; 4],
                },
                GizmoVertex {
                    pos: [first_x, first_y],
                    uv: [first_x / 2.0 + 0.5, 1.0 - (first_y / 2.0 + 0.5)],
                    color: [255; 4],
                },
                GizmoVertex {
                    pos: [next_x, next_y],
                    uv: [next_x / 2.0 + 0.5, 1.0 - (next_y / 2.0 + 0.5)],
                    color: [255; 4],
                },
            ]
        });
        vertices.extend(circle);
        let triangulated_shapes = vk::Buffer::from_iter(
            context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                sharing: vk::Sharing::Exclusive,
                usage: vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                // Prefer device memory, slow access from CPU is fine but it needs to write once.
                // Should this be staged?
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.into_iter(),
        )?;

        let square = triangulated_shapes.clone().slice(0..6);
        let circle = triangulated_shapes.clone().slice(6..);

        Ok((triangulated_shapes, square, circle))
    }
    pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        let vertex = shaders::vertex::load(context.device().clone())?;
        let textured_fragment = shaders::fragment_textured::load(context.device().clone())?;
        let untextured_fragment = shaders::fragment_untextured::load(context.device().clone())?;
        let vertex = vertex.entry_point("main").unwrap();
        let textured_fragment = textured_fragment.entry_point("main").unwrap();
        let untextured_fragment = untextured_fragment.entry_point("main").unwrap();
        let vertex_stage = vk::PipelineShaderStageCreateInfo::new(vertex.clone());
        let textured_fragment_stage = vk::PipelineShaderStageCreateInfo::new(textured_fragment);
        let untextured_fragment_stage = vk::PipelineShaderStageCreateInfo::new(untextured_fragment);

        let (textured_pipeline_layout, untextured_pipeline_layout) =
            Self::layout(context.as_ref())?;

        let alpha_blend = {
            let alpha_blend = vk::AttachmentBlend::alpha();
            let blend_states = vk::ColorBlendAttachmentState {
                blend: Some(alpha_blend),
                ..Default::default()
            };
            vk::ColorBlendState::with_attachment_states(1, blend_states)
        };
        // ad hoc rendering for now, lazy lazy
        let render_pass =
            vk::PipelineSubpassType::BeginRendering(vk::PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(vk::Format::B8G8R8A8_SRGB)],
                ..Default::default()
            });
        let textured_pipeline_info = vk::GraphicsPipelineCreateInfo {
            color_blend_state: Some(alpha_blend),
            input_assembly_state: Some(vk::InputAssemblyState {
                topology: vk::PrimitiveTopology::TriangleList,
                primitive_restart_enable: false,
                ..Default::default()
            }),
            multisample_state: Some(vk::MultisampleState::default()),
            rasterization_state: Some(vk::RasterizationState {
                cull_mode: vk::CullMode::None,
                ..Default::default()
            }),
            vertex_input_state: Some(
                GizmoVertex::per_vertex().definition(&vertex.info().input_interface)?,
            ),
            // One viewport and scissor, scissor irrelevant and viewport dynamic
            viewport_state: Some(vk::ViewportState::default()),
            dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
            subpass: Some(render_pass),
            stages: smallvec::smallvec![vertex_stage.clone(), textured_fragment_stage],
            ..vk::GraphicsPipelineCreateInfo::layout(textured_pipeline_layout)
        };
        let textured_pipeline = vk::GraphicsPipeline::new(
            context.device().clone(),
            None,
            textured_pipeline_info.clone(),
        )?;
        let untextured_pipeline = vk::GraphicsPipeline::new(
            context.device().clone(),
            None,
            vk::GraphicsPipelineCreateInfo {
                layout: untextured_pipeline_layout,
                stages: smallvec::smallvec![vertex_stage, untextured_fragment_stage],
                ..textured_pipeline_info
            },
        )?;

        let (shapes, square, circle) = Self::make_shapes(context.as_ref())?;

        Ok(Self {
            context,
            textured_pipeline,
            untextured_pipeline,

            triangulated_shapes: shapes,
            _triangulated_circle: circle,
            triangulated_square: square,
        })
    }
    // Temporary api. passing around swapchain images and proj matrices like this feels dirty :P
    pub fn render_visit(
        &self,
        into_image: Arc<vk::Image>,
        image_size: [f32; 2],
        document_transform: crate::view_transform::ViewTransform,
        proj: cgmath::Matrix4<f32>,
    ) -> anyhow::Result<RenderVisitor<'_>> {
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        let attachment = vk::RenderingAttachmentInfo {
            clear_value: None,
            load_op: vk::AttachmentLoadOp::Load,
            store_op: vk::AttachmentStoreOp::Store,
            ..vk::RenderingAttachmentInfo::image_view(vk::ImageView::new_default(into_image)?)
        };
        command_buffer
            .begin_rendering(vk::RenderingInfo {
                color_attachments: vec![Some(attachment)],
                contents: vk::SubpassContents::Inline,
                depth_attachment: None,
                ..Default::default()
            })?
            .set_viewport(
                0,
                smallvec::smallvec![vk::Viewport {
                    depth_range: 0.0..=1.0,
                    extent: image_size,
                    offset: [0.0; 2],
                }],
            )?
            .bind_vertex_buffers(0, self.triangulated_shapes.clone())?;

        Ok(RenderVisitor {
            renderer: self,
            xform_stack: vec![document_transform],
            command_buffer,
            current_pipeline: None,
            proj,
        })
    }
}

pub struct RenderVisitor<'a> {
    renderer: &'a GizmoRenderer,
    xform_stack: Vec<crate::view_transform::ViewTransform>,
    command_buffer: vk::AutoCommandBufferBuilder<vk::PrimaryAutoCommandBuffer>,
    current_pipeline: Option<&'a vk::GraphicsPipeline>,
    proj: cgmath::Matrix4<f32>,
}
impl RenderVisitor<'_> {
    pub fn build(mut self) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        self.command_buffer.end_rendering()?;
        let build = self.command_buffer.build()?;

        Ok(build)
    }
}
// Thats... a unique way to propogate errors... :P
impl<'a> super::GizmoVisitor<anyhow::Error> for RenderVisitor<'a> {
    fn visit_collection(
        &mut self,
        gizmo: &super::Collection,
    ) -> std::ops::ControlFlow<anyhow::Error> {
        let Some(parent_xform) = self.xform_stack.last() else {
            // Shouldn't happen! visit_ and end_collection should be symmetric.
            // Some to short circuit the visitation
            return std::ops::ControlFlow::Break(anyhow::anyhow!("xform stack empty!"));
        };
        // unwrap ok - checked above.
        let base_xform = self.xform_stack.first().unwrap();
        let new_xform = gizmo.transform.apply(base_xform, parent_xform);

        self.xform_stack.push(new_xform);

        std::ops::ControlFlow::Continue(())
    }
    fn visit_gizmo(&mut self, gizmo: &super::Gizmo) -> std::ops::ControlFlow<anyhow::Error> {
        // try_block macro doesn't impl FnMut it's kinda weird :V
        let mut try_block = || -> anyhow::Result<()> {
            // Get shape if any, early return if not.
            let super::GizmoVisual::Shape {
                shape,
                texture,
                color,
            } = &gizmo.visual
            else {
                return Ok(());
            };

            let Some(parent_xform) = self.xform_stack.last() else {
                // Shouldn't happen! visit_ and end_collection should be symmetric.
                // Some to short circuit the visitation
                anyhow::bail!("xform stack empty!")
            };
            // unwrap ok - checked above.
            let base_xform = self.xform_stack.first().unwrap();
            let local_xform = gizmo.transform.apply(base_xform, parent_xform);

            // draw gizmo using local_xform.
            if let Some(texture) = texture {
                //swap pipeline, if needed:
                if self.current_pipeline != Some(&*self.renderer.textured_pipeline) {
                    self.current_pipeline = Some(&*self.renderer.textured_pipeline);
                    self.command_buffer
                        .bind_pipeline_graphics(self.renderer.textured_pipeline.clone())?;
                }
                self.command_buffer.bind_descriptor_sets(
                    vk::PipelineBindPoint::Graphics,
                    self.renderer.textured_pipeline.layout().clone(),
                    0,
                    texture.clone(),
                )?;
            } else {
                //swap pipeline, if needed:
                if self.current_pipeline != Some(&*self.renderer.untextured_pipeline) {
                    self.current_pipeline = Some(&*self.renderer.untextured_pipeline);
                    self.command_buffer
                        .bind_pipeline_graphics(self.renderer.untextured_pipeline.clone())?;
                }
            }
            let shape_xform: cgmath::Matrix4<f32> = {
                // might seem silly. maybe.. maybe...
                let (offs, scale, rotation) = match shape {
                    super::RenderShape::Rectangle {
                        position,
                        size,
                        rotation,
                    } => (position, size, rotation),
                    super::RenderShape::Ellipse {
                        origin,
                        radii,
                        rotation,
                    } => (origin, radii, rotation),
                };
                cgmath::Matrix4::from_translation(cgmath::Vector3 {
                    x: offs.x,
                    y: offs.y,
                    z: 0.0,
                }) * cgmath::Matrix4::from_nonuniform_scale(scale.x, scale.y, 1.0)
                    * cgmath::Matrix4::from_angle_z(cgmath::Rad(*rotation))
            };
            let matrix: cgmath::Matrix4<f32> = local_xform.into();
            // Stretch/position shape, then move from local to viewspace, then project to NDC
            let matrix = self.proj * matrix * shape_xform;
            let push_constants = shaders::PushConstants {
                color: [
                    f32::from(color[0]) / 255.0,
                    f32::from(color[1]) / 255.0,
                    f32::from(color[2]) / 255.0,
                    f32::from(color[3]) / 255.0,
                ],
                transform: matrix.into(),
            };

            self.command_buffer.push_constants(
                self.current_pipeline.unwrap().layout().clone(),
                0,
                push_constants,
            )?;
            match shape {
                super::RenderShape::Rectangle { .. } => self.command_buffer.draw(
                    self.renderer.triangulated_square.len() as u32,
                    1,
                    self.renderer.triangulated_square.offset() as u32,
                    0,
                )?,
                super::RenderShape::Ellipse { .. } => {
                    self.command_buffer
                        .draw(GizmoRenderer::CIRCLE_RES as u32 * 3, 1, 6, 0)?
                }
            };
            Ok(())
        };
        match try_block() {
            Ok(()) => std::ops::ControlFlow::Continue(()),
            Err(anyhow) => std::ops::ControlFlow::Break(anyhow),
        }
    }
    fn end_collection(&mut self, _: &super::Collection) -> std::ops::ControlFlow<anyhow::Error> {
        if self.xform_stack.pop().is_some() {
            std::ops::ControlFlow::Continue(())
        } else {
            // would be a gizmo implementation error.
            std::ops::ControlFlow::Break(anyhow::anyhow!("Unbalanced gizmo tree!"))
        }
    }
}

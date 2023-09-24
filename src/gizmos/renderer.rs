use std::sync::Arc;

use vulkano::pipeline::Pipeline;

use crate::vk;
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
            };

            layout(location = 0) in vec2 inPos;
            layout(location = 1) in vec4 inColor;
            layout(location = 2) in vec2 inUV;

            layout(location = 0) out vec4 outColor;
            layout(location = 1) out vec2 outUV;

            void main() {
                outColor = inColor;
                outUV = inUV;

                gl_Position = transform * vec4(inPos, 0.0, 1.0);
            }"#
        }
    }
    pub mod fragment_textured {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460
            
            layout(set = 0, binding = 0) uniform sampler2D tex;

            layout(std430, push_constant) uniform Push {
                vec4 gizmo_color;
            };

            layout(location = 0) in vec4 inColor;
            layout(location = 1) in vec2 inUV;

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = texture(tex, inUV) * inColor * gizmo_color;
            }"#
        }
    }
    pub mod fragment_untextured {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460

            layout(std430, push_constant) uniform Push {
                vec4 gizmo_color;
            };

            layout(location = 0) in vec4 inColor;
            layout(location = 1) in vec2 _;

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = inColor * gizmo_color;
            }"#
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
    triangulated_circle: vk::Subbuffer<[GizmoVertex]>,
}
impl GizmoRenderer {
    const CIRCLE_RES: usize = 16;
    fn layout(
        context: &crate::render_device::RenderContext,
    ) -> anyhow::Result<Arc<vk::PipelineLayout>> {
        todo!()
    }
    /// Make static shape buffers. (unit square origin at 0.0, unit circle origin at 0.0)
    fn make_shapes(
        context: &crate::render_device::RenderContext,
    ) -> anyhow::Result<(vk::Subbuffer<[GizmoVertex]>, vk::Subbuffer<[GizmoVertex]>)> {
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
        let circle = (0..Self::CIRCLE_RES).into_iter().flat_map(|idx| {
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
            context.allocators().memory(),
            vk::BufferCreateInfo {
                sharing: vulkano::sync::Sharing::Exclusive,
                usage: vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                usage: vulkano::memory::allocator::MemoryUsage::Upload,
                ..Default::default()
            },
            vertices.into_iter(),
        )?;

        let square = triangulated_shapes.clone().slice(0..6);
        let circle = triangulated_shapes.clone().slice(6..);

        Ok((square, circle))
    }
    pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        todo!()
    }
    pub fn render_visit<'s>(
        &'s self,
        document_transform: crate::view_transform::ViewTransform,
        proj: &cgmath::Matrix4<f32>,
    ) -> anyhow::Result<RenderVisitor<'s>> {
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().graphics().idx(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.bind_vertex_buffers(0, self.triangulated_shapes.clone());

        Ok(RenderVisitor {
            renderer: self,
            xform_stack: vec![document_transform],
            command_buffer,
            current_pipeline: None,
            proj: *proj,
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
    pub fn build(mut self) -> anyhow::Result<vk::PrimaryAutoCommandBuffer> {
        self.command_buffer.end_rendering()?;
        let build = self.command_buffer.build()?;

        Ok(build)
    }
}
// Thats... a unique way to propogate errors... :P
impl<'a> super::GizmoVisitor<anyhow::Error> for RenderVisitor<'a> {
    fn visit_collection(&mut self, gizmo: &super::Collection) -> Option<anyhow::Error> {
        let Some(parent_xform) = self.xform_stack.last() else {
            // Shouldn't happen! visit_ and end_collection should be symmetric.
            // Some to short circuit the visitation
            return Some(anyhow::anyhow!("xform stack empty!"));
        };
        // unwrap ok - checked above.
        let base_xform = self.xform_stack.first().unwrap();
        let new_xform = gizmo.transform.apply(base_xform, parent_xform);

        self.xform_stack.push(new_xform);

        None
    }
    fn visit_gizmo(&mut self, gizmo: &super::Gizmo) -> Option<anyhow::Error> {
        // Get shape if any, early return if not.
        let super::GizmoVisual::Shape {
            shape,
            texture,
            color,
        } = &gizmo.visual
        else {
            return None;
        };

        let Some(parent_xform) = self.xform_stack.last() else {
            // Shouldn't happen! visit_ and end_collection should be symmetric.
            // Some to short circuit the visitation
            return Some(anyhow::anyhow!("xform stack empty!"));
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
                    .bind_pipeline_graphics(self.renderer.textured_pipeline.clone());
            }
            self.command_buffer.bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                self.renderer.textured_pipeline.layout().clone(),
                0,
                texture.clone(),
            );
        } else {
            //swap pipeline, if needed:
            if self.current_pipeline != Some(&*self.renderer.untextured_pipeline) {
                self.current_pipeline = Some(&*self.renderer.untextured_pipeline);
                self.command_buffer
                    .bind_pipeline_graphics(self.renderer.untextured_pipeline.clone());
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
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
                color[3] as f32 / 255.0,
            ],
            transform: matrix.into(),
        };

        self.command_buffer.push_constants(
            self.current_pipeline.unwrap().layout().clone(),
            0,
            push_constants,
        );
        match shape {
            super::RenderShape::Rectangle { .. } => {
                if let Err(e) = self.command_buffer.draw(
                    self.renderer.triangulated_square.len() as u32,
                    1,
                    self.renderer.triangulated_square.offset() as u32,
                    0,
                ) {
                    return Some(e.into());
                }
            }
            super::RenderShape::Ellipse { .. } => {
                if let Err(e) = self.command_buffer.draw(
                    self.renderer.triangulated_circle.len() as u32,
                    1,
                    self.renderer.triangulated_circle.offset() as u32,
                    0,
                ) {
                    return Some(e.into());
                }
            }
        }
        None
    }
    fn end_collection(&mut self, _: &super::Collection) -> Option<anyhow::Error> {
        let Some(_) = self.xform_stack.pop() else {
            return Some(anyhow::anyhow!("No xform  to pop!"));
        };
        None
    }
}

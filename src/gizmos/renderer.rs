
use std::sync::Arc;

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
        /// Column major 2x3 matrix, for 2D affine transforms
        // layout kinda confusing: https://www.oreilly.com/library/view/opengl-programming-guide/9780132748445/app09lev1sec3.html
        // Slightly implies the align should be 12, clearly not right. Only other interpretation
        // I could come up with is align of four.
        transform: [[f32; 3]; 2],
        /// The color the whole object is multiplied by.
        color: [f32; 4],
    }
    pub mod vertex {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: r#"#version 460
            
            layout(std430, push_constant) uniform Push {
                mat3x2 transform;
                vec4 gizmo_color;
            };

            layout(location = 0) in vec2 inPos;
            layout(location = 1) in vec4 inColor;
            layout(location = 2) in vec2 inUV;

            layout(location = 0) out vec4 outColor;
            layout(location = 1) out vec2 outUV;

            void main() {
                outColor = inColor;
                outUV = inUV;

                gl_Position = vec4(transform * vec3(inPos, 1.0), 0.0, 1.0);
            }"#
        }
    }
    pub mod fragment {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: r#"#version 460
            
            layout(set = 0, binding = 0) uniform sampler2D tex;

            layout(std430, push_constant) uniform Push {
                mat3x2 transform;
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
}
pub struct GizmoRenderer {

}
impl GizmoRenderer {
    fn layout(context: &crate::render_device::RenderContext) -> anyhow::Result<Arc<vk::PipelineLayout>> {
        todo!()
    }
    pub fn new(context: &crate::render_device::RenderContext) -> anyhow::Result<Self> {
        todo!()
    }
}
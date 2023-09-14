use crate::vk;
use std::sync::Arc;
pub mod interface {
    #[derive(super::vk::Vertex, super::vk::BufferContents, Copy, Clone)]
    #[repr(C)]
    pub struct OutputStrokeVertex {
        #[format(R32G32_SFLOAT)]
        pub pos: [f32; 2],
        #[format(R32G32_SFLOAT)]
        pub uv: [f32; 2],
        #[format(R32G32B32A32_SFLOAT)]
        pub color: [f32; 4],
        #[format(R32_SFLOAT)]
        pub erase: f32,
    }
    #[derive(super::vk::Vertex, super::vk::BufferContents, Copy, Clone)]
    #[repr(C)]
    pub struct InputStrokeInfo {
        // Indices into inputStrokeVertices buffer
        #[format(R32_UINT)]
        pub start_point_idx: u32,
        #[format(R32_UINT)]
        pub num_points: u32,

        // Indices into outputStrokeVertices
        #[format(R32_UINT)]
        pub out_vert_offset: u32,
        #[format(R32_UINT)]
        pub out_vert_limit: u32,

        // Number of pixels between each stamp
        #[format(R32_SFLOAT)]
        pub density: f32,
        // The CPU will dictate how many groups to allocate to this work.
        // Mesh shaders would make this all nicer ;)
        #[format(R32_UINT)]
        pub start_group: u32,
        #[format(R32_UINT)]
        pub num_groups: u32,

        // Color and eraser settings
        #[format(R32G32B32A32_SFLOAT)]
        pub modulate: [f32; 4],
        // Bool32
        #[format(R32_UINT)]
        pub eraser: u32,
    }
    #[derive(super::vk::Vertex, super::vk::BufferContents, Copy, Clone)]
    #[repr(C)]
    pub struct InputStrokeVertex {
        #[format(R32G32_SFLOAT)]
        pub pos: [f32; 2],
        #[format(R32_SFLOAT)]
        pub pressure: f32,
        #[format(R32_SFLOAT)]
        pub dist: f32,
    }
    pub type OutputStrokeInfo = vulkano::command_buffer::DrawIndirectCommand;
}

mod shaders {
    pub mod tessellate {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "./src/shaders/tessellate_stamp.comp",
        }
    }
}

pub struct GpuStampTess {
    context: Arc<crate::render_device::RenderContext>,
    pipeline: Arc<vk::ComputePipeline>,
}
impl GpuStampTess {
    pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        let shader = shaders::tessellate::load(context.device().clone())?;
        let entry = shader.entry_point("main").unwrap();
        let pipeline = vk::ComputePipeline::new(
            context.device().clone(),
            entry,
            &shaders::tessellate::SpecializationConstants::default(),
            None,
            |_| (),
        )?;

        todo!()
    }
}

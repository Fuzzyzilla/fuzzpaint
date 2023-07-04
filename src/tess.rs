pub mod rayon;

#[derive(Debug, Clone)]
pub enum TessellationError {
    VertexBufferTooSmall { needed_size: usize },
    InfosBufferTooSmall { needed_size: usize },
    BufferError(vulkano::buffer::BufferError),
}

pub trait StrokeTessellator {
    fn tessellate(
        &self,
        strokes: &[crate::Stroke],
        infos_into: &mut [TessellatedStrokeInfo],
        vertices_into: crate::vk::Subbuffer<[TessellatedStrokeVertex]>,
        base_vertex: usize,
    ) -> ::std::result::Result<(), TessellationError>;
    /// Exact number of vertices to allocate and draw for this stroke.
    /// No method for estimates for now.
    fn num_vertices_of(&self, stroke: &crate::Stroke) -> usize;
    /// Exact number of vertices to allocate and draw for all strokes.
    fn num_vertices_of_slice(&self, strokes: &[crate::Stroke]) -> usize {
        strokes.iter().map(|s| self.num_vertices_of(s)).sum()
    }
}

/// All the data needed to render tessellated output.
#[derive(Clone, Copy)]
pub struct TessellatedStrokeInfo {
    pub source: crate::FuzzID<crate::Stroke>,
    pub first_vertex: u32,
    pub vertices: u32,
    pub blend_constants: [f32; 4],
}

#[derive(crate::vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct TessellatedStrokeVertex {
    // Must be f32, as f16 only supports up to a 2048px document with pixel precision.
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    // Could maybe be f16, as long as brush textures are <2048px.
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    // Nearly 100% coverage on this vertex input type.
    #[format(R16G16B16A16_SFLOAT)]
    pub color: [vulkano::half::f16; 4],
}

//! # Tessellator
//! The tessellator is the component responsible for converting formatted stroke data into a GPU-renderable mesh.
//! This is done via the `StrokeTessellator` trait. Currently, this is implemented in Rayon, however preparations
//! have been made to do more efficient tessellation on the GPU directly. This pipeline may also be skipped by
//! EXT_mesh_shader.
pub mod rayon;

#[derive(Debug, Clone)]
pub enum TessellationError {
    VertexBufferTooSmall { needed_size: usize },
    BufferError(vulkano::buffer::BufferError),
}

pub trait StrokeTessellator {
    type Stream<'a>: StreamStrokeTessellator<'a>
    where
        Self: 'a; // I do not understand the implications of this bound :V
    /// Tessellate all the strokes into the given subbuffer.
    fn tessellate(
        &self,
        strokes: &[crate::Stroke],
        vertices_into: crate::vk::Subbuffer<[TessellatedStrokeVertex]>,
        base_vertex: usize,
    ) -> ::std::result::Result<(), TessellationError>;
    /// Construct a stream tessellator, which can tessellate many strokes into a
    /// smaller buffer. Repeatedly call `StreamStrokeTessellator::tessellate` to continuously fill new buffers
    // Stream does not borrow self. This makes sense in rayon and vulkano, where the
    // stream maintains all the internals within itself.
    fn stream<'a>(&self, strokes: &'a [crate::Stroke]) -> Self::Stream<'a>;
    /// Exact number of vertices to allocate and draw for this stroke.
    /// No method for estimates for now.
    fn num_vertices_of(&self, stroke: &crate::Stroke) -> usize;
    /// Exact number of vertices to allocate and draw for all strokes.
    fn num_vertices_of_slice(&self, strokes: &[crate::Stroke]) -> usize {
        strokes.iter().map(|s| self.num_vertices_of(s)).sum()
    }
}

pub trait StreamStrokeTessellator<'a> {
    /// Tessellate as many as possible into a buffer.
    /// Returns the number of vertices generated, or None if complete.
    /// Repeated calls to tessellate will continue to make forward progress.
    /// Some overhead is expected for a tessellation pass, so call with a large-ish buffer!
    fn tessellate(&mut self, vertices: &mut [TessellatedStrokeVertex]) -> Option<usize>;
}

/// All the data needed to render tessellated output.
#[derive(Copy, Clone)]
pub struct TessellatedStrokeInfo {
    pub source: crate::WeakID<crate::Stroke>,
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

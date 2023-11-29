//! # Tessellator
//! The tessellator is the component responsible for converting formatted stroke data into a GPU-renderable mesh.
//! This is done via the `StrokeTessellator` trait. Currently, this is implemented in Rayon, however preparations
//! have been made to do more efficient tessellation on the GPU directly. This pipeline may also be skipped by
//! EXT_mesh_shader.
pub mod rayon;

#[derive(Debug)]
pub enum TessellationError {
    VertexBufferTooSmall { needed_size: usize },
    Anyhow(anyhow::Error),
}

pub trait StrokeTessellator {
    /// Tessellate all the strokes into the given subbuffer.
    fn tessellate(
        &self,
        strokes: &[crate::Stroke],
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

pub trait StreamStrokeTessellator<'a> {
    /// Tessellate many vertices into a buffer.
    /// Must be a whole number of primitives! (Eg. TRIANGLE_LIST must have a multiple of 3 verts)
    /// Returns a status - which buffer was exhausted, or it completed successfully.
    /// Repeated calls to tessellate will continue to make forward progress.
    /// Some overhead is expected for a tessellation pass, so call with a large-ish buffer!
    /// infos are always in order, but there may be any number of infos (including zero) per stroke!
    fn tessellate(
        &mut self,
        vertices: &mut [TessellatedStrokeVertex],
        infos: &mut [TessellatedStrokeInfo],
    ) -> StreamStatus;
}

/// All the data needed to render tessellated output.
#[derive(Copy, Clone)]
pub struct TessellatedStrokeInfo {
    pub source: Option<crate::FuzzID<crate::Stroke>>,
    pub first_vertex: u32,
    pub vertices: u32,
}
impl TessellatedStrokeInfo {
    pub fn empty() -> Self {
        Self {
            source: None,
            first_vertex: 0,
            vertices: 0,
        }
    }
}
impl From<TessellatedStrokeInfo> for vulkano::command_buffer::DrawIndirectCommand {
    fn from(value: TessellatedStrokeInfo) -> Self {
        Self {
            first_instance: 0,
            instance_count: 1,
            vertex_count: value.vertices,
            first_vertex: value.first_vertex,
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum StreamStatus {
    VerticesFull,
    InfosFull,
    Complete,
}

#[derive(crate::vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug)]
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

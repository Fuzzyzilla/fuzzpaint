#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct Point {
    pub pos: [f32; 2],
    pub pressure: f32,
    /// Arc length of stroke from beginning to this point
    pub dist: f32,
}

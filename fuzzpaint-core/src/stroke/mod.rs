#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct Point {
    pub pos: [f32; 2],
    pub pressure: f32,
    /// Arc length of stroke from beginning to this point
    pub dist: f32,
}
impl Point {
    #[must_use]
    pub const fn archetype() -> crate::repositories::points::PointArchetype {
        use crate::repositories::points::PointArchetype;
        // | isn't const except for on the bits type! x3
        // This is infallible but unwrap also isn't const.
        match PointArchetype::from_bits(
            PointArchetype::POSITION.bits()
                | PointArchetype::PRESSURE.bits()
                | PointArchetype::ARC_LENGTH.bits(),
        ) {
            Some(s) => s,
            None => unreachable!(),
        }
    }
}

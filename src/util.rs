//! Utility types, used throughout the crate.
//! This will grow as I refactor eg `(vec2, vec2) -> Rect`

/// A float which is non-NaN
// Because of the preconditions invalidating many bitpatterns, this is not Pod.
#[derive(Copy, Clone, PartialEq, PartialOrd, bytemuck::NoUninit, bytemuck::Zeroable, Debug)]
#[repr(transparent)]
pub struct NonNanF32(f32);
impl NonNanF32 {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);
    /// Wrap a value. None if NaN.
    #[must_use]
    pub fn new(val: f32) -> Option<Self> {
        if val.is_nan() {
            None
        } else {
            Some(Self(val))
        }
    }
}
// This is safe - even though f32 is !Eq, we guarantee that no component is ever NaN
// So PartialEq can act like Eq
impl Eq for NonNanF32 {}
// Doing this on purpose! taking partial ord logic to impl Ord because of struct invariants.
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for NonNanF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Unwrap OK - we guarantee that f32's are non-NaN and thus will never
        // compare as None.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}
/// Premultiplied, linear HDR color.
/// Always non-NaN and well-formed premul color.
// Why cant bytemuck::NoUninit? :(
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, bytemuck::Zeroable, Debug)]
pub struct Color([NonNanF32; 4]);
impl Color {
    pub const TRANSPARENT: Self = Self([NonNanF32::ZERO; 4]);
    pub const WHITE: Self = Self([NonNanF32::ONE; 4]);
    pub const BLACK: Self = Self([
        NonNanF32::ZERO,
        NonNanF32::ZERO,
        NonNanF32::ZERO,
        NonNanF32::ONE,
    ]);
    /// Create a new color from premul linear channels. None if any are NaN
    ///
    /// Normalizes all fully transparent colors to 0.0
    #[must_use]
    pub fn new_lossy(r: f32, g: f32, b: f32, a: f32) -> Option<Self> {
        let s = Self([
            NonNanF32::new(r)?,
            NonNanF32::new(g)?,
            NonNanF32::new(b)?,
            NonNanF32::new(a)?,
        ]);
        // mmmm glorious syntax soup
        if s.0[3].0 == 0.0 {
            Some(Self::BLACK)
        } else {
            Some(s)
        }
    }
}

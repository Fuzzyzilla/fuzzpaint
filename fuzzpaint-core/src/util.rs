//! Utility types, used throughout the crate.
//! This will grow as I refactor eg `(vec2, vec2) -> Rect`

/// A float which is non-NaN
// Because of the preconditions invalidating many bitpatterns, this is not Pod.
#[derive(Copy, Clone, PartialEq, PartialOrd, bytemuck::NoUninit, bytemuck::Zeroable, Debug)]
#[repr(transparent)]
pub struct FiniteF32(f32);
impl FiniteF32 {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);
    pub fn new(val: f32) -> Result<Self, FiniteF32Error> {
        if val.is_finite() {
            Ok(Self(val))
        } else {
            Err(FiniteF32Error::NotFinite)
        }
    }
    #[must_use]
    pub fn get(self) -> f32 {
        // Would `assume` be helpful here? I think, for the most part, floats are impenetrable for the optimizer,
        // so it wouldn't matter anyway :<
        self.0
    }
}

impl TryFrom<f32> for FiniteF32 {
    type Error = FiniteF32Error;
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}
impl From<FiniteF32> for f32 {
    fn from(value: FiniteF32) -> Self {
        value.get()
    }
}

#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiniteF32Error {
    #[error("not finite")]
    NotFinite,
}

// This is safe - even though f32 is !Eq, we guarantee that no component is ever NaN
// So PartialEq can act like Eq
impl Eq for FiniteF32 {}
// Doing this on purpose! taking partial ord logic to impl Ord because of struct invariants.
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for FiniteF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Unwrap OK - we guarantee that the wrapped f32's are non-NaN and thus will never
        // compare as None.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}
impl std::hash::Hash for FiniteF32 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Since we now impl *Eq, we can now impl Hash too!
        // (As x == y is required to imply Hash(x) == Hash(y) which isn't possible with NaN)

        state.write_u32(self.0.to_bits());
    }
}
// Would be fun to impl the operators here too, but unfortunately *None of them* are closed over the set of Non-NaN floats!!
// Ie, Inf - Inf = NaN, 0 * Inf = NaN....
// Even if some were, we can't trust that no FPU is quirked.

/// Premultiplied, linear HDR color.
/// Always non-NaN and well-formed premul color.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, bytemuck::Zeroable, Debug)]
pub struct Color([FiniteF32; 4]);
impl Color {
    pub const TRANSPARENT: Self = Self([FiniteF32::ZERO; 4]);
    pub const WHITE: Self = Self([FiniteF32::ONE; 4]);
    pub const BLACK: Self = Self([
        FiniteF32::ZERO,
        FiniteF32::ZERO,
        FiniteF32::ZERO,
        FiniteF32::ONE,
    ]);
    /// Create a new color from premul linear channels. Normalizes all fully transparent colors to 0.0.
    pub fn new_lossy(r: f32, g: f32, b: f32, a: f32) -> Result<Self, FiniteF32Error> {
        let raw = Self([
            FiniteF32::new(r)?,
            FiniteF32::new(g)?,
            FiniteF32::new(b)?,
            FiniteF32::new(a)?,
        ]);
        // mmmm glorious syntax soup
        if raw.0[3].get() == 0.0 {
            Ok(Self::TRANSPARENT)
        } else {
            Ok(raw)
        }
    }
    /// Create a new color from premul linear channels. Normalizes all fully transparent colors to 0.0.
    pub fn from_array_lossy([r, g, b, a]: [f32; 4]) -> Result<Self, FiniteF32Error> {
        Self::new_lossy(r, g, b, a)
    }
    pub fn as_array(&self) -> [f32; 4] {
        [
            self.0[0].get(),
            self.0[1].get(),
            self.0[2].get(),
            self.0[3].get(),
        ]
    }
    pub fn as_slice(&self) -> &[FiniteF32] {
        self.0.as_slice()
    }
}
// Safety: FiniteF32 is NoUninit, arrays have no uninit bytes of their own.
unsafe impl bytemuck::NoUninit for Color {}

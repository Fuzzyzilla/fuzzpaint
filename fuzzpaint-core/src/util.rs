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

impl Default for FiniteF32 {
    fn default() -> Self {
        Self::ZERO
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

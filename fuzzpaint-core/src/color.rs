use either::Either;

use crate::util::{FiniteF32, FiniteF32Error};

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, bytemuck::Zeroable, Debug, Hash)]
pub struct PaletteIndex(pub u64);

/// Either a premultiplied, linear HDR color, or a palette index.
///
/// This utilizes the niche that all colors with alpha == 0 are normalized to black to store an additional 96 bits of data.
/// If `a == 0u32` and `r != 0u32`, then `[b,g]` encodes the low and high bytes of the palette index, respectively.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, bytemuck::Zeroable, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct ColorOrPalette([u32; 4]);
impl ColorOrPalette {
    pub const TRANSPARENT: Self = Self::from_color(Color::TRANSPARENT);
    pub const WHITE: Self = Self::from_color(Color::WHITE);
    pub const BLACK: Self = Self::from_color(Color::BLACK);
    /// Store a color.
    #[must_use]
    pub const fn from_color(color: Color) -> Self {
        // Safety - transmute OK - Every bitpattern of f32 is a valid bitpattern of u32 (asserted by bytemuck::NoUninit).
        // This also upholds our own invariants - `Color` normalizes all fully transparent colors to [0,0,0,1].
        Self(unsafe { std::mem::transmute(color.as_finite_array()) })
    }
    /// Store a palette index.
    #[must_use]
    pub const fn from_palette_index(index: PaletteIndex) -> Self {
        let low = (index.0 & 0xFFFF_FFFF) as u32;
        let high = ((index.0 >> 32) & 0xFFFF_FFFF) as u32;
        Self([
            // Set R to nonzero to notify reader that a niche is in use.
            // Set B,G to the stored value.
            // Nich is only triggered if a == 0 and any rgb is nonzero.
            1, low, high, 0,
        ])
    }
    /// Get the stored value, whether that is a color or an index.
    #[must_use]
    pub fn get(self) -> Either<Color, PaletteIndex> {
        if self.is_palette() {
            Either::Right(PaletteIndex(
                u64::from(self.0[2]) << 32 | u64::from(self.0[1]),
            ))
        } else {
            // Just a regular ol' color.
            // Safety: Underlying transmute is fine, all u32 bitpatterns are f32 bitpatterns.
            // But we must also uphold Color and FiniteF32's invariants:
            // If the conditions above are not met, then we stored a transmuted color in here unchanged on construction.
            // We have no mutating methods.
            // Since it was a valid color before, it will be valid now.

            // This could actually be a transmute of ref self to ref Color...
            Either::Left(Color(unsafe { std::mem::transmute(self.0) }))
        }
    }
    /// Checks if the contained value is a [`Color`]
    #[must_use]
    pub fn is_color(&self) -> bool {
        !self.is_palette()
    }
    /// Checks if the contained value is a [`PaletteIndex`]
    #[must_use]
    pub fn is_palette(&self) -> bool {
        // Alpha is zero, red is nonzero - a niche is in use!
        self.0[3] == 0 && self.0[0] != 0
    }
}
impl From<Color> for ColorOrPalette {
    fn from(value: Color) -> Self {
        // This upholds our invariants for us. If
        Self::from_color(value)
    }
}
impl From<PaletteIndex> for ColorOrPalette {
    fn from(value: PaletteIndex) -> Self {
        Self::from_palette_index(value)
    }
}
// Safety: FiniteF32 is NoUninit, arrays have no uninit bytes of their own.
unsafe impl bytemuck::NoUninit for ColorOrPalette {}

/// A premultiplied, linear HDR color.
/// All transparent values (alpha == 0) are normalized to transparent black.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, bytemuck::Zeroable, Debug)]
#[allow(clippy::module_name_repetitions)]
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
    pub const fn as_finite_array(&self) -> [FiniteF32; 4] {
        [self.0[0], self.0[1], self.0[2], self.0[3]]
    }
    pub fn as_slice(&self) -> &[FiniteF32] {
        self.0.as_slice()
    }
}
// Safety: FiniteF32 is NoUninit, arrays have no uninit bytes of their own.
unsafe impl bytemuck::NoUninit for Color {}

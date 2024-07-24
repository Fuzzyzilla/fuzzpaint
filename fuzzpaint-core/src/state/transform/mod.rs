/// A transform consisting of an optional horizontal flip, then uniform scale,
/// then rotation, then translation.
///
/// This transform maintains the "Similarity" of shapes and their image, maintaining
/// all angles and the ratios between all lengths.
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Similarity {
    /// Special interpretation: Negative bit set == hflip.
    /// Uniform scale should occur as abs(scale).
    pub flip_scale: f32,
    /// Rotation, in radians *CW* from positive X
    pub rotation: f32,
    /// Translation, in logical pixels. 0,0 is top left, +X Right, +Y down.
    pub translation: [f32; 2],
}

impl Similarity {
    /// Get the horizontal flip flag.
    #[must_use]
    pub fn hflip(&self) -> bool {
        // We want the literal sign bit, regardless of the numerical interpretation.
        self.flip_scale.is_sign_negative()
    }
    /// Get the scale of the similarity. This is not the same as reading from [`Self::flip_scale`]!
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.flip_scale.abs()
    }
    /// Set the horizontal flip flag.
    pub fn set_hflip(&mut self, flip: bool) {
        // We want to set the bit directly, regardless of numerical interpretation.
        // (eg., we don't want negation of zero or NaN to get lost in translation)

        // Reinterpret float ref into u32
        // Safety: Every bitpattern of f32 is a valid u32 and vice-versa.
        let bits = unsafe { &mut *std::ptr::from_mut(&mut self.flip_scale).cast::<u32>() };

        if flip {
            *bits |= 0x8000_0000;
        } else {
            *bits &= 0x7FFF_FFFF;
        }
    }
    /// Set the scale. Negative values will be made positive.
    pub fn set_scale(&mut self, scale: f32) {
        // Funny lil dance :3
        let flip = self.hflip();
        self.flip_scale = scale;
        self.set_hflip(flip);
    }
}

impl Default for Similarity {
    fn default() -> Self {
        Self {
            flip_scale: 1.0,
            rotation: 0.0,
            translation: [0.0; 2],
        }
    }
}

/// An arbitrary transform. Units of output are logical pixels.
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Matrix {
    /// Column-major matrix elements
    pub elements: [[f32; 2]; 3],
}

impl Matrix {
    #[must_use = "Returns a new matrix and does not modify self"]
    pub fn then(&self, other: &Self) -> Self {
        // Compute other * self.
        // Ordered this way for plain english purposes :P
        let r = &self.elements;
        let l = &other.elements;
        Self {
            elements: [
                [
                    r[0][0] * l[0][0] + r[0][1] * l[1][0],
                    r[0][0] * l[0][1] + r[0][1] * l[1][1],
                ],
                [
                    r[1][0] * l[0][0] + r[1][1] * l[1][0],
                    r[1][0] * l[0][1] + r[1][1] * l[1][1],
                ],
                [
                    r[2][0] * l[0][0] + r[2][1] * l[1][0] + l[2][0],
                    r[2][0] * l[0][1] + r[2][1] * l[1][1] + l[2][1],
                ],
            ],
        }
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self {
            elements: [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        }
    }
}

impl From<Similarity> for Matrix {
    fn from(value: Similarity) -> Self {
        // We want the flip bit to mean negative--
        // wait a minute, the flip bit is already stored there!
        let h_scale = value.flip_scale;
        let v_scale = value.scale();

        let (sin, cos) = value.rotation.sin_cos();

        Self {
            // Scale times rotation, and then translate.
            elements: [
                [h_scale * cos, v_scale * sin],
                [h_scale * -sin, v_scale * cos],
                value.translation,
            ],
        }
    }
}

impl From<[[f32; 2]; 3]> for Matrix {
    fn from(elements: [[f32; 2]; 3]) -> Self {
        Self { elements }
    }
}

impl From<Matrix> for [[f32; 2]; 3] {
    fn from(value: Matrix) -> Self {
        value.elements
    }
}

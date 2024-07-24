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
    #[must_use]
    pub fn hflip(&self) -> bool {
        // We want the literal sign bit, regardless of the numerical interpretation.
        self.flip_scale.is_sign_negative()
    }
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.flip_scale.abs()
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

/// A transform that maintains relative angles and lengths of lines. Comprised
/// of:
/// * A flip,
/// * A uniform scale,
/// * A rotation,
/// * and a translation.
///
/// If the scale of the transform is zero, or any of the fields are non-finite,
/// the results of any operations on this type are meaningless.
#[derive(Copy, Clone)]
pub struct Similarity {
    /// Crimes: The sign bit is used to store whether or not the transform
    /// includes a horizontal flip. Negative scales dont need to be represented,
    /// as they can be rewritten as a positive scale with a rotation. Saves 4
    /// bytes.
    scale: f32,
    /// Usually, though not strictly, in the (-TAU, TAU) range.
    rotation: f32,
    translation: ultraviolet::Vec2,
}
impl Default for Similarity {
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Similarity {
    /// The identity transformation.
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        rotation: 0.0,
        translation: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
    };
    /// Returns true if the transform is well-defined.
    /// * The scale is non-zero (approx).
    /// * All fields are finite.
    pub fn valid(&self) -> bool {
        self.scale.is_finite()
            && self.scale != 0.0
            && self.rotation.is_finite()
            && self.translation.x.is_finite()
            && self.translation.y.is_finite()
    }
    /// Get the inverse of this transform.
    #[must_use = "returns a new transform and does not modify `self`"]
    pub fn inverse(&self) -> Self {
        let mut inverse = Self {
            rotation: -self.rotation,
            translation: -self.translation,
            // Always positive.
            scale: 1.0 / self.scale.abs(),
        };
        // Take the opposite of the flip bit from self, and set it into the new
        // scale. (inverse.scale is always sign positive, assuming the scale is
        // valid. Which if it isnt, results are unspecified anyway :3).
        inverse.scale =
            f32::from_bits(!self.scale.to_bits() & 0x8000_0000 | inverse.scale.to_bits());
        inverse
    }
    /// Create a transform interpolated between self and other, *visually*
    /// linear. Notably, this neans that scale is actually interpolated
    /// exponentially.
    ///
    /// When `t` == 0.0, `self` is returned and at `1.0`, `other` is returned.
    #[must_use = "returns a new transform and does not modify `self` or `other`"]
    pub fn visual_lerp_to(&self, t: f32, other: &Self) -> Self {
        use std::f32::consts::{PI, TAU};
        let rev_t = 1.0 - t;

        // Both onto [0, TAU)
        let modular_self_rotation = self.rotation.div_euclid(TAU);
        let modular_other_rotation = other.rotation.div_euclid(TAU);

        // Find the closest angle. E.g. if other is at 5rad and self is at 1rad,
        // it is closer to rotate to -1.28rad (equivalent to 5rad mod TAU) than
        // to rotate all the way to 5.
        let rotation_candidates = [
            modular_other_rotation,
            // If self is less than half a rotation, it'll never be faster to
            // rotate to 1+ rotations, and vice-versa.
            if modular_self_rotation < PI {
                modular_other_rotation - TAU
            } else {
                modular_other_rotation + TAU
            },
        ];

        // Find min by absolute difference to self's rotation
        let abs_differences =
            rotation_candidates.map(|candidate| (candidate - modular_self_rotation).abs());
        let closest_other_rotation = if abs_differences[0] < abs_differences[1] {
            rotation_candidates[0]
        } else {
            rotation_candidates[1]
        };

        // Exponential interpolation, e.g. a lerp from 1x -> 100x will see 10x
        // at t = 0.5. Always positive.
        let scale = (other.scale() / self.scale()).powf(t) * self.scale();

        // FIXME: This aint so hot, visually :3
        let flip = if t < 0.5 {
            self.flip_h()
        } else {
            other.flip_h()
        };

        Self {
            translation: self.translation * rev_t + other.translation * t,
            rotation: modular_self_rotation * rev_t + closest_other_rotation * t,
            scale: if flip {
                // Scale is (mathematically) always positive. This invariant
                // could break with NaNs or zeros or infinities, in which case
                // it is documented that all bets are off anyway :3
                f32::from_bits(0x8000_0000 | scale.to_bits())
            } else {
                scale
            },
        }
    }
    /// Translate by the given distance in the destination coordinate system.
    pub fn translate_by(&mut self, by: ultraviolet::Vec2) {
        self.translation += by;
    }

    /// Rotate by the given radians, such that the `center` point remains fixed.
    /// `center` is in the destination coordinate system.
    ///
    /// Rotation angle is defined as being positive going from +X to +Y.
    pub fn rotate_around(&mut self, by: f32, center: ultraviolet::Vec2) {
        self.rotation += by;
        self.rotation %= std::f32::consts::TAU;
        // Translate so the center is at 0,0, rotate, then translate back.
        self.translation =
            (self.translation - center).rotated_by(ultraviolet::Rotor2::from_angle(by)) + center;
    }

    /// Scale by the given multiplier such that the `center` point remains
    /// fixed. `center` is in the destination coordinate system.
    pub fn scale_around(&mut self, by: f32, center: ultraviolet::Vec2) {
        self.translation = (self.translation - center) * by + center;
        if by < 0.0 {
            self.rotation += std::f32::consts::PI;
        }
        self.scale *= by.abs();
    }

    /// Directly toggle the flip flag. Prefer [`Self::flip_h_around`] and
    /// [`Self::flip_v_around`] for more intuitive operation.
    pub fn toggle_flip_h(&mut self) {
        self.scale = f32::from_bits(0x8000_0000 ^ self.scale.to_bits());
    }
    /// Flip the image horizontally relative to the destination coordinate
    /// system such that the `center` point remains fixed. `center` is in the
    /// destination coordinate system.
    pub fn flip_h_around(&mut self, center_x: f32) {
        self.toggle_flip_h();
        self.rotation = -self.rotation;

        self.translation.x = 2.0 * center_x - self.translation.x;
    }
    /// Flip the image vertically relative to the destination coordinate system
    /// such that the `center` point remains fixed. `center` is in the
    /// destination coordinate system.
    pub fn flip_v_around(&mut self, center_y: f32) {
        self.toggle_flip_h();
        self.rotation = std::f32::consts::PI - self.rotation;

        self.translation.y = 2.0 * center_y - self.translation.y;
    }

    /// Convert into a homogenous transform matrix. Where possible, prefer doing
    /// transformations on this type before converting to a matrix, as they will
    /// be faster and more precise prior to matrixification.
    pub fn into_mat3(&self) -> ultraviolet::Mat3 {
        let mut mat3 = self.into_basis2().into_homogeneous();
        // X, Y, 1
        mat3.cols[2] = self.translation.into_homogeneous_point();

        mat3
    }
    /// Convert into a basis transform, which is missing the translation.
    pub fn into_basis2(&self) -> ultraviolet::Mat2 {
        use ultraviolet::Vec2;
        let (sin, cos) = self.rotation.sin_cos();
        let scale = self.scale();
        let sin = sin * scale;
        let cos = cos * scale;

        let x = Vec2::new(cos, sin);

        ultraviolet::Mat2 {
            cols: [if self.flip_h() { -x } else { x }, Vec2::new(-sin, cos)],
        }
    }

    pub fn translation(&self) -> ultraviolet::Vec2 {
        self.translation
    }
    /// Radians. Make no assumptions about the range this falls in. Value may
    /// change to something geometrically identical (i.e. plus or minus any
    /// multiple of `TAU`) on method calls that should otherwise have no effect
    /// (for example, [`Self::rotate_around`] with an angle of zero.)
    pub fn rotation(&self) -> f32 {
        self.rotation
    }
    /// Scale of the transformation. Never negative, as negative scales get
    /// transparently rewritten as a positive scale with a 180 degree rotation.
    pub fn scale(&self) -> f32 {
        self.scale.abs()
    }
    pub fn flip_h(&self) -> bool {
        self.scale.is_sign_negative()
    }
}
impl From<Similarity> for ultraviolet::Mat3 {
    fn from(value: Similarity) -> Self {
        value.into_mat3()
    }
}
impl std::fmt::Debug for Similarity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut this = f.debug_struct("Similarity");
        this.field("flip_h", &self.flip_h());
        this.field("scale", &self.scale());
        this.field("rotation", &self.rotation());
        this.field("translation", &self.translation());
        this.finish()
    }
}

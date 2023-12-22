use cgmath::prelude::*;

type Decomposed2 = cgmath::Decomposed<cgmath::Vector2<f32>, cgmath::Basis2<f32>>;

/// An affine transform for views. Includes offset, rotation, uniform scale, and horizontal flip.
/// (vertical flipping can be achieved by horizontal flip and rotate 180*)
#[derive(Clone, Copy, Debug)]
pub struct ViewTransform {
    // Marker flag for flipping on the x axis. cgmath::Decomposed cannot represent this.
    // todo: keeping it simple by not implementing this yet.
    // flip_x: bool,

    // current convention is to position based on top-left corner. This is an
    // implementation detail however!
    pub decomposed: Decomposed2,
}

#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    /// The transform cannot be inverted anymore, and has become useless.
    /// Occurs if scale gets too close to zero.
    #[error("uninvertable")]
    Uninvertable,
}

impl ViewTransform {
    /// Is the view flipped (determinate negative)?
    /// Doesn't differentiate between horizontal and vertical flipping.
    #[must_use]
    pub fn is_flipped(&self) -> bool {
        false //self.flip_x
    }
    /// Flip the view horizontally about this center in viewspace such that the x-coordinate of the center
    /// remains in the same spot in the viewport after rotating.
    pub fn flip_x_about(&mut self, _view_center: cgmath::Point2<f32>) {
        todo!()
        // let local_center = self.unproject(view_center);
        // transform such that center is at 0,0
        // flip
        // transform back
        //todo!()
    }
    /// Rotate about this center in viewspace such that the center remains in the same spot in the viewport after rotating.
    pub fn rotate_about(&mut self, view_center: cgmath::Point2<f32>, rotate: cgmath::Rad<f32>) {
        // vec from mouse to top-left
        let local_center = view_center.to_vec() - self.decomposed.disp;
        let rotate = cgmath::Basis2::from_angle(rotate);

        let local_center = rotate.rotate_vector(local_center);
        self.decomposed.rot = rotate * self.decomposed.rot;
        self.decomposed.disp = view_center.to_vec() - local_center;
    }
    /// Scale about this center in viewspace such that the center remains in the same spot in the viewport after scaling.
    pub fn scale_about(&mut self, view_center: cgmath::Point2<f32>, scale_by: f32) {
        // vec from mouse to top-left
        let local_center = view_center.to_vec() - self.decomposed.disp;

        // Scale, then adjust translation.
        self.decomposed.scale *= scale_by;
        // Scale that vec from mouse to top-left by the same factor.
        self.decomposed.disp = view_center.to_vec() - (local_center * scale_by);
    }
    /// Pan by this displacement in viewspace.
    pub fn pan(&mut self, delta: cgmath::Vector2<f32>) {
        self.decomposed.disp += delta;
    }
    /// Convert this point in view space to local space
    pub fn unproject(
        &self,
        view_point: cgmath::Point2<f32>,
    ) -> Result<cgmath::Point2<f32>, TransformError> {
        Ok(self
            .decomposed
            .inverse_transform()
            .ok_or(TransformError::Uninvertable)?
            .transform_point(view_point))
    }
    /// Convert this point in local space to view space
    #[must_use]
    pub fn project(&self, local_point: cgmath::Point2<f32>) -> cgmath::Point2<f32> {
        self.decomposed.transform_point(local_point)
    }
    /// Create a transform where the document's center is located at `view_center`
    #[must_use]
    pub fn center_on(
        view_center: cgmath::Point2<f32>,
        document_size: cgmath::Vector2<f32>,
        rotation: cgmath::Rad<f32>,
        scale: f32,
    ) -> Self {
        let rot = cgmath::Basis2::from_angle(rotation);
        let disp = view_center.to_vec() - scale * rot.rotate_vector(document_size / 2.0);

        Self {
            decomposed: Decomposed2 { scale, rot, disp },
        }
    }
    /// Scale self by this input scale factor.
    /// e.g., with a `factor` of 2, the point formerly selected by `self` at (10, 20) will now be selected by (20, 40).
    #[must_use]
    pub fn with_scale_factor(mut self, factor: f32) -> Self {
        // We know the 0,0 point in viewspace must remain fixed
        self.scale_about(cgmath::Point2 { x: 0.0, y: 0.0 }, factor);
        self
    }
    /// Calculate the effective "scale" of the document, as viewed by this viewport.
    /// e.g. Some(2.0) means each document unit maps to two viewport units.
    #[must_use]
    pub fn view_points_per_document_point(&self) -> f32 {
        self.decomposed.scale
    }
}

impl From<ViewTransform> for cgmath::Matrix3<f32> {
    fn from(value: ViewTransform) -> Self {
        value.decomposed.into()
    }
}
impl From<ViewTransform> for cgmath::Matrix4<f32> {
    #[rustfmt::skip]
    fn from(value: ViewTransform) -> Self {
        let mat3 = cgmath::Matrix3::<f32>::from(value.decomposed);
        // Is this the same op as mat3.into()?
        // found out - it's NOT! keep doin this :>
        Self {
            x: cgmath::Vector4 {x: mat3.x.x, y: mat3.x.y, z: 0.0, w: mat3.x.z,},
            y: cgmath::Vector4 {x: mat3.y.x, y: mat3.y.y, z: 0.0, w: mat3.y.z,},
            z: cgmath::Vector4 {x: 0.0,      y: 0.0,      z: 1.0, w: 0.0,     },
            w: cgmath::Vector4 {x: mat3.z.x, y: mat3.z.y, z: 0.0, w: mat3.z.z,},
        }
    }
}

#[derive(Clone, Copy)]
pub struct DocumentFit {
    pub flip_x: bool,
    pub rotation: cgmath::Rad<f32>,
    pub margin: f32,
}

impl DocumentFit {
    /// Make a transform from the given document size and viewport rect (pos, size)
    /// Returns `None` if the resulting scale is too small to be reasonably caclulated or used.
    #[must_use]
    pub fn make_transform(
        &self,
        document_size: cgmath::Vector2<f32>,
        view_pos: cgmath::Point2<f32>,
        view_size: cgmath::Vector2<f32>,
    ) -> Option<ViewTransform> {
        // rotate two rays. These will give us the max bounds of the rotated document.
        // probably a easier and less literal way to do this x3
        let bottom_right_corner_ray = document_size / 2.0;
        let bottom_left_corner_ray = {
            let mut ray = document_size / 2.0;
            ray.x *= -1.0;
            ray
        };

        let rot_basis = cgmath::Basis2::from_angle(self.rotation);
        let bottom_right_corner_ray = rot_basis.rotate_vector(bottom_right_corner_ray);
        let bottom_left_corner_ray = rot_basis.rotate_vector(bottom_left_corner_ray);

        let half_max_range = cgmath::vec2(
            bottom_left_corner_ray
                .x
                .abs()
                .max(bottom_right_corner_ray.x.abs()),
            bottom_left_corner_ray
                .y
                .abs()
                .max(bottom_right_corner_ray.y.abs()),
        );
        // Adjust viewport for margin.
        let view_pos_margin = view_pos + cgmath::vec2(self.margin, self.margin);
        let view_size_margin = view_size - 2.0 * cgmath::vec2(self.margin, self.margin);

        // pretend the document is the bounding rect of the rotated document
        let document_size = half_max_range * 2.0;

        // Calculate x,y fitting scales. Choose the smaller scale.
        let document_scales = cgmath::vec2(
            view_size_margin.x / document_size.x,
            view_size_margin.y / document_size.y,
        );
        let document_scale = document_scales.x.min(document_scales.y);

        if document_scale < 0.001 {
            None
        } else {
            Some(ViewTransform::center_on(
                view_pos_margin + view_size_margin / 2.0,
                document_size,
                self.rotation,
                document_scale,
            ))
        }
    }
    #[must_use]
    pub fn with_scale_factor(self, factor: f32) -> Self {
        Self {
            margin: self.margin * factor,
            ..self
        }
    }
}

impl Default for DocumentFit {
    fn default() -> Self {
        Self {
            flip_x: false,
            margin: 8.0,
            rotation: Zero::zero(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum DocumentTransform {
    /// Auto positioned and sized, with given flip and rotation, to fit into the viewport.
    ///
    /// Used in initial document state and zoom-fit state
    Fit(DocumentFit),
    // Opted out of a center state, which would pretty much only be if the user was in Fit state and
    // used zoom control (eg. ctrl+plus) and then didn't move the viewport.
    // That's wildly niche, just use custom transform instead UwU
    /// User-defined transform, after a scrub or drag.
    Transform(ViewTransform),
}
impl Default for DocumentTransform {
    fn default() -> Self {
        Self::Fit(DocumentFit::default())
    }
}
impl DocumentTransform {
    #[must_use]
    pub fn with_scale_factor(self, factor: f32) -> Self {
        match self {
            DocumentTransform::Fit(f) => DocumentTransform::Fit(f.with_scale_factor(factor)),
            DocumentTransform::Transform(f) => {
                DocumentTransform::Transform(f.with_scale_factor(factor))
            }
        }
    }
}
#[derive(Clone, Copy)]
pub struct ViewInfo {
    pub transform: crate::view_transform::DocumentTransform,
    pub viewport_position: ultraviolet::Vec2,
    pub viewport_size: ultraviolet::Vec2,
}
impl ViewInfo {
    #[must_use]
    pub fn center(&self) -> ultraviolet::Vec2 {
        self.viewport_position + self.viewport_size / 2.0
    }
    /// Make an explicit transform out of this view.
    /// Leaves `DocumentTransform::Transform` as-is, and converts `DocumentTransform::Fit` into `ViewTransform`
    /// `None` if too small to be usable.
    #[must_use]
    pub fn calculate_transform(&self) -> Option<crate::view_transform::ViewTransform> {
        match &self.transform {
            crate::view_transform::DocumentTransform::Fit(f) => f.make_transform(
                cgmath::Vector2 {
                    x: crate::DOCUMENT_DIMENSION as f32,
                    y: crate::DOCUMENT_DIMENSION as f32,
                },
                cgmath::Point2 {
                    x: self.viewport_position.x,
                    y: self.viewport_position.y,
                },
                cgmath::Vector2 {
                    x: self.viewport_size.x,
                    y: self.viewport_size.y,
                },
            ),
            crate::view_transform::DocumentTransform::Transform(t) => Some(*t),
        }
    }
    /// Convert self in-place into a `ViewTransform` representation, returning mutable access to that transform.
    /// `None` if too small to be usable.
    pub fn make_transformed(&mut self) -> Option<&mut crate::view_transform::ViewTransform> {
        let xform = self.calculate_transform()?;
        self.transform = crate::view_transform::DocumentTransform::Transform(xform);
        let crate::view_transform::DocumentTransform::Transform(xform) = &mut self.transform else {
            unreachable!()
        };

        Some(xform)
    }
    /// Given a scale factor (relative to `self`), calculate a new view info.
    #[must_use]
    pub fn with_scale_factor(self, factor: f32) -> Self {
        Self {
            viewport_position: self.viewport_position * factor,
            viewport_size: self.viewport_size * factor,
            transform: self.transform.with_scale_factor(factor),
        }
    }
    /// Calculate the position and size of the AABB the viewport covers, in document space.
    /// Size will be non-negative. The rectangle is not clamped to the bounds of the document.
    // I really need a rectangle type x3
    #[must_use]
    pub fn document_space_aabb(&self) -> Option<(cgmath::Point2<f32>, cgmath::Vector2<f32>)> {
        let xform = self.calculate_transform()?;
        let uv_to_cg = |ultraviolet::Vec2 { x, y }| cgmath::Point2 { x, y };
        // Get corners of the viewport...
        // This could be done with only three corners, but my brain too smol
        // unwraps ok - if the first one succeeds, the rest will as well.
        let points = [
            xform.unproject(uv_to_cg(self.viewport_position)).ok()?,
            xform
                .unproject(uv_to_cg(
                    self.viewport_position
                        + ultraviolet::Vec2 {
                            x: 0.0,
                            ..self.viewport_size
                        },
                ))
                .unwrap(),
            xform
                .unproject(uv_to_cg(
                    self.viewport_position
                        + ultraviolet::Vec2 {
                            y: 0.0,
                            ..self.viewport_size
                        },
                ))
                .unwrap(),
            xform
                .unproject(uv_to_cg(self.viewport_position + self.viewport_size))
                .unwrap(),
        ];
        // Unwrap ok - of course this array isn't empty!
        // Use total ordering to silently ignore NaN/Inf
        let min_x = points
            .iter()
            .map(|p| p.x)
            .min_by(|a, b| a.total_cmp(&b))
            .unwrap();
        let max_x = points
            .iter()
            .map(|p| p.x)
            .max_by(|a, b| a.total_cmp(&b))
            .unwrap();
        let min_y = points
            .iter()
            .map(|p| p.y)
            .min_by(|a, b| a.total_cmp(&b))
            .unwrap();
        let max_y = points
            .iter()
            .map(|p| p.y)
            .max_by(|a, b| a.total_cmp(&b))
            .unwrap();

        Some((
            cgmath::Point2 { x: min_x, y: min_y },
            cgmath::Vector2 {
                x: max_x - min_x,
                y: max_y - min_y,
            },
        ))
    }
}

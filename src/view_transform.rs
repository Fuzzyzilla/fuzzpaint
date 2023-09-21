use cgmath::prelude::*;

type Decomposed2 = cgmath::Decomposed<cgmath::Vector2<f32>, cgmath::Basis2<f32>>;

/// An affine transform for views. Includes offset, rotation, uniform scale, and horizontal flip.
/// (vertical flipping can be achieved by horizontal flip and rotate 180*)
#[derive(Clone)]
pub struct ViewTransform {
    // Marker flag for flipping on the x axis. cgmath::Decomposed cannot represent this.
    // todo: keeping it simple by not implementing this yet.
    // flip_x: bool,
    decomposed: Decomposed2,
}
pub enum TransformError {
    /// The transform cannot be inverted anymore, and has become useless.
    /// Occurs if scale gets too close to zero.
    Uninvertable,
}
impl ViewTransform {
    /// Is the view flipped (determinate negative)?
    /// Doesn't differentiate between horizontal and vertical flipping.
    pub fn is_flipped(&self) -> bool {
        false //self.flip_x
    }
    /// Flip the view horizontally about this center in viewspace such that the x-coordinate of the center
    /// remains in the same spot in the viewport after rotating.
    pub fn flip_x_about(&mut self, view_center: cgmath::Point2<f32>) {
        let local_center = self.unproject(view_center);
        todo!()
        // transform such that center is at 0,0
        // flip
        // transform back
        //todo!()
    }
    /// Rotate about this center in viewspace such that the center remains in the same spot in the viewport after rotating.
    pub fn rotate_about(
        &mut self,
        view_center: cgmath::Point2<f32>,
        rotate: cgmath::Rad<f32>,
    ) -> Result<(), TransformError> {
        let local_center = self.unproject(view_center)?.to_vec();
        // Translate so that local center is at origin
        self.decomposed.concat_self(&Decomposed2 {
            disp: -1.0 * local_center,
            rot: One::one(),
            scale: 1.0,
        });
        // then rotate
        self.decomposed.concat_self(&Decomposed2 {
            disp: Zero::zero(),
            rot: Rotation2::from_angle(rotate),
            scale: 1.0,
        });
        // then translate back.
        self.decomposed.concat_self(&Decomposed2 {
            disp: local_center,
            rot: One::one(),
            scale: 1.0,
        });

        Ok(())
    }
    /// Scale about this center in viewspace such that the center remains in the same spot in the viewport after scaling.
    pub fn scale_about(
        &mut self,
        view_center: cgmath::Point2<f32>,
        scale_by: f32,
    ) -> Result<(), TransformError> {
        let local_center = self.unproject(view_center)?.to_vec();
        // Translate so that local center is at origin
        self.decomposed.concat_self(&Decomposed2 {
            disp: -1.0 * local_center,
            rot: One::one(),
            scale: 1.0,
        });
        // then scale. May result in an un-invertible matrix, but this will only be reported later.
        self.decomposed.scale *= scale_by;
        // then translate back.
        self.decomposed.concat_self(&Decomposed2 {
            disp: local_center,
            rot: One::one(),
            scale: 1.0,
        });

        Ok(())
    }
    /// Pan by this displacement in viewspace.
    pub fn pan(&mut self, delta: cgmath::Vector2<f32>) -> Result<(), TransformError> {
        let local_delta = self
            .decomposed
            .inverse_transform_vector(delta)
            .ok_or(TransformError::Uninvertable)?;

        self.decomposed.disp += local_delta;

        Ok(())
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
    pub fn project(&self, local_point: cgmath::Point2<f32>) -> cgmath::Point2<f32> {
        self.decomposed.transform_point(local_point)
    }
}

impl From<ViewTransform> for cgmath::Matrix3<f32> {
    fn from(value: ViewTransform) -> Self {
        value.decomposed.into()
    }
}

pub struct DocumentFit {
    pub flip_x: bool,
    pub rotation: cgmath::Rad<f32>,
}

impl DocumentFit {
    /// Make a transform from the given document size and viewport rect (pos, size)
    pub fn make_transform(
        &self,
        document_size: cgmath::Vector2<f32>,
        (view_pos, view_size): (cgmath::Point2<f32>, cgmath::Vector2<f32>),
    ) -> Option<ViewTransform> {
        let widest_viewport_dimension = view_size.x.max(view_size.y);

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

        // new fitting rectangle for document after rotation.
        let document_size = half_max_range * 2.0;

        // Margin around document
        const MARGIN: f32 = 0.0;
        let document_max_dimension = document_size.x.max(document_size.y) + MARGIN;
        let document_scale = widest_viewport_dimension / document_max_dimension;

        if document_scale < 0.001 {
            None
        } else {
            Some(ViewTransform {
                decomposed: cgmath::Decomposed {
                    scale: document_scale,
                    rot: rot_basis,
                    // Wrong vector, todo.
                    disp: view_pos.to_vec(),
                },
            })
        }
    }
}

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

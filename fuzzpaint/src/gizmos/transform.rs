/// The origin of the gizmo will be pinned according to it's position and this value.
///
/// As of right now, no viewport pinning - that's a non-goal of this API. use egui for that ;)
pub enum OriginPinning {
    /// The origin of the gizmo is pinned to a specific pixel location on the document
    Document,
    /// Position is in parent's coordinate space.
    /// Takes the parent's size and position (and pinning thereof) into account.
    ///
    /// A top-level gizmo set to `Inherit` will behave as `Document`.
    Inherit,
}
/// The coordinate system of the gizmo will be calculated according to it's size, rotation and this value.
/// Size and rotation are pinned separately, but use the same logic.
pub enum BasisPinning {
    /// Size is in document pixels. Rotation is relative to the document space.
    Document,
    /// Size is in viewport logical pixels, rotation is absolute.
    Viewport,
    /// Calculate based on parent's transform.
    ///
    /// A top-level gizmo set to `Inherit` will behave as `Document`.
    Inherit,
}

pub struct Transform {
    pub position: ultraviolet::Vec2,
    pub origin_pinning: OriginPinning,
    pub scale_pinning: BasisPinning,
    pub rotation: f32,
    pub rotation_pinning: BasisPinning,
}
impl Transform {
    /// Apply this gizmo transform to the given document and parent gizmo transforms, returning a new transform representing
    /// this gizmo transform's local space. For top level gizmos, it is valid for `parent_transform` to equal `document_transform`.
    #[must_use]
    pub fn apply(
        &self,
        document_transform: &crate::view_transform::ViewTransform,
        parent_transform: &crate::view_transform::ViewTransform,
    ) -> crate::view_transform::ViewTransform {
        use cgmath::{EuclideanSpace, Rotation2};

        let disp = match self.origin_pinning {
            OriginPinning::Document => {
                let pos = cgmath::point2(self.position.x, self.position.y);
                document_transform.project(pos)
            }
            OriginPinning::Inherit => {
                let pos = cgmath::point2(self.position.x, self.position.y);
                parent_transform.project(pos)
            }
        }
        .to_vec();
        let scale = match self.scale_pinning {
            BasisPinning::Document => document_transform.decomposed.scale,
            BasisPinning::Inherit => parent_transform.decomposed.scale,
            BasisPinning::Viewport => 1.0,
        };
        let rotation = cgmath::Basis2::from_angle(cgmath::Rad(self.rotation));
        let rot = match self.rotation_pinning {
            BasisPinning::Document => document_transform.decomposed.rot * rotation,
            BasisPinning::Inherit => parent_transform.decomposed.rot * rotation,
            BasisPinning::Viewport => rotation,
        };

        crate::view_transform::ViewTransform {
            decomposed: cgmath::Decomposed { scale, rot, disp },
        }
    }
    #[must_use]
    pub fn inherit_all() -> Self {
        Self {
            origin_pinning: OriginPinning::Inherit,
            rotation_pinning: BasisPinning::Inherit,
            scale_pinning: BasisPinning::Inherit,
            position: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
            rotation: 0.0,
        }
    }
}

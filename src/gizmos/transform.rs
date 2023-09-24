/// The origin of the gizmo will be pinned according to it's position and this value.
///
/// As of right now, no viewport pinning - that's a non-goal of this API. use egui for that ;)
pub enum GizmoOriginPinning {
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
pub enum GizmoTransformPinning {
    /// Size is in document pixels. Rotation is relative to the document space.
    Document,
    /// Size is in viewport logical pixels, rotation is absolute.
    Viewport,
    /// Calculate based on parent's transform.
    ///
    /// A top-level gizmo set to `Inherit` will behave as `Document`.
    Inherit,
}

pub struct GizmoTransform {
    pub position: ultraviolet::Vec2,
    pub origin_pinning: GizmoOriginPinning,
    pub scale_pinning: GizmoTransformPinning,
    pub rotation: f32,
    pub rotation_pinning: GizmoTransformPinning,
}
impl GizmoTransform {
    /// Apply this gizmo transform to the given document and parent gizmo transforms, returning a new transform representing
    /// this gizmo transform's local space. For top level gizmos, it is valid for parent_transform to equal document_transform.
    pub fn apply(
        &self,
        document_transform: &crate::view_transform::ViewTransform,
        parent_transform: &crate::view_transform::ViewTransform,
    ) -> crate::view_transform::ViewTransform {
        use cgmath::{EuclideanSpace, Rotation2};

        let disp = match self.origin_pinning {
            GizmoOriginPinning::Document => {
                let pos = cgmath::point2(self.position.x, self.position.y);
                document_transform.project(pos)
            }
            GizmoOriginPinning::Inherit => {
                let pos = cgmath::point2(self.position.x, self.position.y);
                parent_transform.project(pos)
            }
        }
        .to_vec();
        let scale = match self.scale_pinning {
            GizmoTransformPinning::Document => document_transform.decomposed.scale,
            GizmoTransformPinning::Inherit => parent_transform.decomposed.scale,
            GizmoTransformPinning::Viewport => 1.0,
        };
        let rotation = cgmath::Basis2::from_angle(cgmath::Rad(self.rotation));
        let rot = match self.rotation_pinning {
            GizmoTransformPinning::Document => document_transform.decomposed.rot * rotation,
            GizmoTransformPinning::Inherit => parent_transform.decomposed.rot * rotation,
            GizmoTransformPinning::Viewport => rotation,
        };

        crate::view_transform::ViewTransform {
            decomposed: cgmath::Decomposed { scale, rot, disp },
        }
    }
}

/// Metadata for a click event, required to deal with transforms as the gizmo tree is searched for a hit.
pub struct ClickInfo {
    /// Transform of this gizmo's parent, relative to the viewport coordinates.
    parent_transform: crate::view_transform::ViewTransform,
    /// Transform of the viewport coordinates into document coordinates.
    viewport_to_document_transform: crate::view_transform::ViewTransform,
    /// Where in the viewport this click occured
    coords_viewport: ultraviolet::Vec2,
    /// Where in the document this click occured
    coords_document: ultraviolet::Vec2,
    /// Where in the local space this click occured
    coords_local: ultraviolet::Vec2,
}
impl ClickInfo {
    /// Create a info for the document transform and the given viewport mouse coordinate.
    /// `document_transform` maps viewport coordinates to document coordinates.
    ///
    /// Returns None if `document_transform` is not an invertable transform (ie. if scale = ~0.0).
    pub fn new(
        document_transform: crate::view_transform::ViewTransform,
        click_coord_viewport: ultraviolet::Vec2,
    ) -> Option<Self> {
        todo!()
    }
}

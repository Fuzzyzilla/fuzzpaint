use cgmath::Point2;


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
    pub position: [f32; 2],
    pub origin_pinning: GizmoOriginPinning,
    pub scale_pinning: GizmoTransformPinning,
    pub rotation: f32,
    pub rotation_pinning: GizmoTransformPinning,
}

/// An abosolute transformation, the units of which are contextual.
#[derive(Copy, Clone)]
pub struct AbsoluteTransform {
    /// Where the 0,0 point of this object lands on the absolute coordinates.
    origin: Point2<f32>,
    /// Rotation in radians, + is CCW. 0 is in the direction of the absolute X axis.
    rotation: f32,
    /// How many absolute units per self unit?
    scale: f32,
}
impl AbsoluteTransform {
    const SCALE_EPSILON : f32 = 0.0001;
    /// Invert the transform, returning it. None if not invertable (ie scale = ~0.0)
    #[must_use = "Does not modify self - returns an inverted copy"]
    pub fn invert(&self) -> Option<Self> {
        if self.scale < Self::SCALE_EPSILON {
            None
        } else {
            Some(
                // This feels too simple to be right lol
                Self {
                    origin: -1.0 * self.origin,
                    rotation: -self.rotation,
                    scale: 1.0/self.scale,
                }
            )
        }
    }
    /// Project a point from the local space into the absolute space.
    pub fn project(&self, point: Point2<f32>) -> Point2<f32> {
        todo!()
    }
}

/// Metadata for a click event, required to deal with transforms as the gizmo tree is searched for a hit.
pub struct ClickInfo {
    /// Transform of this gizmo's parent, relative to the viewport coordinates.
    parent_transform: AbsoluteTransform,
    /// Transform of the viewport coordinates into document coordinates.
    viewport_to_document_transform: AbsoluteTransform,
    /// Where in the viewport this click occured
    coords_viewport: [f32; 2],
    /// Where in the document this click occured
    coords_document: [f32; 2],
    /// Where in the local space this click occured
    coords_local: [f32; 2],
}
impl ClickInfo {
    /// Create a info for the document transform and the given viewport mouse coordinate.
    /// `document_transform` maps viewport coordinates to document coordinates.
    /// 
    /// Returns None if `document_transform` is not an invertable transform (ie. if scale = ~0.0).
    pub fn new(document_transform: AbsoluteTransform, click_coord_viewport: [f32; 2]) -> Option<Self> {
        todo!()
    }
}
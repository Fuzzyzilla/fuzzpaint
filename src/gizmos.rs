//! # Gizmos
//!
//! Represents an interactive overlay atop the document editing workspace but behind the UI workspace. Useful for interactive
//! elements, mouse hit detection, rendering immediate previews, ect. without re-rendering the document.
//! 
//! More complex gizmos are made up of simple composable elements.
//
// (Todo: Should crate::document_viewport_proxy be a kind of gizmo? the parallels are clear...)

/// The origin of the gizmo will be pinned according to it's position and this value.
///
/// As of right now, no viewport pinning - that's a non-goal of this API. use egui for that ;)
pub enum GizmoOriginPinning {
    /// The origin of the gizmo is pinned to a specific pixel location on the document
    Document,
    /// The origin of the gizmo follows the mouse.
    Mouse,
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

pub enum GizmoMeshStyle {
    Triangles,
    LineStrip,
}

/// How is a gizmo displayed?
/// For efficiency in rendering, the options are intentionally limited.
/// For more complex visuals, several gizmos may be combined arbitrarily.
pub enum GizmoVisual {
    Mesh{
        style: GizmoMeshStyle,
        /// The descriptor of the texture. Should be immutable, as read usage
        /// lifetime is not currently bounded.
        /// 
        /// Set binding 0 should be the combined image sampler, which will be rendered with
        /// standard alpha blending.
        texture: Option<crate::vk::PersistentDescriptorSet>,
        /// Packed floats: \[X,Y,  R,G,B,A,  U,V\]
        /// 
        /// Color will be multiplied with the texture sampled at UV, or white if no texture.
        /// If no texture, UV is ignored and may be invalid.
        mesh: (),
        /// Whether the mesh can mutate from frame-to-frame.
        /// If true, it will be re-uploaded to the GPU every frame,
        /// otherwise it may be cached and changes may be missed
        mutable: bool,
    },
    None,
}

/// How can a gizmo be interacted with by the mouse?
#[repr(u8)]
pub enum GizmoInteraction {
    None,
    /// Can be dragged, and arbitrarily constrained.
    Move,
    /// Can be clicked to open
    Open,
    /// Both `Move`-able and `Open`-able.
    MoveOpen,
    /// Can be rotated around it's origin by dragging, can be arbitrarily constrained.
    Rotate,
}

pub struct Gizmo {
    visual: GizmoVisual,
    position: ([f32; 2], GizmoOriginPinning),
    scale: ([f32; 2], GizmoTransformPinning),
    rotation: (f32, GizmoTransformPinning)
}
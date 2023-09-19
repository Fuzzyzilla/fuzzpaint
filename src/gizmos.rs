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
/// For more complex visuals, combined several gizmos in a group.
pub enum GizmoVisual {
    Mesh{
        /// Interpret mesh as TriangleList or as a wide LineStrip?
        /// Texturing is supported for lines.
        style: GizmoMeshStyle,
        /// The descriptor of the texture. Should be immutable, as read usage
        /// lifetime is not currently bounded.
        /// 
        /// Set binding 0 should be the combined image sampler, which will be rendered with
        /// standard alpha blending.
        texture: Option<crate::vk::PersistentDescriptorSet>,
        /// Color modulation of this gizmo. Can be changed at will, and will be respected by the renderer.
        color: [f32; 4],
        /// Packed f32s: \[X,Y,  R,G,B,A,  U,V\]
        /// 
        /// Coordinates are in logical pixels or document pixels, as determined by GizmoTransformPinning.
        /// 
        /// Color will be multiplied with the texture sampled at UV (or white if no texture), and
        /// further multiplied by `Mesh::color`.
        /// 
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
pub enum GizmoInteraction {
    None,
    /// Can be dragged, and arbitrarily constrained.
    Move,
    /// Can be clicked to open
    Open,
    /// Both `Move`-able and `Open`-able.
    MoveOpen,
    /// Can be rotated around its origin by dragging, can be arbitrarily constrained.
    Rotate,
}

/// A kind of inverse iterator, where the visitor be passed down the whole
/// tree to visit every gizmo in order.
pub trait GizmoVisitor {
    fn visit_gizmo(&mut self, gizmo: &Gizmo);
    fn visit_collection(&mut self, gizmo: &Collection);
}

pub struct Gizmo {
    visual: GizmoVisual,
    interaction: GizmoInteraction,
    position: ([f32; 2], GizmoOriginPinning),
    scale: ([f32; 2], GizmoTransformPinning),
    rotation: (f32, GizmoTransformPinning)
}

/// A collection of many gizmos. It itself is a Gizmo,
/// meaning Collections-in-Collections is supported.
pub struct Collection {
    this_gizmo: Gizmo,
    /// Path to the currently open gizmo. (todo!)
    open: Option<()>,
    /// Children of this gizmo, sorted top to bottom. 
    children: Vec<Gizmo>,
}

mod seal {
    pub trait _Sealed {}
}

use winit::window::CursorIcon as CursorIcon;
/// None to hide the cursor, or Some to choose a winit cursor.
type CursorOrInvisible = Option<CursorIcon>;
// Idk what to name this lol
/// Sealed, because we assume Gizmo and GizmoCollection are the only two valid
/// gizmos. Just keeps logic clean, and that's the whole point of the composable style
/// of this API :3
pub trait Gizmooooo : seal::_Sealed {
    type Meta;
    /// Bounding box for hit checks, in the parent's coordinate space. Purely optimization, None is always valid.
    /// Gizmos may escape their bounding box visually, but inputs *may* be skipped.
    fn hit_bounding_box(&self) -> Option<()>;
    /// If hovering this local coordinate, what cursor do we show?
    fn cursor_at(&self, point:[f32;2]) -> CursorOrInvisible;
    /// While grabbed with this path, what cursor do we show?
    fn grabbed_cursor(&self, path: &Self::Meta) -> CursorOrInvisible;
    /// A click was registered to this gizmo - return some metadata to allow future tracking,
    /// or None to pass-thru the click event.
    fn click_at(&self, point: [f32; 2]) -> Option<Self::Meta>;
    /// The mouse dragged by delta viewport pixels after a click.
    /// 
    /// May be smaller or larger than the physical distance travelled by the
    /// mouse, to allow things like holding ctrl to drag more precisely or shift to drag more coursely.
    fn dragged_delta(&self, path: &Self::Meta, delta: [f32;2]);
    /// The mouse stopped dragging. Returns ownership of the Meta given when the 
    /// mouse first clicked this gizmo.
    fn drag_release(&self, path: Self::Meta);
    /// The mouse clicked the gizmo. Drags may have been emitted, but it is retroactively treated
    /// as a click instead. This is detected for example if the cumulative drag delta is sufficiently small after releasing.
    fn click_release(&self, path: Self::Meta);
    /// Pass the visitor to self and all children!
    /// Should visit in painters order, back-to-front.
    fn visit(&self, visitor: &mut impl GizmoVisitor);
}

// Possible types of path emitted by a gizmo.
struct GizmoMeta {
    /// Offset of the mouse at the time of mouse down from this gizmo's origin,
    /// in units determined by GizmoTransformPinning
    offs: [f32; 2],
}
/// Some number of indicies to drill down into the nested structure,
/// followed by the terminating gizmo metadata.
struct CollectionMeta(Vec<usize>, GizmoMeta);

enum AnyMeta {
    Gizmo(GizmoMeta),
    Collection(CollectionMeta),
}
/// Newtype around all these implementation details to allow outside code
/// to store and work with these paths.
pub struct Path(AnyMeta);
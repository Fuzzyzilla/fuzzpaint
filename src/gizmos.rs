//! # Gizmos
//!
//! Represents an interactive overlay atop the document editing workspace but behind the UI workspace. Useful for interactive
//! elements, mouse hit detection, rendering immediate previews, ect. without re-rendering the document.

// (Todo: Should crate::document_viewport_proxy be a kind of gizmo? the parallels are clear...)

/// The structure that manages the rendering of gizmos. Exists on a per-workspace basis, not a per-document basis!
pub struct GizmoCollection {}

/// A gizmo. May simply be a visual, implement mouse hit detection, set a cursor icon, ect.
pub trait Gizmo {
    type DragMetadata: Sized;
    /// Cursor to display when the gizmo is hovered. None to hide.
    fn hover_cursor(&self) -> Option<winit::window::CursorIcon>;
    /// Cursor to display when the gizmo has been clicked on and held. Defaults to `hover_cursor`.
    fn grab_cursor(&self) -> Option<winit::window::CursorIcon> {
        self.hover_cursor()
    }
    /// Check for a hit at the given **document coordinate**.
    fn hit(&self, point: [f32; 2]) -> bool;
    /// Check for click at the given **document coordinate**.
    ///
    /// Returns a metadata object that should be given to drag_to.
    fn click(&self, point: [f32; 2]) -> Option<Self::DragMetadata>;
    /// This gizmo was clicked and dragged to this point in **document space**.
    /// The metadata from click or click_boxed will be passed as an argument, which is useful
    /// e.g. to track which part of a complex gizmo is being interacted with.
    /// Will update frequently as a drag is in progress.
    ///
    /// To implement snapping or constraints, it's safe to modify the given point or ignore it entirely.
    fn drag_to(&self, point: [f32; 2], metadata: &Self::DragMetadata);
    /// Mouse stopped dragging. last point to `drag_to` was the final position. Returns ownership
    /// of the metadata back to the gizmo.
    fn release(&self, metadata: Self::DragMetadata);
}

/// This feels like a C++ style solution, unrusty!!
/// Although, it is an implementation detail, and only leaks
/// the trait bond `Gizmo::DragMetadata: Any + Clone` to the outside world.
///
/// Stores a gizmo in a type-erased fasion.
struct AnyGizmo<G: Gizmo>(G);
impl<Meta, G: Gizmo<DragMetadata = Meta>> Gizmo for AnyGizmo<G>
where
    Meta: std::any::Any + Clone,
{
    type DragMetadata = Box<dyn std::any::Any>;
    fn hover_cursor(&self) -> Option<winit::window::CursorIcon> {
        self.0.hover_cursor()
    }
    fn grab_cursor(&self) -> Option<winit::window::CursorIcon> {
        self.0.grab_cursor()
    }
    fn hit(&self, point: [f32; 2]) -> bool {
        self.0.hit(point)
    }
    fn click(&self, point: [f32; 2]) -> Option<Self::DragMetadata> {
        self.0
            .click(point)
            .map(|meta| Box::new(meta) as Box<(dyn std::any::Any)>)
    }
    fn drag_to(&self, point: [f32; 2], metadata: &Self::DragMetadata) {
        if let Some(meta) = metadata.downcast_ref::<Meta>() {
            self.0.drag_to(point, meta)
        }
    }
    fn release(&self, metadata: Self::DragMetadata) {
        // Here implies the trait bound Meta: Clone
        // How to relax this?
        if let Some(meta) = metadata.downcast_ref::<Meta>() {
            self.0.release(meta.clone())
        }
    }
}

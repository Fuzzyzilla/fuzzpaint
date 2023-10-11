//! Definitions for immutable snapshots of the state of every object at a given instant.
//! Built, maintained, and updated by the document's command queue.
//! Always accessed via reference borrowed from a document queue's [state reader](crate::commands::queue::state_reader).
use crate::FuzzID;
// Namespaces and IDs for them:
// Special namespace types are needed, as BorrowedDocument and BorrowedStrokeLayer are not std::any::Any, a requirement
// for ID generation.

pub struct BlendGraph<'s> {
    // Todoooo
    _p: std::marker::PhantomData<&'s ()>,
}
pub struct StrokeLayer<'s> {
    pub id: super::StrokeLayerID,
    pub strokes: &'s [ImmutableStroke],
}
pub struct Document<'s> {
    /// The path from which the file was loaded or saved, or None if opened as new.
    pub path: Option<&'s std::path::Path>,
    /// Name of the document, inferred from its path or generated.
    pub name: &'s str,
    /// ID that is unique within this execution of the program
    pub id: super::DocumentID,
    // Size, position, dpi, ect todo!
}

#[derive(Clone)]
pub struct ImmutableStroke {
    pub id: FuzzID<Self>,
    pub brush: StrokeBrushSettings,
    /// Points are managed and owned by the (point repository)[crate::repositories::points::PointRepository], not the stroke nor the queue.
    pub point_collection: crate::repositories::points::PointCollectionID,
}

#[derive(Clone)]
/// Per-stroke settings, i.e. ones we expect the user to change frequently without counting it as a "new brush."
pub struct StrokeBrushSettings {
    /// Brushes are managed and owned by an external entity (todo), not the stroke nor the queue.
    pub brush: crate::brush::BrushID,
    /// `a` is flow, NOT opacity, since the stroke is blended continuously not blended as a group.
    pub color_modulate: [f32; 4],
    /// What diameter brush (in docment pixels) should full pen pressure draw with?
    pub size_mul: f32,
    /// If true, the blend constants must be set to generate an erasing effect.
    pub is_eraser: bool,
    /// This should be a property of the brush, not the settings! brushes still todo tho :3
    /// For now, also the minimum size (diameter of brush at pressure near 0)
    pub spacing_px: f32,
}

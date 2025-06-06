//! # State
//!
//! Objects that are owned by the document, representing it's internal state.

pub mod document;
pub mod graph;
pub mod palette;
pub mod rich_text;
pub mod stroke_collection;
pub mod transform;

#[derive(Copy, Clone, PartialEq, Debug)]
/// Per-stroke settings, i.e. ones we expect the user to change frequently without counting it as a "new brush."
pub struct StrokeBrushSettings {
    /// Brushes are managed and owned by an external entity (todo), not the stroke nor the queue.
    pub brush: crate::brush::UniqueID,
    /// `a` is flow, NOT opacity, since the stroke is blended continuously not blended as a group.
    pub color_modulate: crate::color::ColorOrPalette,
    /// What diameter brush (in docment pixels) should full pen pressure draw with?
    pub size_mul: crate::util::FiniteF32,
    /// If true, the blend constants must be set to generate an erasing effect.
    pub is_eraser: bool,
    /// This should be a property of the brush, not the settings! brushes still todo tho :3
    /// For now, also the minimum size (diameter of brush at pressure near 0)
    pub spacing_px: crate::util::FiniteF32,
}

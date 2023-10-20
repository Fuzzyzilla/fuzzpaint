pub mod borrowed;
pub mod graph;

pub type DocumentID = crate::FuzzID<Document>;
pub type StrokeLayerID = crate::FuzzID<StrokeLayer>;

use crate::FuzzID;

#[derive(Clone)]
pub struct StrokeLayer {
    pub id: StrokeLayerID,
    pub strokes: Vec<ImmutableStroke>,
}
impl<'s> From<&'s StrokeLayer> for borrowed::StrokeLayer<'s> {
    fn from(value: &'s StrokeLayer) -> Self {
        Self {
            id: value.id,
            strokes: &value.strokes,
        }
    }
}
pub struct Document {
    /// The path from which the file was loaded or saved, or None if opened as new.
    pub path: Option<std::path::PathBuf>,
    /// Name of the document, inferred from its path or generated.
    pub name: String,
    /// ID that is unique within this execution of the program
    pub id: DocumentID,
    // Size, position, dpi, ect todo!
}
impl Default for Document {
    fn default() -> Self {
        let id = Default::default();
        Self {
            path: None,
            name: format!("{}", id),
            id,
        }
    }
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

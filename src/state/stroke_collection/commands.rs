#[derive(Clone, Debug)]
pub enum StrokeCollectionCommand {
    Stroke(StrokeCommand),
}
#[derive(Clone, Debug)]
pub enum StrokeCommand {
    Created {
        target: super::ImmutableStrokeID,
        brush: crate::state::StrokeBrushSettings,
        points: crate::repositories::points::PointCollectionID,
    },
}

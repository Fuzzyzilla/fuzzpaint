#[derive(Clone, Debug)]
pub enum StrokeCollectionCommand {
    Created(super::StrokeCollectionID),
    Stroke {
        target: super::StrokeCollectionID,
        command: StrokeCommand,
    },
}
impl StrokeCollectionCommand {
    pub(super) fn stroke(&self) -> Option<&StrokeCommand> {
        match self {
            Self::Stroke { command, .. } => Some(command),
            StrokeCollectionCommand::Created(_) => None,
        }
    }
}
#[derive(Clone, Debug)]
pub enum StrokeCommand {
    Created {
        target: super::ImmutableStrokeID,
        brush: crate::state::StrokeBrushSettings,
        points: crate::repositories::points::PointCollectionID,
    },
}

#[derive(Clone, Debug)]
pub enum Command {
    Created(super::StrokeCollectionID),
    Stroke {
        target: super::StrokeCollectionID,
        command: StrokeCommand,
    },
}
impl Command {
    pub(super) fn stroke(&self) -> Option<&StrokeCommand> {
        match self {
            Self::Stroke { command, .. } => Some(command),
            Command::Created(_) => None,
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

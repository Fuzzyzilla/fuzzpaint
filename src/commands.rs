// These commands imply an ownership hierarchy:
// Document owns (is?) the blend graph.
// The blend graph owns layers.
// Layers own strokes.
// Strokes own points.

pub mod queue;

pub enum LayerCommand {
    Created(crate::WeakID<crate::StrokeLayer>),
    Stroke(StrokeCommand),
}
/* Todo: need to figure out how documents play into all this...
pub enum DocumentCommand {
    Created(crate::FuzzID<crate::Document>),
    Layer(LayerCommand),
    // Order or blend of layers changed. Unsure how to represent this. :V
    // BlendGraph,
}
*/
pub enum StrokeCommand {
    Created {
        id: crate::WeakID<crate::Stroke>,
        brush: crate::StrokeBrushSettings,
        points: crate::repositories::points::WeakPointCollectionID,
    },
    ReassignPoints {
        id: crate::WeakID<crate::Stroke>,
        from: crate::repositories::points::WeakPointCollectionID,
        to: crate::repositories::points::WeakPointCollectionID,
    },
}
pub enum ScopeType {
    /// Commands are grouped because they were amalgamated after the user undid many commands
    /// and made an edit. All commands that were trucated from this operation are then pushed as a `Redo` scope.
    Redo,
    /// Commands are grouped because they were individual parts in part of a single, larger operation.
    Atoms,
}
/// Commands about commands!
pub enum MetaCommand {
    /// Bundle many commands into one big group. Can be nested many times.
    /// Grouped commands are treated as a single command, as far as the user can tell.
    PushScope(ScopeType),
    /// Pop the recent scope.
    PopScope,
    /// The document was saved.
    Save,
}

pub enum Command {
    Meta(MetaCommand),
    Layer(LayerCommand),
    // We need a dummy command to serve as the root of the command tree. :V
    // Invalid anywhere else.
    Dummy,
}

pub enum DoUndo<'c> {
    Do(&'c Command),
    Undo(&'c Command),
}

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
pub enum GraphCommand {
    // Waaa
}
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
    /// Commands are grouped because they were individual parts in part of a single, larger operation.
    Atoms,
}
/// Commands about commands!
pub enum MetaCommand {
    /// Bundle many commands into one big group. Can be nested many times.
    /// Grouped commands are treated as a single command, as far as the user can tell.
    // Instead of storing these inline to the tree, store them in a separate slice.
    // Prevents invalid usage (i.e., tree branching in the middle of a scope!)
    Scope(ScopeType, Box<[Command]>),
    /// The document was saved.
    Save,
}

pub enum Command {
    Meta(MetaCommand),
    Layer(LayerCommand),
    Graph(GraphCommand),
    // We need a dummy command to serve as the root of the command tree. :V
    // Invalid anywhere else.
    Dummy,
}

#[derive(PartialEq, Eq)]
pub enum DoUndo<'c, T> {
    Do(&'c T),
    Undo(&'c T),
}

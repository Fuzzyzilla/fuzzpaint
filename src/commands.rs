// These commands imply an ownership hierarchy:
// Document owns (is?) the blend graph.
// The blend graph owns layers.
// Layers own strokes.
// Strokes own points.

pub mod queue;

use crate::state;

pub enum LayerCommand {
    Created(state::StrokeLayerID),
    Stroke(StrokeCommand),
}
pub enum GraphCommand {
    // Todo: IDs are not stable, due to id_tree's inner workings.
    // Thus this is merely a conceptual sketch :P
    BlendChanged {
        from: crate::blend::Blend,
        to: crate::blend::Blend,
        target: state::graph::AnyID,
    },
    Reparent {
        target: state::graph::AnyID,
        /// New parent, or None if root.
        destination: Option<state::graph::NodeID>,
        child_idx: usize,
    },
    LeafCreated {
        id: state::graph::LeafID,
        ty: state::graph::LeafType,
    },
    LeafTyChanged {
        target: state::graph::LeafID,
        old_ty: state::graph::LeafType,
        ty: state::graph::LeafType,
    },
    NodeCreated {
        id: state::graph::NodeID,
        ty: state::graph::NodeType,
    },
    NodeTyChanged {
        target: state::graph::NodeID,
        old_ty: state::graph::NodeType,
        ty: state::graph::NodeType,
    },
    // Waaa
}
pub enum StrokeCommand {
    Created {
        id: crate::FuzzID<crate::Stroke>,
        brush: crate::StrokeBrushSettings,
        points: crate::repositories::points::PointCollectionID,
    },
    ReassignPoints {
        id: crate::FuzzID<crate::Stroke>,
        from: crate::repositories::points::PointCollectionID,
        to: crate::repositories::points::PointCollectionID,
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

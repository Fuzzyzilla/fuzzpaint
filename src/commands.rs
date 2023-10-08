// These commands imply an ownership hierarchy:
// Document owns (is?) the blend graph.
// The blend graph owns layers.
// Layers own strokes.
// Strokes own points.

pub mod queue;

pub enum LayerCommand {
    Created(crate::FuzzID<crate::StrokeLayer>),
    Stroke(StrokeCommand),
}
use crate::graph;
pub enum GraphCommand {
    // Todo: IDs are not stable, due to id_tree's inner workings.
    // Thus this is merely a conceptual sketch :P
    BlendChanged {
        from: crate::blend::Blend,
        to: crate::blend::Blend,
        target: crate::graph::AnyID,
    },
    Reparent {
        target: crate::graph::AnyID,
        /// New parent, or None if root.
        destination: Option<crate::graph::NodeID>,
        child_idx: usize,
    },
    LeafCreated {
        id: crate::graph::LeafID,
        ty: crate::graph::LeafType,
    },
    LeafTyChanged {
        target: crate::graph::LeafID,
        old_ty: crate::graph::LeafType,
        ty: crate::graph::LeafType,
    },
    NodeCreated {
        id: crate::graph::NodeID,
        ty: crate::graph::NodeType,
    },
    NodeTyChanged {
        target: crate::graph::NodeID,
        old_ty: crate::graph::NodeType,
        ty: crate::graph::NodeType,
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

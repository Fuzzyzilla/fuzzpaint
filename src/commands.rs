// These commands imply an ownership hierarchy:
// Document owns (is?) the blend graph.
// The blend graph owns layers.
// Layers own strokes.
// Strokes own points.

pub mod queue;

use crate::state;

#[derive(thiserror::Error, Debug)]
pub enum CommandError {
    #[error("The state this command was constructed for does not match the true state.")]
    MismatchedState,
    #[error("A resource ID referenced by the command is not known.")]
    UnknownResource,
}
pub trait CommandConsumer<C> {
    fn apply(&mut self, command: DoUndo<'_, C>) -> Result<(), CommandError>;
}

#[derive(Clone)]
pub enum LayerCommand {
    Created(state::StrokeLayerID),
    Stroke(StrokeCommand),
}
#[derive(Clone)]
pub enum GraphCommand {
    BlendChanged {
        from: crate::blend::Blend,
        to: crate::blend::Blend,
        target: state::graph::AnyID,
    },
    Reparent {
        target: state::graph::AnyID,
        /// New parent, or None if root.
        new_parent: Option<state::graph::NodeID>,
        new_child_idx: usize,
        /// Old parent, or None if root.
        old_parent: Option<state::graph::NodeID>,
        old_child_idx: usize,
    },
    LeafCreated {
        target: state::graph::LeafID,
        ty: state::graph::LeafType,
        /// New parent, or None if root.
        destination: Option<state::graph::NodeID>,
        child_idx: usize,
    },
    LeafTyChanged {
        target: state::graph::LeafID,
        old_ty: state::graph::LeafType,
        ty: state::graph::LeafType,
    },
    NodeCreated {
        target: state::graph::NodeID,
        ty: state::graph::NodeType,
        /// New parent, or None if root.
        destination: Option<state::graph::NodeID>,
        child_idx: usize,
    },
    NodeTyChanged {
        target: state::graph::NodeID,
        old_ty: state::graph::NodeType,
        ty: state::graph::NodeType,
    },
}
#[derive(Clone)]
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
#[derive(Clone)]
pub enum ScopeType {
    /// Commands are grouped because they were individual parts in part of a single, larger operation.
    Atoms,
}
/// Commands about commands!
#[derive(Clone)]
pub enum MetaCommand {
    /// Bundle many commands into one big group. Can be nested many times.
    /// Grouped commands are treated as a single command, as far as the user can tell.
    // Instead of storing these inline to the tree, store them in a separate slice.
    // Prevents invalid usage (i.e., tree branching in the middle of a scope!)
    Scope(ScopeType, Box<[Command]>),
    /// The document was saved.
    ///
    /// Of course, the concept of "undo"-ing a save does not make sense, but this
    /// event is still very much part of the command tree!
    Save(std::path::PathBuf),
}

#[derive(Clone)]
pub enum Command {
    Meta(MetaCommand),
    Layer(LayerCommand),
    Graph(GraphCommand),
    // We need a dummy command to serve as the root of the command tree. :V
    // Invalid anywhere else.
    Dummy,
}
impl Command {
    pub fn meta(&self) -> Option<&MetaCommand> {
        match self {
            Self::Meta(m) => Some(m),
            _ => None,
        }
    }
    pub fn layer(&self) -> Option<&LayerCommand> {
        match self {
            Self::Layer(m) => Some(m),
            _ => None,
        }
    }
    pub fn graph(&self) -> Option<&GraphCommand> {
        match self {
            Self::Graph(m) => Some(m),
            _ => None,
        }
    }
    pub fn dummy(&self) -> Option<()> {
        match self {
            Self::Dummy => Some(()),
            _ => None,
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum DoUndo<'c, T> {
    Do(&'c T),
    Undo(&'c T),
}
impl<'c, T> DoUndo<'c, T> {
    /// Apply a closure to the inner type T, maintaining the
    /// Do or Undo status. Returns None if the closure returns None.
    pub fn filter_map<Func, Return>(&self, f: Func) -> Option<DoUndo<'c, Return>>
    where
        Func: FnOnce(&'c T) -> Option<&'c Return>,
        Return: 'c,
    {
        match self {
            Self::Do(c) => Some(DoUndo::Do(f(c)?)),
            Self::Undo(c) => Some(DoUndo::Undo(f(c)?)),
        }
    }
}

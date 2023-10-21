// These commands imply an ownership hierarchy:
// Document owns (is?) the blend graph.
// The blend graph owns layers.
// Layers own strokes.
// Strokes own points.

pub mod queue;
pub use state::graph::commands::GraphCommand;
pub use state::stroke_collection::commands::StrokeCollectionCommand;

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
#[derive(Clone, Debug)]
pub enum ScopeType {
    /// Commands are grouped because they were individual parts in part of a single, larger operation.
    Atoms,
    /// A command writer panicked mid write. The commands contained may be part of an incomplete operation,
    /// but are still tracked to ensure integrity of the tree as a whole.
    WritePanic,
}
/// Commands about commands!
#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub enum Command {
    Meta(MetaCommand),
    Graph(GraphCommand),
    StrokeCollection(StrokeCollectionCommand),
    // We need a dummy command to serve as the root of the command tree. :V
    // Invalid anywhere else.
    Dummy,
}
impl From<MetaCommand> for Command {
    fn from(value: MetaCommand) -> Self {
        Self::Meta(value)
    }
}
impl From<GraphCommand> for Command {
    fn from(value: GraphCommand) -> Self {
        Self::Graph(value)
    }
}
impl From<StrokeCollectionCommand> for Command {
    fn from(value: StrokeCollectionCommand) -> Self {
        Self::StrokeCollection(value)
    }
}
impl Command {
    pub fn meta(&self) -> Option<&MetaCommand> {
        match self {
            Self::Meta(m) => Some(m),
            _ => None,
        }
    }
    pub fn stroke_collection(&self) -> Option<&StrokeCollectionCommand> {
        match self {
            Self::StrokeCollection(m) => Some(m),
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

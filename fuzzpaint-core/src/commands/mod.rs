//! # Commands
//!
//! Commands are the way the shared state of the document are modified. Every (nontrivial, like renaming a layer) change
//! is recorded automatically as a command by a [`queue::writer`].

pub use state::graph::commands::Command as GraphCommand;
pub use state::palette::commands::Command as PaletteCommand;
pub use state::stroke_collection::commands::Command as StrokeCollectionCommand;

use crate::state;

#[derive(thiserror::Error, Debug)]
pub enum CommandError {
    #[error("command constructed for a state that does not match the current state")]
    MismatchedState,
    #[error("resource referenced by the command is not found")]
    UnknownResource,
    #[error("command makes no changes")]
    NoOp,
}
pub trait CommandConsumer<C> {
    /// Apply a single command. If this generates an error,
    /// the state of `self` should *not* be observably changed.
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
    Palette(PaletteCommand),
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
impl From<PaletteCommand> for Command {
    fn from(value: PaletteCommand) -> Self {
        Self::Palette(value)
    }
}
impl From<StrokeCollectionCommand> for Command {
    fn from(value: StrokeCollectionCommand) -> Self {
        Self::StrokeCollection(value)
    }
}
impl Command {
    #[must_use]
    pub fn meta(&self) -> Option<&MetaCommand> {
        match self {
            Self::Meta(m) => Some(m),
            _ => None,
        }
    }
    #[must_use]
    pub fn stroke_collection(&self) -> Option<&StrokeCollectionCommand> {
        match self {
            Self::StrokeCollection(m) => Some(m),
            _ => None,
        }
    }
    #[must_use]
    pub fn graph(&self) -> Option<&GraphCommand> {
        match self {
            Self::Graph(m) => Some(m),
            _ => None,
        }
    }
    #[must_use]
    pub fn palette(&self) -> Option<&PaletteCommand> {
        match self {
            Self::Palette(m) => Some(m),
            _ => None,
        }
    }
    #[must_use]
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

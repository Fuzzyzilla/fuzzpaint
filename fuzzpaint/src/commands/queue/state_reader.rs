//! Views into the document state represented by a command queue.

use super::super::*;

use crate::state;
pub trait CommandQueueStateReader {
    fn graph(&self) -> &state::graph::BlendGraph;
    fn stroke_collections(&self) -> &state::stroke_collection::StrokeCollectionState;
    /*fn stroke_layers(&self) -> &[state::StrokeLayer];
    fn stroke_layer(&self, id: state::StrokeLayerID) -> Option<&state::StrokeLayer> {
        self.stroke_layers().iter().find(|layer| layer.id == id)
    }*/
    fn changes(&'_ self) -> impl Iterator<Item = DoUndo<'_, Command>> + '_;
    fn has_changes(&self) -> bool;
}

/// Represents a lock on the global queue state.
/// Guarunteed to be the most up-to-date view, as the queue can not be modified
/// while this is held. Should only be used for short operations, so that other threads
/// trying to write commands do not get starved.
///
/// It is an error for the task that owns this lock to attempt to mutatably access the queue! Doing so
/// will result in a deadlock.
pub struct CommandQueueReadLock {}
pub(super) enum OwnedDoUndo<C> {
    Do(C),
    Undo(C),
}
impl<C> From<DoUndo<'_, C>> for OwnedDoUndo<C>
where
    C: Clone,
{
    fn from(value: DoUndo<'_, C>) -> Self {
        match value {
            DoUndo::Do(c) => Self::Do(c.clone()),
            DoUndo::Undo(c) => Self::Undo(c.clone()),
        }
    }
}
/// Represents a weak lock on the global queue state, that will create a copy of the state if the queue is modified.
/// As such, it may drift out-of-date and may incur additional allocations. Use for long operations, to prevent
/// blocking other threads from pushing to the queue!
pub struct CommandQueueCloneLock {
    /// Eager clone of the commands. Since commands are immutable, it *should*
    /// be possible to architect this to not need to clone. But for now since queue's
    /// commands are stored in a Vec, their addresses are not 'static :/
    pub(super) commands: Vec<OwnedDoUndo<Command>>,
    pub(super) shared_state: std::sync::Arc<super::queue_state::State>,
    pub(super) inner: std::sync::Weak<parking_lot::RwLock<super::DocumentCommandQueueInner>>,
}
pub enum Stale {
    /// The lock state matches the true state.
    UpToDate,
    /// The locked state has commands not present in the true state.
    /// i.e., the path from this state to true state contains at least one `Undo`
    ///
    /// Takes precedence over `Behind`.
    Ahead,
    /// The locked state is missing commands that are present in the true state.
    Behind,
    /// The true state has been dropped. The clone lock retains access to the resources, however
    /// this hint means it's unlikely the consumer's work will be useful.
    Dropped,
}
impl CommandQueueCloneLock {
    /// Query whether the lock still represents the present state of the document, otherwise giving a
    /// description of its temporal relationship to the true state.
    /// Can be used as a hint to cancel a long operation, if it's found to be invalidated by new changes.
    ///
    /// Immutably locks the queue during the query.
    #[must_use]
    pub fn stale(&self) -> Stale {
        let Some(inner) = self.inner.upgrade() else {
            return Stale::Dropped;
        };
        let read = inner.read();
        // Traverse to find the relationship between self state and true state
        let mut traverse = super::traverse(
            &read.command_tree,
            self.shared_state.present,
            read.state.present,
        )
        .unwrap();
        // First step is all we need to find the relationship!
        let first_step = traverse.next();

        match first_step {
            None => Stale::UpToDate,
            Some(DoUndo::Undo(..)) => Stale::Ahead,
            Some(DoUndo::Do(..)) => Stale::Behind,
        }
    }
}

impl CommandQueueStateReader for CommandQueueCloneLock {
    fn changes(&'_ self) -> impl Iterator<Item = DoUndo<'_, Command>> + '_ {
        self.commands.iter().map(|owned| match owned {
            OwnedDoUndo::Do(c) => DoUndo::Do(c),
            OwnedDoUndo::Undo(c) => DoUndo::Undo(c),
        })
    }
    fn graph(&self) -> &state::graph::BlendGraph {
        &self.shared_state.graph
    }
    fn stroke_collections(&self) -> &state::stroke_collection::StrokeCollectionState {
        &self.shared_state.stroke_state
    }
    fn has_changes(&self) -> bool {
        !self.commands.is_empty()
    }
}

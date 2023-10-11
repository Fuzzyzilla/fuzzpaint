//! Views into the document state represented by a command queue.

/// Represents a lock on the global queue state.
/// Guarunteed to be the most up-to-date view, as the queue can not be modified
/// while this is held. Should only be used for short operations, so that other threads
/// trying to write commands do not get starved.
///
/// It is an error for the task that owns this lock to attempt to push to the queue! Doing so
/// will result in a deadlock.
pub struct CommandQueueLock {}
enum OwnedDoUndo<C> {
    Do(C),
    Undo(C),
}
impl<C> From<super::super::DoUndo<'_, C>> for OwnedDoUndo<C>
where
    C: Clone,
{
    fn from(value: super::super::DoUndo<'_, C>) -> Self {
        match value {
            super::super::DoUndo::Do(c) => Self::Do(c.clone()),
            super::super::DoUndo::Undo(c) => Self::Undo(c.clone()),
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
    commands: Vec<OwnedDoUndo<super::super::Command>>,
}
pub enum Stale {
    /// The locked state has commands not present in the true state.
    /// i.e., the path from this state to true state contains at least one `Undo`
    ///
    /// Takes precedence over `Behind`.
    Ahead,
    /// The locked state is missing commands that are present in the true state.
    Behind,
}
impl CommandQueueCloneLock {
    /// Query whether the lock still represents the present state of the document.
    /// None means it is up to date, or Stale provides a description of the relative position.
    /// Can be used as a hint to cancel a long operation, if it's found to be invalidated by new changes.
    pub fn stale(&self) -> Option<Stale> {
        todo!();
    }
}
use crate::state;
pub trait CommandQueueStateReader {
    fn graph(&self) -> &state::borrowed::BlendGraph;
    fn stroke_layers(&self) -> &[state::borrowed::StrokeLayer];
    fn stroke_layer(&self, id: state::StrokeLayerID) -> Option<&state::borrowed::StrokeLayer>;
    type ChangesIter<'s>: Iterator<Item = super::super::DoUndo<'s, super::super::Command>> + 's
    where
        Self: 's;
    fn changes(&self) -> Self::ChangesIter<'_>;
}

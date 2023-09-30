//! Command Queue
//!
//! The global queues manage all the actions performed by the user, keeping track of commands, undo/redo state, etc.
//! The queues are the ground truth for the current state of the program and their corresponding document. Listeners to the queue
//! can be various stages of out-of-date, at any point they can view all new commands and bring themselves back to the present.
//!
//! There exists one command queue per document.

pub struct CommandAtomsWriter {}
pub struct DocumentCommandQueue {
    commands: Vec<super::Command>,
    document: crate::FuzzID<crate::Document>,
}
impl DocumentCommandQueue {
    /// Atomically push write some number of commands in an Atoms scope, such that they are treated as one larger command.
    pub fn write_atoms(&self, f: impl FnOnce(&mut CommandAtomsWriter)) {}
}
pub struct DocumentCommandListener {
    document: crate::WeakID<crate::Document>,
}
impl DocumentCommandListener {}

//! # Queue state
//!
//! Maintains the structure of the document as a whole. These are the structures commands act upon, writen to via
//! `DocumentCommandQueue::write_with` and read via `CommandQueueStateReader` implementors.
use crate::commands::*;
use crate::state;

pub struct State {
    //pub stroke_layers: Vec<state::StrokeLayer>,
    //pub document: state::Document,
    pub graph: state::graph::BlendGraph,
    pub stroke_state: state::stroke_collection::StrokeCollectionState,
    /// The node in the command tree that this state corresponds to
    pub present: slab_tree::NodeId,
}
impl State {
    pub fn new(root: slab_tree::NodeId) -> Self {
        Self {
            graph: Default::default(),
            stroke_state: Default::default(),
            present: root,
        }
    }
    /// Make a copy of self, keeping all IDs stable.
    /// This is a very expensive operation! Try to avoid where possible :3
    pub fn fork(&self) -> Self {
        // For now, this is just clone.
        Self {
            graph: self.graph.clone(),
            stroke_state: self.stroke_state.clone(),
            present: self.present.clone(),
        }
    }
}
impl CommandConsumer<Command> for State {
    fn apply(&mut self, action: DoUndo<Command>) -> Result<(), CommandError> {
        match action {
            DoUndo::Do(Command::Graph(..)) | DoUndo::Undo(Command::Graph(..)) => {
                // Unwrap ok - guarded by match arm.
                self.graph.apply(action.filter_map(Command::graph).unwrap())
            }
            DoUndo::Do(Command::StrokeCollection(..))
            | DoUndo::Undo(Command::StrokeCollection(..)) => {
                // Unwrap ok - guarded by match arm.
                self.stroke_state
                    .apply(action.filter_map(Command::stroke_collection).unwrap())
            }
            // Recursively do each of the commands in the scope, in order.
            DoUndo::Do(Command::Meta(MetaCommand::Scope(_, commands))) => commands
                .iter()
                .try_for_each(|command| self.apply(DoUndo::Do(command))),
            // Recursively undo each of the commands of the scope, in reverse order.
            DoUndo::Undo(Command::Meta(MetaCommand::Scope(_, commands))) => commands
                .iter()
                .rev()
                .try_for_each(|command| self.apply(DoUndo::Undo(command))),
            DoUndo::Do(Command::Dummy) | DoUndo::Undo(Command::Dummy) => Ok(()),
            DoUndo::Do(Command::Meta(MetaCommand::Save(..)))
            | DoUndo::Undo(Command::Meta(MetaCommand::Save(..))) => unimplemented!(),
        }
    }
}

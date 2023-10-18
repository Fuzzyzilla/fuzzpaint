use crate::commands::*;
use crate::state::{self, borrowed};

pub struct State {
    //pub stroke_layers: Vec<state::StrokeLayer>,
    //pub document: state::Document,
    pub graph: state::graph::BlendGraph,
    /// The node in the command tree that this state corresponds to
    pub present: slab_tree::NodeId,
}
impl State {
    pub fn new(root: slab_tree::NodeId) -> Self {
        Self {
            //stroke_layers: Default::default(),
            //document: Default::default(),
            graph: Default::default(),
            present: root,
        }
    }
    /// Make a copy of self, keeping all IDs stable.
    /// This is a very expensive operation! Try to avoid where possible :3
    pub fn fork(&self) -> Self {
        // For now, this is just clone.
        Self {
            //stroke_layers: self.stroke_layers.clone(),
            /*document: state::Document {
                id: self.document.id,
                path: self.document.path.clone(),
                name: self.document.name.clone(),
            },*/
            graph: self.graph.clone(),
            present: self.present.clone(),
        }
    }
}
impl CommandConsumer<Command> for State {
    fn apply(&mut self, action: DoUndo<Command>) -> Result<(), CommandError> {
        match action {
            DoUndo::Do(Command::Graph(..)) | DoUndo::Undo(Command::Graph(..)) => {
                self.graph.apply(action.filter_map(Command::graph).unwrap())
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
            _ => unimplemented!(),
        }
    }
}
/*
// BONK no premature optimization!
struct BorrowedState<'s> {
    stroke_layers: &'s [state::StrokeLayer],
    document: borrowed::Document<'s>,
    graph: &'s state::graph::BlendGraph,
    /// The node in the command tree that this state corresponds to
    present: &'s slab_tree::NodeId,
}
/// State where some fields have been overwritten, but the rest are inherited from an older state.
/// Can be flattened back into a State without incurring extra clones when the Arcs are only owned by this object!
struct PartialState {
    base: either::Either<std::sync::Arc<State>, std::sync::Arc<PartialState>>,
}*/

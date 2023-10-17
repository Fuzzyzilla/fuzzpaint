use super::commands::GraphCommand;
use crate::commands::queue::writer::CommandWrite;

#[derive(thiserror::Error)]
pub enum CommandError<Err: std::error::Error> {
    #[error("{}", .0)]
    Inner(Err),
    /// The command cannot be applied to the current state.
    #[error("The command cannot be applied to the current state")]
    MismatchedState,
}
impl<Err: std::error::Error> From<Err> for CommandError<Err> {
    fn from(value: Err) -> Self {
        Self::Inner(value)
    }
}

type Result<T, Err: std::error::Error> = std::result::Result<T, CommandError<Err>>;

pub struct GraphWriter<'a, Write: CommandWrite<GraphCommand>> {
    writer: Write,
    graph: &'a mut super::BlendGraph,
}
impl<'a, Write: CommandWrite<GraphCommand>> std::ops::Deref for GraphWriter<'a, Write> {
    type Target = super::BlendGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}
impl<'a, Write: CommandWrite<GraphCommand>> GraphWriter<'a, Write> {
    pub fn new(writer: Write, graph: &'a mut super::BlendGraph) -> Self {
        Self { writer, graph }
    }
    pub fn change_blend(
        &mut self,
        id: super::AnyID,
        to: crate::blend::Blend,
    ) -> Result<(), super::TargetError> {
        todo!()
    }
    pub fn reparent(
        &mut self,
        id: super::AnyID,
        location: super::Location<'_>,
    ) -> Result<(), super::ReparentError> {
        todo!()
    }
    pub fn add_leaf(
        &mut self,
        leaf_ty: super::LeafType,
        location: super::Location<'_>,
    ) -> Result<super::LeafID, super::TargetError> {
        todo!()
    }
    pub fn set_leaf(
        &mut self,
        leaf: super::LeafID,
        leaf_ty: super::LeafType,
    ) -> Result<(), super::TargetError> {
        todo!()
    }
    pub fn add_node(
        &mut self,
        node_ty: super::NodeType,
        location: super::Location<'_>,
    ) -> Result<super::NodeID, super::TargetError> {
        todo!()
    }
    pub fn set_node(
        &mut self,
        node: super::NodeID,
        node_ty: super::NodeType,
    ) -> Result<(), super::TargetError> {
        todo!()
    }
}

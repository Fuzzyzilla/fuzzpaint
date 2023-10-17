use super::commands::GraphCommand;
use crate::commands::queue::writer::CommandWrite;

pub struct GraphWriter<'a, Write: CommandWrite<GraphCommand>> {
    writer: Write,
    graph: &'a mut super::BlendGraph,
}
impl<'a, Write: CommandWrite<GraphCommand>> GraphWriter<'a, Write> {
    pub fn new(writer: Write, graph: &'a mut super::BlendGraph) -> Self {
        Self { writer, graph }
    }
    pub fn uwu(&mut self) {
        todo!()
    }
}

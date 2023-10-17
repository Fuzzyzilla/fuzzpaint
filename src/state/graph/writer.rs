use super::commands::GraphCommand;
use crate::commands::queue::writer::CommandWrite;

pub struct GraphWriter<Write: CommandWrite<GraphCommand>> {
    writer: Write,
}
impl<Write: CommandWrite<GraphCommand>> GraphWriter<Write> {
    pub fn uwu(&mut self) {
        todo!()
    }
}

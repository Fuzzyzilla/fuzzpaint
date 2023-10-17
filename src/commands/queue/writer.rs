pub struct CommandQueueWriter<'a> {
    pub(super) lock: parking_lot::RwLockWriteGuard<'a, super::DocumentCommandQueueInner>,
    // Optimize for exactly one command (the most common case)
    pub(super) commands: smallvec::SmallVec<[crate::commands::Command; 1]>,
}
impl Drop for CommandQueueWriter<'_> {
    fn drop(&mut self) {
        use crate::commands;

        // We always write exactly one command - bundle into one if more!
        // If panic exit, write as a panic scope (even if the scope is just one command long)
        let command = if std::thread::panicking() {
            commands::Command::Meta(commands::MetaCommand::Scope(
                commands::ScopeType::WritePanic,
                std::mem::take(&mut self.commands).into_boxed_slice(),
            ))
        } else {
            // Not panicking. Write the single command, or write as Atoms scope if multiple.
            if self.commands.len() == 1 {
                self.commands.pop().unwrap()
            } else {
                commands::Command::Meta(commands::MetaCommand::Scope(
                    commands::ScopeType::Atoms,
                    std::mem::take(&mut self.commands).into_boxed_slice(),
                ))
            }
        };

        // Weird borrow issue :P
        let present = self.lock.state.present;

        // Write the command or scope (as last child, as that corresponds to "latest change")
        // and update cursor.
        let new = self
            .lock
            .command_tree
            .get_mut(present)
            // It's a logic error for "present" node to not exist. Not much error handling we could do here!
            // Neglecting to write the command is just as bad, as then the State and command queue would be mismatched.
            .expect("Present node not found in the command tree.")
            .append(command)
            .node_id();
        self.lock.state.present = new;
    }
}
impl<'q> CommandQueueWriter<'q> {
    pub fn queue<'s>(&'s mut self) -> GraphWriter<'s, 'q>
    where
        'q: 's,
    {
        GraphWriter { writer: self }
    }
}

pub struct GraphWriter<'a, 'q> {
    writer: &'a mut CommandQueueWriter<'q>,
}
impl GraphWriter<'_, '_> {
    pub fn uwu(&mut self) {
        todo!()
    }
}

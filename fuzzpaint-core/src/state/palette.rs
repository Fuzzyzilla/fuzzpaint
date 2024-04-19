use crate::{
    color::{Color, PaletteIndex},
    commands::{CommandConsumer, CommandError, DoUndo},
};

pub mod commands {
    use super::{Color, PaletteIndex};
    #[derive(Clone, Debug)]
    pub enum Command {
        Added {
            target: PaletteIndex,
            initial_color: Color,
        },
        Changed {
            target: PaletteIndex,
            from: Color,
            to: Color,
        },
    }
}
pub mod writer {
    use super::{commands::Command, Color, PaletteIndex};
    use crate::queue::writer::CommandWrite;
    pub struct Writer<'a, Write> {
        writer: Write,
        state: &'a mut super::Palette,
    }
    impl<Write> std::ops::Deref for Writer<'_, Write> {
        type Target = super::Palette;
        fn deref(&self) -> &Self::Target {
            self.state
        }
    }

    impl<'a, Write: CommandWrite<Command>> Writer<'a, Write> {
        pub fn new(writer: Write, state: &'a mut super::Palette) -> Self {
            Self { writer, state }
        }
        pub fn insert(&mut self, color: Color) -> PaletteIndex {
            let new_idx = self.state.push(color);
            self.writer.write(Command::Added {
                target: new_idx,
                initial_color: color,
            });
            new_idx
        }
        pub fn set(&mut self, index: PaletteIndex, to: Color) -> Result<(), ()> {
            let (exists, color) = self.state.get_mut(index).ok_or(())?;
            if !*exists {
                return Err(());
            }

            let from = *color;
            *color = to;

            self.writer.write(Command::Changed {
                target: index,
                from,
                to,
            });

            Ok(())
        }
    }
}

#[derive(Default, Clone)]
pub struct Palette {
    // Index -> (exists, color)
    // Color has nore niches to spare to store that exists bit. Implementation detail, oh well.
    colors: Vec<(bool, Color)>,
}
impl Palette {
    /// Add a new color, returning it's index.
    #[must_use]
    pub fn push(&mut self, color: Color) -> PaletteIndex {
        self.colors.push((true, color));

        PaletteIndex(u64::try_from(self.colors.len()).unwrap())
    }
    /// Mutate a color from it's index.
    pub fn get_mut(&mut self, idx: PaletteIndex) -> Option<&mut (bool, Color)> {
        let idx = usize::try_from(idx.0).ok()?;

        self.colors.get_mut(idx)
    }
    /// Get a color from it's index.
    #[must_use]
    pub fn get(&self, idx: PaletteIndex) -> Option<Color> {
        let idx = usize::try_from(idx.0).ok()?;

        self.colors.get(idx).map(|&(_, color)| color)
    }
    pub fn iter(&self) -> impl Iterator<Item = &Color> {
        self.colors
            .iter()
            .filter_map(|(exists, color)| exists.then_some(color))
    }
}

impl CommandConsumer<commands::Command> for Palette {
    fn apply(&mut self, command: DoUndo<'_, commands::Command>) -> Result<(), CommandError> {
        use commands::Command;
        match command {
            DoUndo::Do(Command::Added {
                target,
                initial_color,
            }) => match self.get_mut(*target) {
                Some((exists, color)) => {
                    // Already exists or the initial color is wrong.
                    if *exists || color != initial_color {
                        Err(CommandError::MismatchedState)
                    } else {
                        *exists = true;
                        Ok(())
                    }
                }
                None => Err(CommandError::UnknownResource),
            },
            DoUndo::Undo(Command::Added {
                target,
                initial_color,
            }) => match self.get_mut(*target) {
                Some((exists, color)) => {
                    // Already removed or the initial color is wrong.
                    if !*exists || color != initial_color {
                        Err(CommandError::MismatchedState)
                    } else {
                        *exists = false;
                        Ok(())
                    }
                }
                None => Err(CommandError::UnknownResource),
            },
            DoUndo::Do(Command::Changed { target, from, to })
            | DoUndo::Undo(Command::Changed {
                target,
                from: to,
                to: from,
            }) => match self.get_mut(*target) {
                Some((exists, color)) => {
                    if !*exists || color != from {
                        Err(CommandError::MismatchedState)
                    } else {
                        *color = *to;
                        Ok(())
                    }
                }
                None => Err(CommandError::UnknownResource),
            },
        }
    }
}

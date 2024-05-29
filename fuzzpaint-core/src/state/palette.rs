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

        PaletteIndex(u64::try_from(self.colors.len() - 1).unwrap())
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
    pub fn iter(&self) -> impl Iterator<Item = (PaletteIndex, &Color)> {
        self.colors
            .iter()
            .enumerate()
            .filter_map(|(idx, (exists, color))| {
                let idx = PaletteIndex(u64::try_from(idx).ok()?);
                exists.then_some((idx, color))
            })
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

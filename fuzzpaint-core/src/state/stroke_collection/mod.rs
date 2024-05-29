//! # Strokes
//!
//! States which hold many strokes and their settings, as well as their deletion state.

pub mod commands;

pub type StrokeCollectionID = crate::FuzzID<StrokeCollection>;
pub type ImmutableStrokeID = crate::FuzzID<ImmutableStroke>;

#[derive(Copy, Clone)]
pub struct ImmutableStroke {
    pub id: ImmutableStrokeID,
    pub brush: crate::state::StrokeBrushSettings,
    /// Points are managed and owned by the (point repository)[crate::repositories::points::PointRepository], not the stroke nor the queue.
    pub point_collection: crate::repositories::points::PointCollectionID,
}

#[derive(thiserror::Error, Debug)]
pub enum ImmutableStrokeError {
    #[error("source stroke contains too many points to upload")]
    TooLarge,
}

#[derive(Clone)]
pub struct StrokeCollection {
    pub strokes: Vec<ImmutableStroke>,
    /// Flags to determine which strokes have are active/not "Undone"
    pub strokes_active: bitvec::vec::BitVec,
    /// Is the collection as a whole undone?
    pub active: bool,
}
impl Default for StrokeCollection {
    fn default() -> Self {
        Self {
            strokes: Vec::new(),
            strokes_active: bitvec::vec::BitVec::new(),
            active: true,
        }
    }
}
// Public methods for client
impl StrokeCollection {
    pub fn iter_active(&self) -> impl Iterator<Item = &ImmutableStroke> + '_ {
        // Could also achieve with a zip. really depends on how dense we expect
        // deleted strokes to be, I should bench!
        self.strokes_active
            .iter_ones()
            // Short circuit iteration if we reach out-of-bounds (that'd be weird)
            .map_while(|index| self.strokes.get(index))
    }
    // O(n).. I should do better :3
    // Can't binary search over IDs, as they're not technically
    // required to be ordered, in preparation for network shenanigans.
    /// Get a stroke by the given ID. Returns None if it is not found, or has been deleted.
    #[must_use]
    pub fn get(&self, id: ImmutableStrokeID) -> Option<&ImmutableStroke> {
        let (idx, stroke) = self
            .strokes
            .iter()
            .enumerate()
            .find(|(_, stroke)| stroke.id == id)?;

        // Return the stroke, if it's not deleted.
        self.strokes_active.get(idx)?.then_some(stroke)
    }
}
// Private methods for writer/applier
impl StrokeCollection {
    /// Insert a new stroke at the end, defaulting to active.
    fn push_back(&mut self, stroke: ImmutableStroke) {
        self.strokes.push(stroke);
        // Initially active.
        self.strokes_active.push(true);
    }
    /// Gets a mutable reference to a stroke, and it's activity status.
    #[must_use]
    fn get_mut(
        &mut self,
        id: ImmutableStrokeID,
    ) -> Option<(
        &mut ImmutableStroke,
        impl std::ops::DerefMut<Target = bool> + '_,
    )> {
        let (idx, stroke) = self
            .strokes
            .iter_mut()
            .enumerate()
            .find(|(_, stroke)| stroke.id == id)?;

        let active = self.strokes_active.get_mut(idx)?;

        Some((stroke, active))
    }
}
/// Collection of collections, by ID.
#[derive(Clone, Default)]
pub struct StrokeCollectionState(pub hashbrown::HashMap<StrokeCollectionID, StrokeCollection>);
// Public methods for access by the client
impl StrokeCollectionState {
    #[must_use]
    pub fn get(&self, id: StrokeCollectionID) -> Option<&StrokeCollection> {
        let collection = self.0.get(&id)?;

        // Return, only if active.
        collection.active.then_some(collection)
    }
}
// Private methods for modification by the writer/command applier
impl StrokeCollectionState {
    #[must_use]
    fn get_mut(&mut self, id: StrokeCollectionID) -> Option<&mut StrokeCollection> {
        self.0.get_mut(&id)
    }
    fn insert(&mut self) -> StrokeCollectionID {
        let id = crate::FuzzID::default();
        self.0.insert(id, StrokeCollection::default());
        id
    }
}

impl CommandConsumer<commands::StrokeCommand> for StrokeCollection {
    fn apply(&mut self, command: DoUndo<'_, commands::StrokeCommand>) -> Result<(), CommandError> {
        match command {
            DoUndo::Do(commands::StrokeCommand::Created {
                target,
                brush,
                points,
            }) => {
                const NEW_ACTIVE: bool = true;
                let (stroke, mut active) =
                    self.get_mut(*target).ok_or(CommandError::UnknownResource)?;

                // Was already set! Or, state doesn't match.
                if *active == NEW_ACTIVE
                    || stroke.point_collection != *points
                    || &stroke.brush != brush
                {
                    Err(CommandError::MismatchedState)
                } else {
                    *active = NEW_ACTIVE;
                    Ok(())
                }
            }
            DoUndo::Undo(commands::StrokeCommand::Created {
                target,
                brush,
                points,
            }) => {
                const NEW_ACTIVE: bool = false;
                let (stroke, mut active) =
                    self.get_mut(*target).ok_or(CommandError::UnknownResource)?;

                // Was already set! Or, state doesn't match.
                if *active == NEW_ACTIVE
                    || stroke.point_collection != *points
                    || &stroke.brush != brush
                {
                    Err(CommandError::MismatchedState)
                } else {
                    *active = NEW_ACTIVE;
                    Ok(())
                }
            }
        }
    }
}

use crate::commands::{CommandConsumer, CommandError, DoUndo};
impl CommandConsumer<commands::Command> for StrokeCollectionState {
    fn apply(&mut self, command: DoUndo<'_, commands::Command>) -> Result<(), CommandError> {
        match command {
            DoUndo::Do(commands::Command::Created(id)) => {
                let collection = self.get_mut(*id).ok_or(CommandError::UnknownResource)?;
                if collection.active {
                    Err(CommandError::MismatchedState)
                } else {
                    collection.active = true;

                    Ok(())
                }
            }
            DoUndo::Undo(commands::Command::Created(id)) => {
                let collection = self.get_mut(*id).ok_or(CommandError::UnknownResource)?;
                if collection.active {
                    collection.active = false;

                    Ok(())
                } else {
                    Err(CommandError::MismatchedState)
                }
            }
            DoUndo::Do(commands::Command::Stroke { target, .. })
            | DoUndo::Undo(commands::Command::Stroke { target, .. }) => {
                let collection = self.get_mut(*target).ok_or(CommandError::UnknownResource)?;
                // Unwrap OK - we checked via the match arm.
                collection.apply(command.filter_map(|command| command.stroke()).unwrap())
            }
        }
    }
}

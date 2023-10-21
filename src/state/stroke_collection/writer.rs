use super::*;
use crate::commands::queue::writer::*;

pub struct StrokeCollectionWriter<'s, W: CommandWrite<commands::StrokeCollectionCommand>> {
    id: StrokeCollectionID,
    collection: &'s mut StrokeCollection,
    writer: W,
}
impl<'s, W: CommandWrite<commands::StrokeCollectionCommand>> std::ops::Deref
    for StrokeCollectionWriter<'s, W>
{
    type Target = StrokeCollection;
    fn deref(&self) -> &Self::Target {
        &*self.collection
    }
}
impl<'s, W: CommandWrite<commands::StrokeCollectionCommand>> StrokeCollectionWriter<'s, W> {
    pub fn id(&self) -> StrokeCollectionID {
        self.id
    }
    pub fn push_back(
        &mut self,
        brush: crate::state::StrokeBrushSettings,
        points: crate::repositories::points::PointCollectionID,
    ) -> ImmutableStrokeID {
        let id = Default::default();
        let stroke = ImmutableStroke {
            brush: brush.clone(),
            id,
            point_collection: points,
        };
        self.writer
            .write(commands::StrokeCollectionCommand::Stroke {
                target: self.id,
                command: commands::StrokeCommand::Created {
                    target: id,
                    brush,
                    points,
                },
            });
        self.collection.push_back(stroke);

        id
    }
}

pub struct StrokeCollectionStateWriter<'s, W: CommandWrite<commands::StrokeCollectionCommand>> {
    id: StrokeCollectionID,
    state: &'s mut StrokeCollectionState,
    writer: W,
}
impl<'s, W: CommandWrite<commands::StrokeCollectionCommand>> std::ops::Deref
    for StrokeCollectionStateWriter<'s, W>
{
    type Target = StrokeCollectionState;
    fn deref(&self) -> &Self::Target {
        &*self.state
    }
}
impl<'s, W: CommandWrite<commands::StrokeCollectionCommand>> StrokeCollectionStateWriter<'s, W> {
    pub fn get_mut<'this>(
        &'this mut self,
        id: StrokeCollectionID,
    ) -> Option<StrokeCollectionWriter<'this, &mut W>>
    where
        's: 'this,
    {
        let collection = self.state.get_mut(id)?;

        Some(StrokeCollectionWriter {
            collection,
            id,
            // Results in a double borrow. Oh well!
            writer: &mut self.writer,
        })
    }
    pub fn insert(&mut self) -> StrokeCollectionID {
        let id = self.state.insert();
        self.writer
            .write(commands::StrokeCollectionCommand::Created(id));
        id
    }
}

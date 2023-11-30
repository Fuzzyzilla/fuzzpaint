use super::{
    commands, ImmutableStroke, ImmutableStrokeID, StrokeCollection, StrokeCollectionID,
    StrokeCollectionState,
};
use crate::commands::queue::writer::CommandWrite;

pub struct StrokeCollectionWriter<'s, Writer: CommandWrite<commands::StrokeCollectionCommand>> {
    id: StrokeCollectionID,
    collection: &'s mut StrokeCollection,
    writer: Writer,
}
impl<'s, Writer: CommandWrite<commands::StrokeCollectionCommand>> std::ops::Deref
    for StrokeCollectionWriter<'s, Writer>
{
    type Target = StrokeCollection;
    fn deref(&self) -> &Self::Target {
        &*self.collection
    }
}
impl<'s, Writer: CommandWrite<commands::StrokeCollectionCommand>>
    StrokeCollectionWriter<'s, Writer>
{
    pub fn id(&self) -> StrokeCollectionID {
        self.id
    }
    pub fn push_back(
        &mut self,
        brush: crate::state::StrokeBrushSettings,
        points: crate::repositories::points::PointCollectionID,
    ) -> ImmutableStrokeID {
        let id = ImmutableStrokeID::default();
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

pub struct StrokeCollectionStateWriter<'s, Writer: CommandWrite<commands::StrokeCollectionCommand>>
{
    state: &'s mut StrokeCollectionState,
    writer: Writer,
}
impl<'s, Writer: CommandWrite<commands::StrokeCollectionCommand>> std::ops::Deref
    for StrokeCollectionStateWriter<'s, Writer>
{
    type Target = StrokeCollectionState;
    fn deref(&self) -> &Self::Target {
        &*self.state
    }
}
impl<'s, Write: CommandWrite<commands::StrokeCollectionCommand>>
    StrokeCollectionStateWriter<'s, Write>
{
    pub fn new(writer: Write, state: &'s mut StrokeCollectionState) -> Self {
        Self { state, writer }
    }
    pub fn get_mut(
        &mut self,
        id: StrokeCollectionID,
    ) -> Option<StrokeCollectionWriter<'_, &mut Write>> {
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

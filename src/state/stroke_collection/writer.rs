use super::*;
use crate::commands::queue::writer::*;

pub struct StrokeCollectionWriter<'s, W: CommandWrite<commands::StrokeCollectionCommand>> {
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
        self.writer.write(commands::StrokeCollectionCommand::Stroke(
            commands::StrokeCommand::Created {
                target: id,
                brush,
                points,
            },
        ));
        self.collection.push_back(stroke);

        id
    }
}

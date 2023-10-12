//! Definitions for immutable snapshots of the state of every object at a given instant.
//! Built, maintained, and updated by the document's command queue.
//! Always accessed via reference borrowed from a document queue's [state reader](crate::commands::queue::state_reader).
use crate::FuzzID;
// Namespaces and IDs for them:
// Special namespace types are needed, as BorrowedDocument and BorrowedStrokeLayer are not std::any::Any, a requirement
// for ID generation.

pub struct BlendGraph<'s> {
    // Todoooo
    _p: std::marker::PhantomData<&'s ()>,
}
pub struct StrokeLayer<'s> {
    pub id: super::StrokeLayerID,
    pub strokes: &'s [super::ImmutableStroke],
}
pub struct Document<'s> {
    /// The path from which the file was loaded or saved, or None if opened as new.
    pub path: Option<&'s std::path::Path>,
    /// Name of the document, inferred from its path or generated.
    pub name: &'s str,
    /// ID that is unique within this execution of the program
    pub id: super::DocumentID,
    // Size, position, dpi, ect todo!
}

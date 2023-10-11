pub mod borrowed;
pub mod graph;

pub enum DocumentNamespaceTy {}
pub type DocumentID = crate::FuzzID<DocumentNamespaceTy>;
pub enum StrokeLayerNamespaceTy {}
pub type StrokeLayerID = crate::FuzzID<StrokeLayerNamespaceTy>;

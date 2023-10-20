/// Requests generated by the user interface.
#[derive(Debug)]
pub enum UiRequest {
    Document {
        target: crate::FuzzID<crate::Document>,
        request: DocumentRequest,
    },
    /// Distinct from a document request, as it gives ownership of the document ID to the listener.
    NewDocument(crate::FuzzID<crate::Document>),
    SetBaseTool {
        tool: crate::pen_tools::StateLayer,
    },
}
/// Requests that apply to a specific layer of a specific document
#[derive(Debug)]
pub enum LayerRequest {
    Undo,
    Redo,
    /// This layer is now focused. For now, focus is a unique role, thus all other
    /// layers are to be unfocused when this request is acknowledged.
    Focus,
    Deleted,
    // Hmmmmm... Still unsure whether layers own their own blend, or if it should be
    // privided by an external entity (blend graph)
    SetBlend(crate::blend::Blend),
}
#[derive(Debug)]
pub enum DocumentViewRequest {
    Fit,
    /// Set the absolute scale. One document pixel = this many screen pixels.
    RealSize(f32),
    /// Set the absolute rotation, in radians CCW.
    SetRotation(f32),
}
/// Request that applies to a specific document
#[derive(Debug)]
pub enum DocumentRequest {
    Layer {
        // Todo: other kinds of layer.
        target: crate::FuzzID<crate::StrokeLayer>,
        request: LayerRequest,
    },
    /// Distinct from a layer request, as it gives ownership of the layer ID to the listener.
    NewLayer(crate::FuzzID<crate::StrokeLayer>),
    View(DocumentViewRequest),
    /// This document is now focused. For now, focus is a unique role, thus all other
    /// documents are to be unfocused when this request is acknowledged.
    Focus,
    Close,
    /// Save the document in-place
    Save,
    /// Save the document to the given path
    SaveCopy(std::path::PathBuf),
}

pub type RequestSender = std::sync::mpsc::Sender<UiRequest>;

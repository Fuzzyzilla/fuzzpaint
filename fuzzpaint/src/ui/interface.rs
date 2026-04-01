//! Data interface for the UI.

/// A viewport, a rectangular punchout of the UI inside of which a document is
/// shown and interacted through.
pub struct Viewport {
    /// The document that is shown.
    pub document: fuzzpaint_core::state::document::ID,
    /// The transform that the document is shown in.
    pub transform: crate::view_transform::ViewInfo,
}
/// The interface between the UI and the outside world. (the renderer, active
/// connections, etc).
pub struct InterfaceInner<'a, 'b: 'a> {
    /// Keyboard hotkey input for this viewport
    pub actions: (),
    /// Rich pointer input.
    pub pointers: &'a mut crate::window::stylus_events::PointerBridge,
    /// Remote and local connections.
    pub connections: &'a mut crate::connections::ConnectionsLock<'b>,
    /// Proxy for previews (any action in progress - dragging an opacity slider,
    /// in the process of drawing, etc.), optionally forwarding them to the
    /// renderer and the remote for realtime visual updates.
    pub preview: (),
    // The state of all document viewports. For now, there is only zero or one.
    pub document_viewport: Option<Viewport>,
}
// Seal the fields.
pub struct Interface<'a, 'b: 'a>(InterfaceInner<'a, 'b>);
impl<'a, 'b: 'a> From<InterfaceInner<'a, 'b>> for Interface<'a, 'b> {
    fn from(value: InterfaceInner<'a, 'b>) -> Self {
        Self(value)
    }
}
impl<'a, 'b: 'a> From<Interface<'a, 'b>> for InterfaceInner<'a, 'b> {
    fn from(value: Interface<'a, 'b>) -> Self {
        value.0
    }
}
impl<'a, 'b: 'a> Interface<'a, 'b> {
    pub fn into_inner(self) -> InterfaceInner<'a, 'b> {
        self.0
    }
    pub fn iter_connections(
        &'_ mut self,
    ) -> impl Iterator<
        Item = (
            crate::connections::ConnectionID,
            crate::connections::ConnectionLock<'_>,
        ),
    > {
        self.0.connections.iter_connections()
    }
    pub fn connect(&self, address: String) -> crate::connections::NewConnectionStatusReciever {
        self.0.connections.connect(address)
    }
    pub fn pointers(&mut self) -> &mut crate::window::stylus_events::PointerBridge {
        self.0.pointers
    }
    pub fn insert_document_viewport(&mut self, viewport: Viewport) {
        self.0.document_viewport = Some(viewport);
    }
}

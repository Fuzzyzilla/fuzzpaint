//! Data interface for the UI.

/// A viewport, a rectangular punchout of the UI inside of which a document is
/// shown and interacted through.
pub struct ViewportInner {
    /// The document that is shown.
    pub document: fuzzpaint_core::state::document::ID,
    /// Keyboard hotkey input for this viewport
    pub actions: (),
    /// The transform that the document is shown in.
    pub transform: ViewportTransform,
}
pub struct ViewportTransform {
    pub view: crate::view_transform::ViewInfo,
    pub changed: bool,
}
/// The interface between the UI and the outside world. (the renderer, active
/// connections, etc).
pub struct InterfaceInner<'a, 'b: 'a> {
    /// Rich pointer input.
    pub pointers: &'a mut crate::window::stylus_events::PointerBridge,
    /// Remote and local connections.
    pub connections: &'a mut crate::connections::ConnectionsLock<'b>,
    /// Proxy for previews (any action in progress - dragging an opacity slider,
    /// in the process of drawing, etc.), optionally forwarding them to the
    /// renderer and the remote for realtime visual updates.
    pub preview: (),
    // The state of all viewports. For now, there is only zero or one.
    pub viewport: Option<ViewportInner>,
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
    pub fn connect(&self, address: String) -> crate::connections::NewConnectionStatus {
        self.0.connections.connect(address)
    }
    pub fn pointers(&mut self) -> &mut crate::window::stylus_events::PointerBridge {
        &mut self.0.pointers
    }
}

//! Data interface for the UI.

/// A viewport, a rectangular punchout of the UI inside of which a document is
/// shown and interacted through.
pub struct Viewport {
    /// The document that is shown.
    document: (),

    /// The rectangle the document is shown in, in logical px.
    logical_rect: egui::Rect,

    /// Keyboard hotkey input for this viewport
    pub actions: (),
    /// The transform that the document is shown in.
    transform: (),
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
    // The state of all viewports. For now, there is exactly one.
    // pub viewports: Viewport,
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
impl Interface<'_, '_> {
    pub fn iter_connections(
        &mut self,
    ) -> impl Iterator<
        Item = (
            crate::connections::ConnectionID,
            crate::connections::ConnectionLock,
        ),
    > {
        self.0.connections.iter_connections()
    }
    pub fn pointers(&mut self) -> &mut crate::window::stylus_events::PointerBridge {
        &mut self.0.pointers
    }
}

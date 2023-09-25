//! # Pen Tools
//!
//! Pen tools are the way the user's pen interacts with the document and viewport. Brush, eraser, viewpan, viewscrub,
//! gizmo interactions, are all examples of pen tools.
//!
//! Implemented as a statemachine transitioning based on Actions. For example, brush will transition to viewpan
//! when DocumentPan action is activated. When DocumentPan is released, it will transition back to brush.
//!
//! Of course, users also must be able to use tools without holding an action down for accessibility as well as
//! conviniece for certain tasks.

/// A trait for the visual components of tools. Completely optional!
/// Register in [StateLayer::make_renderer]
pub trait PenToolPreview {}

enum TransitionCondition {
    Pressed(crate::actions::Action),
    Held(crate::actions::Action),
    NotHeld(crate::actions::Action),
}
#[derive(Copy, Clone, strum::EnumIter, Hash)]
enum StateLayer {
    Brush,
    DocumentPan,
    DocumentScrub,
    Gizmos,
}
impl StateLayer {
    // Owie boxed dyn...
    /// Make the renderer for this type of tool.
    pub fn make_renderer(
        &self,
        render_context: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> Option<Box<dyn PenToolPreview>> {
        match self {
            _ => None,
        }
    }
}
enum Transition {
    ToState(StateLayer),
    ToBase,
    NoChange,
}
pub struct ToolState {
    /// User-defined base state (depending on what tool is selected via the UI)
    base: StateLayer,
    /// Current machine state
    layer: Option<StateLayer>,
}
impl ToolState {
    /// Allow the tool to process the given stylus data and actions, optionally returning preview render commands,
    /// and possibly changing the tool's state.
    pub async fn process(
        &mut self,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: crate::actions::ActionFrame,
    ) -> Option<std::sync::Arc<crate::vk::PrimaryAutoCommandBuffer>> {
        let cur_state = self.layer.unwrap_or(self.base);

        todo!()
    }
}

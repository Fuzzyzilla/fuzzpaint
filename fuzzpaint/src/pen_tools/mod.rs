//! # Pen Tools
//!
//! Pen tools are the way the user's pen interacts with the document and viewport. Brush, eraser, viewpan, viewscrub,
//! gizmo interactions, are all examples of pen tools.
//!
//! Implemented as a statemachine transitioning based on Actions. For example, brush will transition to viewpan
//! when `DocumentPan` action is activated. When `DocumentPan` is released, it will transition back to brush.
//!
//! Of course, users also must be able to use tools without holding an action down for accessibility as well as
//! conviniece for certain tasks.

/// A trait for the visual components of tools. Completely optional!
/// Register in [`StateLayer::make_renderer`]
// This will box the future. It's totally possible for this to be
// static dispatch, but i was getting way caught up in the weeds trying to implement
// that and there's really no need :'P
mod brush;
mod dummy;
mod picker;
use crate::view_transform::ViewInfo;
trait MakePenTool {
    fn new_from_renderer(
        context: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn PenTool>>;
}
#[async_trait::async_trait]
trait PenTool {
    async fn process(
        &mut self,
        view_info: &ViewInfo,
        stylus_input: crate::window::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut ToolStateOutput,
    );
    /// Called when the state is transitioning away from this tool.
    fn exit(&mut self) {}
}

/// Allow tools to specify their transitions at runtime, or leave None
/// to provide default behavior.
struct ToolStateOutput {
    transition: Option<Transition>,
}
impl ToolStateOutput {
    /// Tell the tool state to read the actions and decide for itself what tool to transition
    /// to do, if any. Will happen regardless if no [`ToolStateOutput::with_transition`] is asserted.
    #[allow(dead_code)]
    pub fn with_default_behavior(&mut self) {
        self.transition = None;
    }
    /// Tell the tool state to perform this transition.
    #[allow(dead_code)]
    pub fn with_transition(&mut self, transition: Transition) {
        self.transition = Some(transition);
    }
    /// Compute default transition for the given actions.
    /// Does not have access to the current state on purpose, as custom
    /// behavior per-state should be implemented in the tool itself.
    fn do_default(_actions: &crate::actions::ActionFrame) -> Transition {
        Transition::ToBase
    }
}
#[derive(Copy, Clone, strum::EnumIter, Hash, PartialEq, Eq, Debug)]
pub enum StateLayer {
    Picker,
    Brush,
    Eraser,
}
#[derive(Clone, Copy)]
enum Transition {
    /// Layer this state on top the base. Note that states may not modify what
    /// state the base is!
    ToLayer(StateLayer),
    ToBase,
}
pub struct ToolState {
    /// User-defined base state (depending on what tool is selected via the UI)
    base: StateLayer,
    /// Current machine state
    layer: Option<StateLayer>,

    brush: Box<dyn PenTool>,
    eraser: Box<dyn PenTool>,
    picker: Box<dyn PenTool>,
}
impl ToolState {
    pub fn new_from_renderer(
        context: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            base: StateLayer::Brush,
            layer: None,
            brush: brush::Brush::new_from_renderer(context)?,
            eraser: brush::Eraser::new_from_renderer(context)?,
            picker: picker::Picker::new_from_renderer(context)?,
        })
    }
    /// Allow the tool to process the given stylus data and actions, optionally returning preview render commands,
    /// and possibly changing the tool's state.
    pub async fn process(
        &mut self,
        view_info: &ViewInfo,
        stylus_input: crate::window::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        ui_requests: &crossbeam::channel::Receiver<crate::ui::requests::UiRequest>,
    ) {
        use crate::ui::requests::{DocumentRequest, UiRequest};
        // Prepare output structs
        let mut tool_output = ToolStateOutput { transition: None };

        // Handle ui requests
        for request in ui_requests.try_iter() {
            match request {
                UiRequest::Document {
                    request: DocumentRequest::View(view_request),
                    ..
                } => (),
                UiRequest::SetBaseTool { tool } => self.set_base_state(tool),
                UiRequest::Document { .. } => (),
            }
        }

        // Get current tool and run
        let cur_state = self.get_current_state();
        let tool = self.tool_for_state(cur_state);

        tool.process(view_info, stylus_input, actions, &mut tool_output)
            .await;

        // Apply output structs
        let transition = tool_output
            .transition
            .unwrap_or_else(|| ToolStateOutput::do_default(actions));
        self.apply_state_transition(transition);

        let new_state = self.get_current_state();
        // Changed - tell cur_state to exit
        if cur_state != new_state {
            self.tool_for_state(cur_state).exit();
        }
    }
    fn tool_for_state(&mut self, state: StateLayer) -> &mut dyn PenTool {
        match state {
            StateLayer::Brush => self.brush.as_mut(),
            StateLayer::Eraser => self.eraser.as_mut(),
            StateLayer::Picker => self.picker.as_mut(),
        }
    }
    fn apply_state_transition(&mut self, transition: Transition) {
        match transition {
            Transition::ToBase => self.layer = None,
            Transition::ToLayer(layer) => self.layer = Some(layer),
        }
    }
    /// Set the resting state, where tools will go when no hotkey set.
    pub fn set_base_state(&mut self, state: StateLayer) {
        // Layer is none means this is an actual transition!
        // Alert it of the exit
        if self.layer.is_none() && self.base != state {
            self.tool_for_state(self.base).exit();
        }
        self.base = state;
    }
    #[must_use]
    pub fn get_current_state(&self) -> StateLayer {
        self.layer.unwrap_or(self.base)
    }
}

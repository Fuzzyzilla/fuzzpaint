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
// This will box the future. It's totally possible for this to be
// static dispatch, but i was getting way caught up in the weeds trying to implement
// that and there's really no need :'P
#[async_trait::async_trait]
trait PenTool {
    fn new_from_renderer(
        context: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: crate::actions::ActionFrame,
        tool_output: &mut ToolStateOutput,
        render_output: &mut ToolRenderOutput,
    );
}

/// Allow tools to specify their transitions at runtime, or leave None
/// to provide default behavior.
struct ToolStateOutput {
    transition: Option<Transition>,
}
impl ToolStateOutput {
    /// Tell the tool state to read the actions and decide for itself what tool to transition
    /// to do, if any. Will happen regardless if no [ToolStateOutput::with_transition] is asserted.
    pub fn with_default_behavior(&mut self) {
        self.transition = None
    }
    /// Tell the tool state to perform this transition.
    pub fn with_transition(&mut self, transition: Transition) {
        self.transition = Some(transition)
    }
    /// Compute default transition for the given actions.
    /// Does not have access to the current state on purpose, as custom
    /// behavior per-state should be implemented in the tool itself.
    fn do_default(actions: &crate::actions::ActionFrame) -> Transition {
        todo!()
    }
}
/// Interface for tools to (optionally) insert and read render data.
struct ToolRenderOutput<'a> {
    // A reference, to avoid the potentially expensive cost of cloning when the tool
    // doesn't end up caring :P
    render_task_messages: &'a tokio::sync::mpsc::UnboundedSender<crate::RenderMessage>,
}
impl ToolRenderOutput<'_> {}

enum TransitionCondition {
    Pressed(crate::actions::Action),
    Held(crate::actions::Action),
    NotHeld(crate::actions::Action),
}
#[derive(Copy, Clone, strum::EnumIter, Hash)]
pub enum StateLayer {
    Brush,
    DocumentPan,
    DocumentScrub,
    Gizmos,
}
enum Transition {
    /// Layer this state on top the base. Note that states may not modify what
    /// state the base is!
    ToLayer(StateLayer),
    ToBase,
    NoChange,
}
pub struct ToolState {
    /// User-defined base state (depending on what tool is selected via the UI)
    base: StateLayer,
    /// Current machine state
    layer: Option<StateLayer>,

    brush: Box<dyn PenTool>,
    document_pan: Box<dyn PenTool>,
    document_scrub: Box<dyn PenTool>,
    gizmos: Box<dyn PenTool>,
}
impl ToolState {
    /// Allow the tool to process the given stylus data and actions, optionally returning preview render commands,
    /// and possibly changing the tool's state.
    pub async fn process(
        &mut self,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: crate::actions::ActionFrame,
    ) -> Option<std::sync::Arc<crate::vk::PrimaryAutoCommandBuffer>> {
        // Prepare output structs
        let mut tool_output = ToolStateOutput { transition: None };
        let mut render_output = ToolRenderOutput {
            render_task_messages: todo!(),
        };

        // Get current tool and run
        let cur_state = self.get_current_state();
        let tool = self.tool_for_state(cur_state);

        tool.process(stylus_input, actions, &mut tool_output, &mut render_output)
            .await;

        // Apply output structs
        let transition = tool_output
            .transition
            .unwrap_or_else(|| ToolStateOutput::do_default(&actions));
        self.apply_state_transition(transition);
    }
    fn tool_for_state(&mut self, state: StateLayer) -> &mut dyn PenTool {
        match state {
            StateLayer::Brush => self.brush.as_mut(),
            StateLayer::DocumentPan => self.document_pan.as_mut(),
            StateLayer::DocumentScrub => self.document_scrub.as_mut(),
            StateLayer::Gizmos => self.gizmos.as_mut(),
        }
    }
    fn apply_state_transition(&mut self, transition: Transition) {
        match transition {
            Transition::NoChange => (),
            Transition::ToBase => self.layer = None,
            Transition::ToLayer(layer) => self.layer = Some(layer),
        }
    }
    /// Set the resting state, where tools will go when no hotkey set.
    pub fn set_base_state(&mut self, state: StateLayer) {
        self.base = state;
    }
    pub fn get_current_state(&self) -> StateLayer {
        self.layer.unwrap_or(self.base)
    }
}

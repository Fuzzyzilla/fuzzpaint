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
mod gizmo;
mod lasso;
mod picker;
mod viewport;
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
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut ToolStateOutput,
        render_output: &mut ToolRenderOutput,
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
    fn do_default(actions: &crate::actions::ActionFrame) -> Transition {
        use crate::actions::Action;
        // Wowie.. horrible... uhm uh
        if actions.is_action_held(Action::ViewportPan) {
            Transition::ToLayer(StateLayer::ViewportPan)
        } else if actions.is_action_held(Action::ViewportRotate) {
            Transition::ToLayer(StateLayer::ViewportRotate)
        } else if actions.is_action_held(Action::ViewportScrub) {
            Transition::ToLayer(StateLayer::ViewportScrub)
        } else if actions.is_action_held(Action::Gizmo) {
            Transition::ToLayer(StateLayer::Gizmos)
        } else {
            Transition::ToBase
        }
    }
}
/// Interface for tools to (optionally) insert and read render data.
pub struct ToolRenderOutput {
    // A reference, to avoid the potentially expensive cost of cloning 500 times per second when the tool
    // doesn't end up caring :P
    pub render_as: RenderAs,
    pub set_view: Option<crate::view_transform::DocumentTransform>,
    /// Set the cursor icon to this if Some, or default if None.
    pub cursor: Option<crate::gizmos::CursorOrInvisible>,
}

pub enum RenderAs {
    /// Render as these gizmos, in order.
    /// Gizmos are not interactible.
    InlineGizmos(smallvec::SmallVec<[crate::gizmos::Gizmo; 1]>),
    /// Render as this shared collection. Will be read locked during rendering.
    /// Gizmo is not interactible.
    SharedGizmoCollection(std::sync::Arc<tokio::sync::RwLock<crate::gizmos::Collection>>),
    /// Nothing to render.
    None,
}
#[derive(Copy, Clone, strum::EnumIter, Hash, PartialEq, Eq, Debug)]
pub enum StateLayer {
    Picker,
    Brush,
    Eraser,
    Gizmos,
    Lasso,
    ViewportPan,
    ViewportScrub,
    ViewportRotate,
}
#[derive(Clone, Copy)]
enum Transition {
    /// Layer this state on top the base. Note that states may not modify what
    /// state the base is!
    ToLayer(StateLayer),
    ToBase,
}
fn apply_transform_request(
    transform: &mut crate::view_transform::DocumentTransform,
    view: &ViewInfo,
    view_request: crate::ui::requests::DocumentViewRequest,
) {
    use crate::ui::requests::DocumentViewRequest;
    use crate::view_transform::{DocumentFit, DocumentTransform};

    // all the others require more work, this one is easy.
    if matches!(view_request, DocumentViewRequest::Fit) {
        // Todo: inherit the rotation, flip state.
        *transform = DocumentTransform::Fit(DocumentFit::default());
        return;
    }

    // I realllyyy need to refactor `cgmath` out
    let uv_to_cg = |ultraviolet::Vec2 { x, y }| cgmath::Point2 { x, y };
    let view_center = uv_to_cg(view.center());

    // Move it into a ViewInfo for easier processing.
    let mut cur_view = ViewInfo {
        transform: *transform,
        ..*view
    };
    let Some(xform) = cur_view.make_transformed() else {
        // Malformed transform.
        return;
    };

    match view_request {
        // Impl above
        DocumentViewRequest::Fit => unreachable!(),
        DocumentViewRequest::ZoomBy(factor) => {
            xform.scale_about(view_center, factor);
        }
        DocumentViewRequest::RealSize(size) => {
            // Calculate factor from current and desired.
            let cur_scale = xform.decomposed.scale;
            let factor = size / cur_scale;
            xform.scale_about(view_center, factor);
        }
        DocumentViewRequest::RotateBy(delta) => xform.rotate_about(view_center, cgmath::Rad(delta)),
        DocumentViewRequest::RotateTo(angle) => {
            // Calculate delta from current and destination.
            use cgmath::Rotation;
            let cur_unit = xform.decomposed.rot.rotate_vector(cgmath::vec2(1.0, 0.0));
            let cur_angle = cur_unit.y.atan2(cur_unit.x);
            let delta = angle - cur_angle;
            xform.rotate_about(view_center, cgmath::Rad(delta));
        }
    }
    *transform = cur_view.transform;
}
pub struct ToolState {
    /// User-defined base state (depending on what tool is selected via the UI)
    base: StateLayer,
    /// Current machine state
    layer: Option<StateLayer>,

    brush: Box<dyn PenTool>,
    eraser: Box<dyn PenTool>,
    picker: Box<dyn PenTool>,
    document_pan: Box<dyn PenTool>,
    document_scrub: Box<dyn PenTool>,
    document_rotate: Box<dyn PenTool>,
    gizmos: Box<dyn PenTool>,
    lasso: Box<dyn PenTool>,
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
            document_pan: viewport::Pan::new_from_renderer(context)?,
            document_scrub: viewport::Scrub::new_from_renderer(context)?,
            document_rotate: viewport::Rotate::new_from_renderer(context)?,
            gizmos: gizmo::Gizmo::new_from_renderer(context)?,
            lasso: lasso::Lasso::new_from_renderer(context)?,
        })
    }
    /// Allow the tool to process the given stylus data and actions, optionally returning preview render commands,
    /// and possibly changing the tool's state.
    pub async fn process(
        &mut self,
        view_info: &ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        ui_requests: &crossbeam::channel::Receiver<crate::ui::requests::UiRequest>,
    ) -> ToolRenderOutput {
        use crate::ui::requests::{DocumentRequest, UiRequest};
        // Prepare output structs
        let mut tool_output = ToolStateOutput { transition: None };
        let mut render_output = ToolRenderOutput {
            render_as: RenderAs::None,
            set_view: None,
            cursor: None,
        };

        // Handle ui requests
        for request in ui_requests.try_iter() {
            match request {
                UiRequest::Document {
                    request: DocumentRequest::View(view_request),
                    ..
                } => {
                    let transform = render_output.set_view.get_or_insert(view_info.transform);
                    apply_transform_request(transform, view_info, view_request);
                }
                UiRequest::SetBaseTool { tool } => self.set_base_state(tool),
                UiRequest::Document { .. } => (),
            }
        }

        // Get current tool and run
        let cur_state = self.get_current_state();
        let tool = self.tool_for_state(cur_state);

        tool.process(
            view_info,
            stylus_input,
            actions,
            &mut tool_output,
            &mut render_output,
        )
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
        // return the output, let the caller handle it.
        render_output
    }
    fn tool_for_state(&mut self, state: StateLayer) -> &mut dyn PenTool {
        match state {
            StateLayer::Brush => self.brush.as_mut(),
            StateLayer::Eraser => self.eraser.as_mut(),
            StateLayer::Picker => self.picker.as_mut(),
            StateLayer::ViewportPan => self.document_pan.as_mut(),
            StateLayer::ViewportScrub => self.document_scrub.as_mut(),
            StateLayer::ViewportRotate => self.document_rotate.as_mut(),
            StateLayer::Gizmos => self.gizmos.as_mut(),
            StateLayer::Lasso => self.lasso.as_mut(),
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

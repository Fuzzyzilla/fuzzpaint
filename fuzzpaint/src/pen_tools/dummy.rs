pub struct Dummy;

impl super::MakePenTool for Dummy {
    fn new_from_renderer(
        _context: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Dummy))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Dummy {
    async fn process(
        &mut self,
        _view_transform: &super::ViewInfo,
        _stylus_input: crate::stylus_events::StylusEventFrame,
        _actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        _render_output: &mut super::ToolRenderOutput,
    ) {
        // Do nothing.
        // Default behavior of tool_output should handle default action transitions.
    }
}

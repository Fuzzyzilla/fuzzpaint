pub struct Dummy;

impl super::MakePenTool for Dummy {
    fn new_from_renderer(
        context: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Dummy))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Dummy {
    async fn process(
        &mut self,
        view_transform: &crate::view_transform::ViewTransform,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        // Do nothing.
        // Default behavior of tool_output should handle default action transitions.
    }
}

pub struct Brush;

impl super::MakePenTool for Brush {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Brush))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Brush {
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        view_transform: &crate::view_transform::ViewTransform,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        todo!()
    }
}

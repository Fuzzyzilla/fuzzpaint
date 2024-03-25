pub struct Picker {
    was_down: bool,
}
impl super::MakePenTool for Picker {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Picker { was_down: false }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Picker {
    fn exit(&mut self) {
        self.was_down = false;
    }
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        _stylus_input: crate::stylus_events::StylusEventFrame,
        _actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        _render_output: &mut super::ToolRenderOutput,
    ) {
        // Someone got bored and frustrated halfway through writing this...
        let _requests: &mut tokio::sync::mpsc::Sender<crate::renderer::requests::RenderRequest> =
            return;

        // If we have a sampler already, track with the pen sampling everwhere where it's down.
        // If we don't have a sampler (or lose it midway through), take the last input, and if it's down, make sampler.
        // For now, naive impl!
        for event in &*_stylus_input {
            if !event.pressed && !self.was_down {
                // Just released, take a sample!
                let Some(globals) = crate::AdHocGlobals::read_clone() else {
                    return;
                };
                let (send, response) = tokio::sync::oneshot::channel();
                let req = crate::renderer::requests::RenderRequest::CreatePicker {
                    document: globals.document,
                    picker: crate::renderer::requests::PickerRequest::Composited(send),
                    info: crate::renderer::requests::PickerInfo {
                        input_points_per_viewport_pixel: 1.0, // TODO! We don't have access to this information at all yet.
                        viewport: *view_info,
                        sample_pos: ultraviolet::Vec2 {
                            x: event.pos.0,
                            y: event.pos.1,
                        },
                    },
                };
                let _ = _requests.send(req).await;
                if let Ok(Err(e)) = response.await {
                    log::trace!("{:?}", e);
                };
            }
            self.was_down = event.pressed;
        }
    }
}

enum ViewportManipulateType {
    Pan,
    Rotate,
    Scrub,
}
struct ViewportManipulateBase {
    manipulate_type: ViewportManipulateType,
    original_transform: Option<crate::view_transform::ViewTransform>,
    drag_start_pos: Option<ultraviolet::Vec2>,
}
impl ViewportManipulateBase {
    fn exit(&mut self) {
        self.drag_start_pos = None;
        self.original_transform = None;
    }
    async fn process(
        &mut self,
        view_transform: &crate::view_transform::ViewTransform,
        stylus_input: crate::stylus_events::StylusEventFrame,
        _actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput<'_>,
    ) {
        let mut new_transform = None::<crate::view_transform::ViewTransform>;
        for event in stylus_input.iter() {
            if event.pressed {
                let initial_transform = self
                    .original_transform
                    .get_or_insert(view_transform.clone());
                let start_pos = self.drag_start_pos.get_or_insert(ultraviolet::Vec2 {
                    x: event.pos.0,
                    y: event.pos.1,
                });

                let delta = (event.pos.0 - start_pos.x, event.pos.1 - start_pos.y);
                match self.manipulate_type {
                    ViewportManipulateType::Scrub => {
                        // Up or right is zoom in. This is natural for me as a right-handed
                        // person, but might ask around and see if this should be adjustable.
                        // certainly the speed should be :P
                        let scale = 1.01f32.powf(delta.0 - delta.1);

                        // Take the initial transform, and scale about the first drag point.
                        // If the transform becomes broken (returns err), don't use it.
                        let mut new = initial_transform.clone();
                        new.scale_about(
                            cgmath::Point2 {
                                x: start_pos.x,
                                y: start_pos.y,
                            },
                            scale,
                        );
                        new_transform = Some(new);
                    }
                    ViewportManipulateType::Pan => {
                        let mut new = initial_transform.clone();
                        new.pan(cgmath::Vector2 {
                            x: delta.0,
                            y: delta.1,
                        });
                        new_transform = Some(new);
                    }
                    ViewportManipulateType::Rotate => {
                        todo!()
                        /*let (viewport_pos, viewport_size) = document_preview.get_viewport();
                        let viewport_middle = viewport_pos + viewport_size / 2.0;

                        let start_angle = (start_pos.0 - viewport_middle.x)
                            .atan2(start_pos.1 - viewport_middle.y);
                        let now_angle = (event.pos.0 - viewport_middle.x)
                            .atan2(event.pos.1 - viewport_middle.y);
                        let delta = start_angle - now_angle;

                        let mut new = initial_transform.clone();
                        new.rotate_about(viewport_middle, cgmath::Rad(delta));
                        new_transform = Some(new);*/
                    }
                }
            } else {
                self.original_transform = None;
                self.drag_start_pos = None;
            }
        }
        // Set transform, if changed.
        if let Some(transform) = new_transform {
            render_output.set_view = Some(crate::view_transform::DocumentTransform::Transform(
                transform,
            ));
        }
    }
}

pub struct ViewportScrub {
    manipulate: ViewportManipulateBase,
}
impl super::MakePenTool for ViewportScrub {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(ViewportScrub {
            manipulate: ViewportManipulateBase {
                manipulate_type: ViewportManipulateType::Scrub,
                original_transform: None,
                drag_start_pos: None,
            },
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for ViewportScrub {
    fn exit(&mut self) {
        self.manipulate.exit()
    }
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        view_transform: &crate::view_transform::ViewTransform,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        self.manipulate
            .process(
                view_transform,
                stylus_input,
                actions,
                tool_output,
                render_output,
            )
            .await
    }
}
pub struct ViewportPan {
    manipulate: ViewportManipulateBase,
}
impl super::MakePenTool for ViewportPan {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(ViewportPan {
            manipulate: ViewportManipulateBase {
                manipulate_type: ViewportManipulateType::Pan,
                original_transform: None,
                drag_start_pos: None,
            },
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for ViewportPan {
    fn exit(&mut self) {
        self.manipulate.exit()
    }
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        view_transform: &crate::view_transform::ViewTransform,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        self.manipulate
            .process(
                view_transform,
                stylus_input,
                actions,
                tool_output,
                render_output,
            )
            .await
    }
}

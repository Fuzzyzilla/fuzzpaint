use crate::gizmos::CursorIcon;

enum ManipulationType {
    Pan,
    Rotate,
    Scrub,
}
impl ManipulationType {
    fn cursor(&self, holding: bool) -> CursorIcon {
        match self {
            Self::Pan => {
                if holding {
                    CursorIcon::Grabbing
                } else {
                    CursorIcon::Grab
                }
            }
            Self::Rotate => CursorIcon::EwResize,
            Self::Scrub => {
                if holding {
                    CursorIcon::NeswResize
                } else {
                    CursorIcon::ZoomIn
                }
            }
        }
    }
}
struct Base {
    manipulate_type: ManipulationType,
    original_transform: Option<crate::view_transform::ViewTransform>,
    drag_start_pos: Option<ultraviolet::Vec2>,
}
impl Base {
    fn exit(&mut self) {
        self.drag_start_pos = None;
        self.original_transform = None;
    }
    fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        _actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        let mut new_transform = None::<crate::view_transform::ViewTransform>;

        for event in &*stylus_input {
            if event.pressed {
                let initial_transform = match &self.original_transform {
                    Some(t) => *t,
                    None => {
                        let view = super::ViewInfo {
                            // Take new if available, else old.
                            transform: render_output.set_view.unwrap_or(view_info.transform),
                            ..*view_info
                        };
                        let Some(xform) = view.calculate_transform() else {
                            // Nothing for us to do with a malformed transform.
                            return;
                        };
                        *self.original_transform.insert(xform)
                    }
                };
                let start_pos = self.drag_start_pos.get_or_insert(ultraviolet::Vec2 {
                    x: event.pos.0,
                    y: event.pos.1,
                });

                let delta = (event.pos.0 - start_pos.x, event.pos.1 - start_pos.y);
                match self.manipulate_type {
                    ManipulationType::Scrub => {
                        // Up or right is zoom in. This is natural for me as a right-handed
                        // person, but might ask around and see if this should be adjustable.
                        // certainly the speed should be :P
                        let scale = 1.01f32.powf(delta.0 - delta.1);

                        // Take the initial transform, and scale about the first drag point.
                        // If the transform becomes broken (returns err), don't use it.
                        let mut new = initial_transform;
                        new.scale_about(
                            cgmath::Point2 {
                                x: start_pos.x,
                                y: start_pos.y,
                            },
                            scale,
                        );
                        new_transform = Some(new);

                        /*let xformed_point = view_info.transform.unproject(cgmath::Point2 { x: event.pos.0, y: event.pos.1 });

                        render_output.render_as = super::RenderAs::InlineGizmos(
                            [crate::gizmos::Gizmo {
                                visual: crate::gizmos::GizmoVisual::Shape {
                                    shape: crate::gizmos::RenderShape::Rectangle {
                                        position: ultraviolet::Vec2 { x: xformed_point.x, y: xformed_point.y },
                                        size: ultraviolet::Vec2 { x: delta.0, y: () },
                                        rotation: (),
                                    },
                                    texture: None,
                                    color: [0, 0, 0, 128],
                                },
                                ..Default::default()
                            }]
                            .into(),
                        )*/
                    }
                    ManipulationType::Pan => {
                        let mut new = initial_transform;
                        new.pan(cgmath::Vector2 {
                            x: delta.0,
                            y: delta.1,
                        });
                        new_transform = Some(new);
                    }
                    ManipulationType::Rotate => {
                        let viewport_middle =
                            view_info.viewport_position + view_info.viewport_size / 2.0;

                        let start_angle = (start_pos.x - viewport_middle.x)
                            .atan2(start_pos.y - viewport_middle.y);
                        let now_angle = (event.pos.0 - viewport_middle.x)
                            .atan2(event.pos.1 - viewport_middle.y);
                        let delta = start_angle - now_angle;

                        let mut new = initial_transform;
                        new.rotate_about(
                            cgmath::Point2 {
                                x: viewport_middle.x,
                                y: viewport_middle.y,
                            },
                            cgmath::Rad(delta),
                        );
                        new_transform = Some(new);
                    }
                }
            } else {
                self.original_transform = None;
                self.drag_start_pos = None;
            }
        }
        render_output.cursor = Some(crate::gizmos::CursorOrInvisible::Icon(
            self.manipulate_type.cursor(self.drag_start_pos.is_some()),
        ));
        // Set transform, if changed.
        if let Some(transform) = new_transform {
            render_output.set_view = Some(crate::view_transform::DocumentTransform::Transform(
                transform,
            ));
        }
    }
}

pub struct Scrub {
    manipulate: Base,
}
impl super::MakePenTool for Scrub {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Scrub {
            manipulate: Base {
                manipulate_type: ManipulationType::Scrub,
                original_transform: None,
                drag_start_pos: None,
            },
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Scrub {
    fn exit(&mut self) {
        self.manipulate.exit();
    }
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        self.manipulate
            .process(view_info, stylus_input, actions, tool_output, render_output);
    }
}
pub struct Pan {
    manipulate: Base,
}
impl super::MakePenTool for Pan {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Pan {
            manipulate: Base {
                manipulate_type: ManipulationType::Pan,
                original_transform: None,
                drag_start_pos: None,
            },
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Pan {
    fn exit(&mut self) {
        self.manipulate.exit();
    }
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        self.manipulate
            .process(view_info, stylus_input, actions, tool_output, render_output);
    }
}
pub struct Rotate {
    manipulate: Base,
}
impl super::MakePenTool for Rotate {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Rotate {
            manipulate: Base {
                manipulate_type: ManipulationType::Rotate,
                original_transform: None,
                drag_start_pos: None,
            },
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Rotate {
    fn exit(&mut self) {
        self.manipulate.exit();
    }
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        self.manipulate
            .process(view_info, stylus_input, actions, tool_output, render_output);
    }
}

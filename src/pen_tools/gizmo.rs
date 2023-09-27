use std::sync::Arc;

use crate::gizmos::GizmoTree;

mod visitors {
    use crate::gizmos::*;
    use std::ops::ControlFlow;
    pub struct CursorFindVisitor {
        pub viewport_cursor: ultraviolet::Vec2,
        pub xform_stack: Vec<crate::view_transform::ViewTransform>,
    }
    impl crate::gizmos::GizmoVisitor<CursorOrInvisible> for CursorFindVisitor {
        fn visit_collection(&mut self, gizmo: &Collection) -> ControlFlow<CursorOrInvisible> {
            // todo: transform point.
            let xformed = gizmo.transform.apply(
                self.xform_stack.first().unwrap(),
                self.xform_stack.last().unwrap(),
            );
            self.xform_stack.push(xformed);
            ControlFlow::Continue(())
        }
        fn end_collection(&mut self, _: &Collection) -> ControlFlow<CursorOrInvisible> {
            self.xform_stack.pop();
            ControlFlow::Continue(())
        }
        fn visit_gizmo(&mut self, gizmo: &Gizmo) -> ControlFlow<CursorOrInvisible> {
            let xform = gizmo.transform.apply(
                self.xform_stack.first().unwrap(),
                self.xform_stack.last().unwrap(),
            );
            let point = self.viewport_cursor;
            let xformed_point = xform
                .unproject(cgmath::Point2 {
                    x: point.x,
                    y: point.y,
                })
                .unwrap();
            // Short circuits the iteration if this returns Some
            if gizmo.hit_shape.hit([xformed_point.x, xformed_point.y]) {
                ControlFlow::Break(gizmo.hover_cursor)
            } else {
                ControlFlow::Continue(())
            }
        }
    }
}
pub struct Gizmo {
    shared_collection: Option<std::sync::Arc<tokio::sync::RwLock<crate::gizmos::Collection>>>,
    cursor_latch: Option<crate::gizmos::CursorOrInvisible>,
}

impl super::MakePenTool for Gizmo {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Gizmo {
            shared_collection: None,
            cursor_latch: None,
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Gizmo {
    fn exit(&mut self) {
        self.shared_collection = None;
        self.cursor_latch = None;
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
        let collection = self.shared_collection.get_or_insert_with(|| {
            use crate::gizmos::*;
            let mut collection = Collection::new(transform::GizmoTransform {
                position: ultraviolet::Vec2 { x: 10.0, y: 10.0 },
                origin_pinning: transform::GizmoOriginPinning::Document,
                scale_pinning: transform::GizmoTransformPinning::Viewport,
                rotation: 0.0,
                rotation_pinning: transform::GizmoTransformPinning::Viewport,
            });
            let square = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: GizmoVisual::Shape {
                    shape: RenderShape::Rectangle {
                        position: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                        size: ultraviolet::Vec2 { x: 20.0, y: 20.0 },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [128, 255, 255, 255],
                },
                hit_shape: GizmoShape::None,
                hover_cursor: CursorOrInvisible::Invisible,
                interaction: GizmoInteraction::None,
                transform: transform::GizmoTransform::inherit_all(),
            };
            let square2 = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: GizmoVisual::Shape {
                    shape: RenderShape::Rectangle {
                        position: ultraviolet::Vec2 { x: 15.0, y: 8.0 },
                        size: ultraviolet::Vec2 { x: 40.0, y: 10.0 },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [128, 0, 200, 255],
                },
                hit_shape: GizmoShape::None,
                hover_cursor: CursorOrInvisible::Invisible,
                interaction: GizmoInteraction::None,
                transform: transform::GizmoTransform {
                    origin_pinning: transform::GizmoOriginPinning::Inherit,
                    rotation_pinning: transform::GizmoTransformPinning::Document,
                    ..transform::GizmoTransform::inherit_all()
                },
            };
            let circle = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: GizmoVisual::Shape {
                    shape: RenderShape::Ellipse {
                        origin: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                        radii: ultraviolet::Vec2 { x: 20.0, y: 20.0 },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [128, 0, 0, 128],
                },
                hit_shape: GizmoShape::Ring {
                    outer: 20.0,
                    inner: 10.0,
                },
                hover_cursor: CursorOrInvisible::Icon(winit::window::CursorIcon::Help),
                interaction: GizmoInteraction::None,
                transform: transform::GizmoTransform {
                    scale_pinning: transform::GizmoTransformPinning::Document,
                    ..transform::GizmoTransform::inherit_all()
                },
            };
            collection.push_top(square);
            collection.push_top(square2);
            collection.push_bottom(circle);
            Arc::new(collection.into())
        });
        render_output.render_as = super::RenderAs::SharedGizmoCollection(collection.clone());

        if let Some(last) = stylus_input.last() {
            let collection = collection.write().await;
            let point = ultraviolet::Vec2 {
                x: last.pos.0,
                y: last.pos.1,
            };
            let base_xform = view_info.transform.clone();
            let mut visitor = visitors::CursorFindVisitor {
                viewport_cursor: point,
                xform_stack: vec![base_xform],
            };

            if let std::ops::ControlFlow::Break(cursor) = collection.visit_hit(&mut visitor) {
                self.cursor_latch = Some(cursor);
            } else {
                self.cursor_latch = None;
            }
        }
        render_output.cursor = self.cursor_latch;
    }
}

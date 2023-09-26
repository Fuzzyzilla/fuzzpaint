use std::sync::Arc;

pub struct Gizmo {
    shared_collection: Option<std::sync::Arc<tokio::sync::RwLock<crate::gizmos::Collection>>>,
}

impl super::MakePenTool for Gizmo {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Gizmo {
            shared_collection: None,
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Gizmo {
    fn exit(&mut self) {
        self.shared_collection = None;
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
                        origin: ultraviolet::Vec2 { x: 10.0, y: 0.0 },
                        radii: ultraviolet::Vec2 { x: 20.0, y: 20.0 },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [128, 0, 0, 128],
                },
                hit_shape: GizmoShape::None,
                hover_cursor: CursorOrInvisible::Invisible,
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
        // for event in stylus_input.iter() {}
    }
}

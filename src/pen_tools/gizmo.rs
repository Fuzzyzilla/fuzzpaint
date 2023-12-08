use std::sync::Arc;

mod visitors {
    use crate::gizmos::{Collection, CursorOrInvisible, Gizmo, GizmoInteraction};
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
    /// A path to find a specific tree node.
    /// A series of indices. [nth parent, nth child, nth grandchild, ...nth node]
    #[derive(Default)]
    pub struct VisitPath {
        indices: Vec<usize>,
    }
    pub struct ClickFindVisitor {
        pub viewport_cursor: ultraviolet::Vec2,
        pub path: VisitPath,
        pub xform_stack: Vec<crate::view_transform::ViewTransform>,
    }
    impl crate::gizmos::GizmoVisitor<VisitPath> for ClickFindVisitor {
        fn visit_collection(&mut self, gizmo: &Collection) -> ControlFlow<VisitPath> {
            self.path.indices.push(0);
            let xformed = gizmo.transform.apply(
                self.xform_stack.first().unwrap(),
                self.xform_stack.last().unwrap(),
            );
            self.xform_stack.push(xformed);
            ControlFlow::Continue(())
        }
        fn end_collection(&mut self, _: &Collection) -> ControlFlow<VisitPath> {
            self.xform_stack.pop();
            self.path.indices.pop();
            // May be none, if this is the top-level collection.
            if let Some(last_idx) = self.path.indices.last_mut() {
                *last_idx += 1;
            }
            ControlFlow::Continue(())
        }
        fn visit_gizmo(&mut self, gizmo: &Gizmo) -> ControlFlow<VisitPath> {
            if matches!(gizmo.interaction, GizmoInteraction::None) {
                *self.path.indices.last_mut().unwrap() += 1;
                return ControlFlow::Continue(());
            }
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
                ControlFlow::Break(std::mem::take(&mut self.path))
            } else {
                *self.path.indices.last_mut().unwrap() += 1;
                ControlFlow::Continue(())
            }
        }
    }
    /// Drills down into the gizmo tree to the given path. If found, calls exec on the gizmo,
    /// returning the results of F. Otherwise, may fallthrough with `ControlFlow::continue` or break with None.
    pub struct MutatorVisitor<'p, T, F: FnOnce(&mut Gizmo) -> T> {
        pub dest_path: &'p VisitPath,
        pub current_path: VisitPath,
        pub exec: Option<F>,
    }
    impl<T, F: FnOnce(&mut Gizmo) -> T> MutatorVisitor<'_, T, F> {
        // Did we advance too far? if so, we can short-circuit the search.
        fn too_far(&self) -> bool {
            // For each layer deep...
            for (dest, current) in self
                .dest_path
                .indices
                .iter()
                .zip(self.current_path.indices.iter())
            {
                match dest.cmp(current) {
                    // Everything matches before, then an index that's too small.
                    // Therefore, more work to do - not too far!
                    std::cmp::Ordering::Less => return false,
                    // Everything matches before, then an index that's too large.
                    // We've gone too far!
                    std::cmp::Ordering::Greater => return true,
                    // If eq, fall through and compare the next level deeper.
                    std::cmp::Ordering::Equal => (),
                }
            }
            // uhm uh how did we get here o.O
            // dest or current would be empty, or they matched exactly (up until one ended, at least.)
            // too eepy to think of correct behavior here, false is always safe :P
            false
        }
    }
    impl<T, F: FnOnce(&mut Gizmo) -> T> crate::gizmos::MutableGizmoVisitor<Option<T>>
        for MutatorVisitor<'_, T, F>
    {
        fn visit_collection_mut(&mut self, _: &mut Collection) -> ControlFlow<Option<T>> {
            self.current_path.indices.push(0);

            ControlFlow::Continue(())
        }
        fn end_collection_mut(&mut self, _: &mut Collection) -> ControlFlow<Option<T>> {
            self.current_path.indices.pop();
            *self.current_path.indices.last_mut().unwrap() += 1;

            if self.too_far() {
                ControlFlow::Break(None)
            } else {
                ControlFlow::Continue(())
            }
        }
        fn visit_gizmo_mut(&mut self, gizmo: &mut Gizmo) -> ControlFlow<Option<T>> {
            // Todo: inefficient for deep trees. prolly not a real issue tho.
            if self.current_path.indices == self.dest_path.indices {
                // Found!
                // Unwrap OK - we short circuit immediately after, no way we could
                // try and take it again.
                let t = (self.exec.take().unwrap())(gizmo);
                ControlFlow::Break(Some(t))
            } else {
                *self.current_path.indices.last_mut().unwrap() += 1;
                ControlFlow::Continue(())
            }
        }
    }
}
pub struct GizmoManipulator {
    shared_collection: Option<std::sync::Arc<tokio::sync::RwLock<crate::gizmos::Collection>>>,
    cursor_latch: Option<crate::gizmos::CursorOrInvisible>,
    clicked_path: Option<visitors::VisitPath>,
    was_pressed: bool,
}

impl super::MakePenTool for GizmoManipulator {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(GizmoManipulator {
            shared_collection: None,
            cursor_latch: None,
            clicked_path: None,
            was_pressed: false,
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for GizmoManipulator {
    fn exit(&mut self) {
        self.shared_collection = None;
        self.cursor_latch = None;
        self.clicked_path = None;
        self.was_pressed = false;
    }
    /// Process input, optionally returning a commandbuffer to be drawn.
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        _actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        use crate::gizmos::{
            transform, Collection, CursorOrInvisible, Gizmo, GizmoInteraction, GizmoShape,
            GizmoTree, MeshMode, MutGizmoTree, RenderShape, TextureMode, Visual,
        };
        let collection = self.shared_collection.get_or_insert_with(|| {
            let mut collection = Collection::new(transform::GizmoTransform {
                position: ultraviolet::Vec2 { x: 10.0, y: 10.0 },
                origin_pinning: transform::GizmoOriginPinning::Document,
                scale_pinning: transform::GizmoTransformPinning::Viewport,
                rotation: 0.0,
                rotation_pinning: transform::GizmoTransformPinning::Viewport,
            });
            let square = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: Visual {
                    mesh: MeshMode::Shape(RenderShape::Rectangle {
                        position: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                        size: ultraviolet::Vec2 { x: 20.0, y: 20.0 },
                        rotation: 0.0,
                    }),
                    texture: TextureMode::Solid([128, 255, 255, 255]),
                },
                hit_shape: GizmoShape::None,
                hover_cursor: CursorOrInvisible::Invisible,
                interaction: GizmoInteraction::None,
                transform: transform::GizmoTransform::inherit_all(),
            };
            let square2 = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: Visual {
                    mesh: MeshMode::Shape(RenderShape::Rectangle {
                        position: ultraviolet::Vec2 { x: 15.0, y: 8.0 },
                        size: ultraviolet::Vec2 { x: 40.0, y: 10.0 },
                        rotation: 0.0,
                    }),
                    texture: TextureMode::AntTrail,
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
                grab_cursor: CursorOrInvisible::Icon(winit::window::CursorIcon::Move),
                visual: Visual {
                    mesh: MeshMode::Shape(RenderShape::Ellipse {
                        origin: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                        radii: ultraviolet::Vec2 { x: 20.0, y: 20.0 },
                        rotation: 0.0,
                    }),
                    texture: TextureMode::AntTrail,
                },
                hit_shape: GizmoShape::Ring {
                    outer: 20.0,
                    inner: 10.0,
                },
                hover_cursor: CursorOrInvisible::Icon(winit::window::CursorIcon::Help),
                interaction: GizmoInteraction::Move,
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

        let mut collection = collection.write().await;

        for event in stylus_input.iter() {
            if event.pressed {
                // A new press!
                if !self.was_pressed {
                    // Perform hit test.
                    let point = ultraviolet::Vec2 {
                        x: event.pos.0,
                        y: event.pos.1,
                    };
                    let base_xform = view_info.transform.clone();
                    let mut visitor = visitors::ClickFindVisitor {
                        path: visitors::VisitPath::default(),
                        viewport_cursor: point,
                        xform_stack: vec![base_xform],
                    };

                    // Found?
                    if let std::ops::ControlFlow::Break(path) = collection.visit_hit(&mut visitor) {
                        self.clicked_path = Some(path);
                    } else {
                        self.clicked_path = None;
                    }
                }

                if let Some(path) = self.clicked_path.as_ref() {
                    let mut mutator_visitor = visitors::MutatorVisitor {
                        current_path: visitors::VisitPath::default(),
                        dest_path: path,
                        exec: Some(|g: &mut crate::gizmos::Gizmo| {
                            self.cursor_latch = Some(g.grab_cursor);
                        }),
                    };
                    collection.visit_hit_mut(&mut mutator_visitor);
                }
            } else {
                // Reset the click status, if any.
                self.clicked_path = None;

                // Not pressed. Search for hover cursor.
                // (might run multiple times per frame, wasteful!)
                let point = ultraviolet::Vec2 {
                    x: event.pos.0,
                    y: event.pos.1,
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
            self.was_pressed = event.pressed;
        }
        render_output.cursor = self.cursor_latch;
    }
}

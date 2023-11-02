pub struct Brush {
    in_progress_stroke: Option<crate::Stroke>,
    last_document: Option<crate::state::DocumentID>,
}

impl super::MakePenTool for Brush {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Brush {
            in_progress_stroke: None,
            last_document: None,
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Brush {
    fn exit(&mut self) {
        self.in_progress_stroke = None;
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
        // destructure the selections. Otherwise, bail.
        let Some(crate::AdHocGlobals {
            document,
            brush,
            node: Some(node),
        }) = crate::AdHocGlobals::read_clone()
        else {
            // Clear and bail.
            self.in_progress_stroke = None;
            return;
        };
        for event in stylus_input.iter() {
            if event.pressed {
                // Get stroke-in-progress or start anew.
                let this_stroke = self
                    .in_progress_stroke
                    .get_or_insert_with(|| crate::Stroke {
                        brush: crate::state::StrokeBrushSettings {
                            is_eraser: actions.is_action_held(crate::actions::Action::Erase),
                            ..brush.clone()
                        },
                        points: Vec::new(),
                    });
                let Ok(pos) = view_info
                    .transform
                    .unproject(cgmath::point2(event.pos.0, event.pos.1))
                else {
                    // If transform is ill-formed, we can't do work.
                    return;
                };

                // Calc cumulative distance from the start, or 0.0 if this is the first point.
                let dist = this_stroke
                    .points
                    .last()
                    .map(|last| {
                        let delta = [last.pos[0] - pos.x, last.pos[1] - pos.y];
                        last.dist + (delta[0] * delta[0] + delta[1] * delta[1]).sqrt()
                    })
                    .unwrap_or(0.0);

                this_stroke.points.push(crate::StrokePoint {
                    pos: [pos.x, pos.y],
                    pressure: event.pressure.unwrap_or(1.0),
                    dist,
                })
            } else {
                if let Some(stroke) = self.in_progress_stroke.take() {
                    // Insert the stroke into the document.
                    if let Some(Err(e)) = crate::default_provider().inspect(document, |queue| {
                        queue.write_with(|write| {
                            // Find the collection to insert into.
                            let collection_id = {
                                let graph = write.graph();
                                let node = graph.get(node).and_then(|node| node.leaf());
                                if let Some(crate::state::graph::LeafType::StrokeLayer {
                                    collection,
                                    ..
                                }) = node
                                {
                                    *collection
                                } else {
                                    anyhow::bail!("Current layer is not a valid stroke layer.")
                                }
                            };

                            // Get the collection
                            let mut collections = write.stroke_collections();
                            let Some(mut collection_writer) = collections.get_mut(collection_id)
                            else {
                                anyhow::bail!(
                                    "Current layer references nonexistant stroke collection"
                                )
                            };
                            // Huzzah, all is good! Upload stroke, and push it.
                            let immutable = TryInto::<
                                crate::state::stroke_collection::ImmutableStroke,
                            >::try_into(stroke)?;

                            // Destructure immutable stroke and push it.
                            // Invokes an extra ID allocation, weh
                            collection_writer
                                .push_back(immutable.brush, immutable.point_collection);

                            Ok(())
                        })
                    }) {
                        log::warn!("Failed to insert stroke: {e:?}");
                    }
                }
            }
        }
        render_output.render_as = if let Some((stroke, last)) = self
            .in_progress_stroke
            .as_ref()
            .and_then(|stroke| Some((stroke, stroke.points.last()?)))
        {
            let size = last.pressure;
            // Lerp between spacing and size_mul (matches gpu tess behaviour)
            let size = stroke.brush.spacing_px * (1.0 - size) + stroke.brush.size_mul * size;

            let gizmo = crate::gizmos::Gizmo {
                visual: crate::gizmos::GizmoVisual::Shape {
                    shape: crate::gizmos::RenderShape::Ellipse {
                        origin: ultraviolet::Vec2 {
                            x: last.pos[0],
                            y: last.pos[1],
                        },
                        radii: ultraviolet::Vec2 {
                            x: size / 2.0,
                            y: size / 2.0,
                        },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [0, 0, 0, 200],
                },
                ..Default::default()
            };
            render_output.cursor = Some(crate::gizmos::CursorOrInvisible::Invisible);
            super::RenderAs::InlineGizmos([gizmo].into())
        } else {
            render_output.cursor = Some(crate::gizmos::CursorOrInvisible::Icon(
                winit::window::CursorIcon::Crosshair,
            ));
            super::RenderAs::None
        }
    }
}

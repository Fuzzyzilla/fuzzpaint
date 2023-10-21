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
        /*
        let Some(globals) = crate::GLOBALS.get() else {
            return;
        };
        let (cur_document, cur_layer, cur_brush, ui_undos) = {
            let read = globals.selections().read().await;
            let Some(cur_document) = read.cur_document else {
                return;
            };
            let Some(cur_layer) = read
                .document_selections
                .get(&cur_document)
                .and_then(|selections| selections.cur_layer)
            else {
                return;
            };
            let brush = read.brush_settings.clone();

            let undos = read.undos.swap(0, std::sync::atomic::Ordering::Relaxed) as usize;

            (cur_document, cur_layer, brush, undos)
        };
        // This shouldn't be the responsibility of brush, but im just porting old code
        // as-is right now
        if Some(cur_document) != self.last_document {
            self.last_document = Some(cur_document.clone());
            // Notify renderer of the change
            let _ = render_output
                .render_task_messages
                .send(crate::RenderMessage::SwitchDocument(cur_document.clone()));
        }

        let key_undos = actions.action_trigger_count(crate::actions::Action::Undo);
        let key_redos = actions.action_trigger_count(crate::actions::Action::Redo);

        // Assume redos cancel undos.
        let net_undos = (key_undos + ui_undos) as isize - (key_redos as isize);

        if net_undos != 0 {
            let mut stroke_manager = globals.strokes().write().await;

            let layer_data =
                stroke_manager
                    .layers
                    .entry(cur_layer)
                    .or_insert_with(|| crate::StrokeLayerData {
                        strokes: vec![],
                        undo_cursor_position: None,
                    });

            match (layer_data.undo_cursor_position, net_undos) {
                // Redone when nothing undone - do nothin!
                (None, ..=0) => (),
                // New undos!
                (None, undos @ 1..) => {
                    // `as` cast ok - we checked that it's positive.
                    // clamp undos to the size of the data.
                    let undos = layer_data.strokes.len().min(undos as usize);
                    // Subtract won't overflow
                    layer_data.undo_cursor_position = Some(layer_data.strokes.len() - undos);
                    // broadcast the truncation request
                    let _ = render_output.render_task_messages.send(
                        crate::RenderMessage::StrokeLayer {
                            layer: cur_layer,
                            kind: crate::StrokeLayerRenderMessageKind::Truncate(undos),
                        },
                    );
                }
                // Redos when there were outstanding undos
                (Some(old_cursor), negative_redos @ ..=0) => {
                    // weird conventions I invented here :V
                    // `as` cast ok - we checked that it's negative, and inverted it.
                    let redos = (-negative_redos) as usize;
                    let new_cursor = old_cursor.saturating_add(redos);

                    // This many redos will move cursor past the end
                    let bounds = if new_cursor >= layer_data.strokes.len() {
                        layer_data.undo_cursor_position = None;
                        // append all strokes from the cursor onward
                        old_cursor..layer_data.strokes.len()
                    } else {
                        layer_data.undo_cursor_position = Some(new_cursor);

                        // append all strokes from the old cursor until the new
                        old_cursor..new_cursor
                    };

                    // Tell the renderer to append these strokes once more
                    for stroke in &layer_data.strokes[bounds] {
                        let _ = render_output.render_task_messages.send(
                            crate::RenderMessage::StrokeLayer {
                                layer: cur_layer,
                                kind: crate::StrokeLayerRenderMessageKind::Append(stroke.clone()),
                            },
                        );
                    }
                }
                // There were outstanding undos, and we're adding to them.
                (Some(old_cursor), undos @ 1..) => {
                    // Prevent undos from taking index below 0
                    // `as` cast ok - we checked that it's positive
                    let undos = old_cursor.min(undos as usize);

                    // Won't overflow
                    layer_data.undo_cursor_position = Some(old_cursor - undos);

                    let _ = render_output.render_task_messages.send(
                        crate::RenderMessage::StrokeLayer {
                            layer: cur_layer,
                            kind: crate::StrokeLayerRenderMessageKind::Truncate(undos),
                        },
                    );
                }
                // All cases are handled.
                _ => unreachable!(),
            }
        }
        for event in stylus_input.iter() {
            if event.pressed {
                // Get stroke-in-progress or start anew.
                let this_stroke = self
                    .in_progress_stroke
                    .get_or_insert_with(|| crate::Stroke {
                        brush: crate::state::StrokeBrushSettings {
                            is_eraser: actions.is_action_held(crate::actions::Action::Erase),
                            ..cur_brush.clone()
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
                    // Not pressed and a stroke exists - take it, freeze it, and put it on current layer!
                    let immutable: crate::ImmutableStroke = stroke.into();

                    let mut stroke_manager = globals.strokes().write().await;

                    let layer_data = stroke_manager.layers.entry(cur_layer).or_insert_with(|| {
                        crate::StrokeLayerData {
                            strokes: vec![],
                            undo_cursor_position: None,
                        }
                    });

                    // If there was an undo cursor, truncate everything after
                    // and replace with new data.
                    if let Some(cursor) = layer_data.undo_cursor_position.take() {
                        layer_data.strokes.drain(cursor..);
                    }
                    layer_data.strokes.push(immutable.clone());

                    let _ = render_output.render_task_messages.send(
                        crate::RenderMessage::StrokeLayer {
                            layer: cur_layer,
                            kind: crate::StrokeLayerRenderMessageKind::Append(immutable),
                        },
                    );
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
        */
    }
}

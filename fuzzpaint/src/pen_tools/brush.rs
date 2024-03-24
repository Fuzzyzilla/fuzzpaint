// Common core between eraser and brush
fn brush(
    is_eraser: bool,
    in_progress_stroke: &mut Option<crate::Stroke>,

    view: &super::ViewInfo,
    stylus_input: crate::stylus_events::StylusEventFrame,

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
        *in_progress_stroke = None;
        return;
    };
    let Some(view_transform) = view.calculate_transform() else {
        return;
    };
    for event in stylus_input.iter() {
        if event.pressed {
            // Get stroke-in-progress or start anew.
            let this_stroke = in_progress_stroke.get_or_insert_with(|| crate::Stroke {
                brush: fuzzpaint_core::state::StrokeBrushSettings { is_eraser, ..brush },
                points: Vec::new(),
            });
            let Ok(pos) = view_transform.unproject(cgmath::point2(event.pos.0, event.pos.1)) else {
                // If transform is ill-formed, we can't do work.
                return;
            };

            // Calc cumulative distance from the start, or 0.0 if this is the first point.
            let dist = this_stroke.points.last().map_or(0.0, |last| {
                let delta = [last.pos[0] - pos.x, last.pos[1] - pos.y];
                last.dist + (delta[0] * delta[0] + delta[1] * delta[1]).sqrt()
            });

            this_stroke.points.push(fuzzpaint_core::stroke::Point {
                pos: [pos.x, pos.y],
                pressure: event.pressure.unwrap_or(1.0),
                dist,
            });
        } else if let Some(stroke) = in_progress_stroke.take() {
            // Insert the stroke into the document.
            if let Some(Err(e)) = crate::global::provider().inspect(document, |queue| {
                queue.write_with(|write| {
                    // Find the collection to insert into.
                    let collection_id = {
                        let graph = write.graph();
                        let node = graph.get(node).and_then(|node| node.leaf());
                        if let Some(fuzzpaint_core::state::graph::LeafType::StrokeLayer {
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
                    let Some(mut collection_writer) = collections.get_mut(collection_id) else {
                        anyhow::bail!("Current layer references nonexistant stroke collection")
                    };
                    // Huzzah, all is good! Upload stroke, and push it.
                    let immutable = stroke.make_immutable();
                    drop(stroke);

                    // Destructure immutable stroke and push it.
                    // Invokes an extra ID allocation, weh
                    collection_writer.push_back(immutable.brush, immutable.point_collection);

                    Ok(())
                })
            }) {
                log::warn!("Failed to insert stroke: {e:?}");
            }
        }
    }
    render_output.render_as = if let Some((stroke, last)) = in_progress_stroke
        .as_ref()
        .and_then(|stroke| Some((stroke, stroke.points.last()?)))
    {
        let size = last.pressure;
        // Lerp between spacing and size_mul (matches gpu tess behaviour)
        let size = stroke.brush.spacing_px * (1.0 - size) + stroke.brush.size_mul * size;

        let gizmo = crate::gizmos::Gizmo {
            visual: crate::gizmos::Visual {
                mesh: crate::gizmos::MeshMode::Shape(crate::gizmos::RenderShape::Ellipse {
                    origin: ultraviolet::Vec2 {
                        x: last.pos[0],
                        y: last.pos[1],
                    },
                    radii: ultraviolet::Vec2 {
                        x: size / 2.0,
                        y: size / 2.0,
                    },
                    rotation: 0.0,
                }),
                texture: crate::gizmos::TextureMode::Solid([0, 0, 0, 200]),
            },
            ..Default::default()
        };
        render_output.cursor = Some(crate::gizmos::CursorOrInvisible::Invisible);
        super::RenderAs::InlineGizmos(
            [make_trail(in_progress_stroke.as_ref().unwrap()), gizmo]
                .into_iter()
                .collect(),
        )
    } else {
        render_output.cursor = Some(crate::gizmos::CursorOrInvisible::Icon(
            winit::window::CursorIcon::Crosshair,
        ));
        super::RenderAs::None
    }
}
fn make_trail(stroke: &crate::Stroke) -> crate::gizmos::Gizmo {
    use crate::gizmos::{transform::GizmoTransform, Gizmo, MeshMode, TextureMode, Visual};

    let mut points = Vec::with_capacity(stroke.points.len());
    let pressure_scale = stroke.brush.size_mul - stroke.brush.spacing_px;
    let pressure_offs = stroke.brush.spacing_px;
    points.extend(
        stroke
            .points
            .iter()
            .map(|point| crate::gizmos::renderer::WideLineVertex {
                pos: point.pos,
                // We use gizmo global color for this
                color: [255; 4],
                tex_coord: 0.0,
                width: point.pressure.mul_add(pressure_scale, pressure_offs),
            }),
    );

    let texture = if stroke.brush.is_eraser {
        TextureMode::AntTrail
    } else {
        let color = stroke.brush.color_modulate;
        // unmultiply
        let color = if color[3].abs() > 0.001 {
            [
                color[0] / color[3],
                color[1] / color[3],
                color[2] / color[3],
                color[3],
            ]
        } else {
            // Avoid div by zero
            [0.0; 4]
        };
        let color = [
            (color[0].clamp(0.0, 1.0) * 255.9999) as u8,
            (color[1].clamp(0.0, 1.0) * 255.9999) as u8,
            (color[2].clamp(0.0, 1.0) * 255.9999) as u8,
            (color[3].clamp(0.0, 1.0) * 255.9999) as u8,
        ];
        TextureMode::Solid(color)
    };

    Gizmo {
        visual: Visual {
            mesh: MeshMode::WideLineStrip(points.into()),
            texture,
        },
        transform: GizmoTransform::inherit_all(),
        ..Default::default()
    }
}

pub struct Brush {
    in_progress_stroke: Option<crate::Stroke>,
}
pub struct Eraser {
    in_progress_stroke: Option<crate::Stroke>,
}

impl super::MakePenTool for Brush {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Brush {
            in_progress_stroke: None,
        }))
    }
}
impl super::MakePenTool for Eraser {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Eraser {
            in_progress_stroke: None,
        }))
    }
}

#[async_trait::async_trait]
impl super::PenTool for Brush {
    fn exit(&mut self) {
        self.in_progress_stroke = None;
    }
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        brush(
            actions.is_action_held(crate::actions::Action::Erase),
            &mut self.in_progress_stroke,
            view_info,
            stylus_input,
            render_output,
        );
    }
}

#[async_trait::async_trait]
impl super::PenTool for Eraser {
    fn exit(&mut self) {
        self.in_progress_stroke = None;
    }
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        _actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        brush(
            true,
            &mut self.in_progress_stroke,
            view_info,
            stylus_input,
            render_output,
        );
    }
}

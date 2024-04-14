use fuzzpaint_core::stroke::{Archetype, Microseconds, StrokeSlice};

#[derive(Clone, Copy)]
pub struct InputPoint {
    pub position: [f32; 2],
    /// Time since start of stroke.
    pub time: Option<Microseconds>,
    /// Pressure, `[0,1]`
    pub pressure: Option<f32>,
    /// X,Y Tilt
    pub tilt: Option<[f32; 2]>,
    /// Distance from the surface, `[0,1]`
    pub distance: Option<f32>,
    /// Rotation of the pen, `[0, TAU]`
    pub roll: Option<f32>,
    /// "Wheel" position, unitless + unbounded.
    pub wheel: Option<f32>,
}
impl InputPoint {
    pub const DEFAULT_TIME: Microseconds = Microseconds(0);
    pub const DEFAULT_PRESSURE: f32 = 1.0;
    pub const DEFAULT_TILT: [f32; 2] = [0.0; 2];
    pub const DEFAULT_DISTANCE: f32 = 0.0;
    pub const DEFAULT_ROLL: f32 = 0.0;
    pub const DEFAULT_WHEEL: f32 = 0.0;
    #[must_use]
    pub fn archetype(&self) -> Archetype {
        const EMPTY: Archetype = Archetype::empty();

        Archetype::POSITION
            | self.time.map_or(EMPTY, |_| Archetype::TIME)
            | self.pressure.map_or(EMPTY, |_| Archetype::PRESSURE)
            | self.tilt.map_or(EMPTY, |_| Archetype::TILT)
            | self.distance.map_or(EMPTY, |_| Archetype::DISTANCE)
            | self.roll.map_or(EMPTY, |_| Archetype::ROLL)
            | self.wheel.map_or(EMPTY, |_| Archetype::WHEEL)
    }
    /// Remove all default values, replacing them with `None`.
    /// Eg, roll of `Some(0)` becomes `None`.
    ///
    /// Useful because some tablet APIs lie about actual capabilities and simply report `Some(0)` always :P
    #[must_use = "returns a new instance without modifying `self`"]
    pub fn without_defaults(self) -> Self {
        let Self {
            position,
            time,
            pressure,
            tilt,
            distance,
            roll,
            wheel,
        } = self;

        Self {
            position, // No default.
            time: time.filter(|&v| v != Self::DEFAULT_TIME),
            pressure: pressure.filter(|&v| v != Self::DEFAULT_PRESSURE),
            tilt: tilt.filter(|&v| v != Self::DEFAULT_TILT),
            distance: distance.filter(|&v| v != Self::DEFAULT_DISTANCE),
            roll: roll.filter(|&v| v != Self::DEFAULT_ROLL),
            wheel: wheel.filter(|&v| v != Self::DEFAULT_WHEEL),
        }
    }
    /// Add defaults to make up for any missing fields to match the given archetype.
    /// # Panics
    /// If `archetype` contains `ARC_LENGTH` which is unrepresentable by `Self`.
    #[must_use = "returns a new instance without modifying `self`"]
    pub fn or_defaults(self, archetype: Archetype) -> Self {
        let Self {
            position,
            time,
            pressure,
            tilt,
            distance,
            roll,
            wheel,
        } = self;

        Self {
            position,
            time: if archetype.intersects(Archetype::TIME) {
                Some(time.unwrap_or(Self::DEFAULT_TIME))
            } else {
                time
            },
            pressure: if archetype.intersects(Archetype::PRESSURE) {
                Some(pressure.unwrap_or(Self::DEFAULT_PRESSURE))
            } else {
                pressure
            },
            tilt: if archetype.intersects(Archetype::TILT) {
                Some(tilt.unwrap_or(Self::DEFAULT_TILT))
            } else {
                tilt
            },
            distance: if archetype.intersects(Archetype::DISTANCE) {
                Some(distance.unwrap_or(Self::DEFAULT_DISTANCE))
            } else {
                distance
            },
            roll: if archetype.intersects(Archetype::ROLL) {
                Some(roll.unwrap_or(Self::DEFAULT_ROLL))
            } else {
                roll
            },
            wheel: if archetype.intersects(Archetype::WHEEL) {
                Some(wheel.unwrap_or(Self::DEFAULT_WHEEL))
            } else {
                wheel
            },
        }
    }
}
impl Default for StrokeBuilder {
    fn default() -> Self {
        Self {
            position: vec![],
            time: vec![],
            pressure: vec![],
            tilt: vec![],
            distance: vec![],
            roll: vec![],
            wheel: vec![],
            current_archetype: Archetype::POSITION,
            packed_elements: vec![],
        }
    }
}
/// Structure-of-arrays form of a list of [`InputPoint`]s which can then be packed into a [`StrokeSlice`].
/// This allows for dynamic packing structure across strokes.
pub struct StrokeBuilder {
    /// Currently required! As such, this indicates len.
    position: Vec<[f32; 2]>,
    time: Vec<Microseconds>,
    pressure: Vec<f32>,
    tilt: Vec<[f32; 2]>,
    distance: Vec<f32>,
    roll: Vec<f32>,
    wheel: Vec<f32>,
    /// Which of the vecs are active?
    current_archetype: Archetype,
    /// On finish, write elements out to here and borrow them as a StrokeSlice.
    packed_elements: Vec<u32>,
}
impl StrokeBuilder {
    pub fn clear(&mut self) {
        self.position.clear();
        self.time.clear();
        self.pressure.clear();
        self.tilt.clear();
        self.distance.clear();
        self.roll.clear();
        self.wheel.clear();
        // Position is required.
        self.current_archetype = Archetype::POSITION;
    }
    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }
    pub fn len(&self) -> usize {
        self.position.len()
    }
    pub fn push(&mut self, point: InputPoint) {
        // Delete empty data
        let stripped = point.without_defaults();
        // See if any elements are new!
        let new_elements = stripped.archetype().difference(self.current_archetype);

        // Add missing elements in with defaults...
        if new_elements.intersects(Archetype::TIME) {
            self.time.resize(self.len(), InputPoint::DEFAULT_TIME);
        }
        if new_elements.intersects(Archetype::PRESSURE) {
            self.pressure
                .resize(self.len(), InputPoint::DEFAULT_PRESSURE);
        }
        if new_elements.intersects(Archetype::TILT) {
            self.tilt.resize(self.len(), InputPoint::DEFAULT_TILT);
        }
        if new_elements.intersects(Archetype::DISTANCE) {
            self.pressure
                .resize(self.len(), InputPoint::DEFAULT_DISTANCE);
        }
        if new_elements.intersects(Archetype::ROLL) {
            self.pressure.resize(self.len(), InputPoint::DEFAULT_ROLL);
        }
        if new_elements.intersects(Archetype::WHEEL) {
            self.pressure.resize(self.len(), InputPoint::DEFAULT_WHEEL);
        }
        self.current_archetype |= stripped.archetype();

        // Fill in the new point to match self
        let InputPoint {
            position,
            time,
            pressure,
            tilt,
            distance,
            roll,
            wheel,
        } = stripped.or_defaults(self.current_archetype);

        // Push it all!
        self.position.push(position);
        if let Some(v) = time {
            self.time.push(v);
        }
        if let Some(v) = pressure {
            self.pressure.push(v);
        }
        if let Some(v) = tilt {
            self.tilt.push(v);
        }
        if let Some(v) = distance {
            self.distance.push(v);
        }
        if let Some(v) = roll {
            self.roll.push(v);
        }
        if let Some(v) = wheel {
            self.wheel.push(v);
        }
    }
    /// Pack the contents and borrow them as a stroke.
    pub fn consume(&mut self) -> StrokeSlice {
        self.packed_elements.clear();

        let archetype = self.current_archetype | Archetype::ARC_LENGTH;
        let point_size = archetype.elements();
        let packed_len = point_size * self.len();
        self.packed_elements.resize(packed_len, 0);

        // Write out positions + calc arclens as we go!
        // (Positions + Arclen are required rn)
        {
            let mut arclen = 0.0f32;
            let mut last_position = None;
            for (idx, &position) in self.position.iter().enumerate() {
                let base = idx * point_size;
                let position_offs = archetype.offset_of(Archetype::POSITION).unwrap();
                self.packed_elements[base + position_offs] = bytemuck::cast(position[0]);
                self.packed_elements[base + position_offs + 1] = bytemuck::cast(position[1]);

                // Calc and write arc len
                let arclen_offs = archetype.offset_of(Archetype::ARC_LENGTH).unwrap();
                if let Some(last_position) = last_position.replace(position) {
                    let delta = [
                        last_position[0] - position[0],
                        last_position[1] - position[1],
                    ];
                    arclen += (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
                }
                self.packed_elements[base + arclen_offs] = bytemuck::cast(arclen);
            }
        }
        if archetype.intersects(Archetype::TIME) {
            for (idx, &v) in self.time.iter().enumerate() {
                let base = idx * point_size;
                let offs = archetype.offset_of(Archetype::TIME).unwrap();
                self.packed_elements[base + offs] = bytemuck::cast(v);
            }
        }
        if archetype.intersects(Archetype::PRESSURE) {
            for (idx, &v) in self.pressure.iter().enumerate() {
                let base = idx * point_size;
                let offs = archetype.offset_of(Archetype::PRESSURE).unwrap();
                self.packed_elements[base + offs] = bytemuck::cast(v);
            }
        }
        if archetype.intersects(Archetype::TILT) {
            for (idx, &v) in self.tilt.iter().enumerate() {
                let base = idx * point_size;
                let offs = archetype.offset_of(Archetype::TILT).unwrap();
                self.packed_elements[base + offs] = bytemuck::cast(v[0]);
                self.packed_elements[base + offs + 1] = bytemuck::cast(v[1]);
            }
        }
        if archetype.intersects(Archetype::DISTANCE) {
            for (idx, &v) in self.distance.iter().enumerate() {
                let base = idx * point_size;
                let offs = archetype.offset_of(Archetype::DISTANCE).unwrap();
                self.packed_elements[base + offs] = bytemuck::cast(v);
            }
        }
        if archetype.intersects(Archetype::ROLL) {
            for (idx, &v) in self.roll.iter().enumerate() {
                let base = idx * point_size;
                let offs = archetype.offset_of(Archetype::ROLL).unwrap();
                self.packed_elements[base + offs] = bytemuck::cast(v);
            }
        }
        if archetype.intersects(Archetype::WHEEL) {
            for (idx, &v) in self.wheel.iter().enumerate() {
                let base = idx * point_size;
                let offs = archetype.offset_of(Archetype::WHEEL).unwrap();
                self.packed_elements[base + offs] = bytemuck::cast(v);
            }
        }

        self.clear();
        StrokeSlice::new(&self.packed_elements, archetype).unwrap()
    }
}

// Common core between eraser and brush
fn brush(
    is_eraser: bool,
    builder: &mut StrokeBuilder,

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
        builder.clear();
        return;
    };
    let Some(view_transform) = view.calculate_transform() else {
        return;
    };
    for event in stylus_input.iter() {
        if event.pressed {
            let Ok(pos) = view_transform.unproject(cgmath::point2(event.pos.0, event.pos.1)) else {
                // If transform is ill-formed, we can't do work.
                return;
            };

            builder.push(InputPoint {
                position: [pos.x, pos.y],
                time: None,
                pressure: event.pressure,
                tilt: event.tilt.map(|(x, y)| [x, y]),
                distance: event.dist,
                roll: None,
                wheel: None,
            });
        } else if !builder.is_empty() {
            // Not pressed but a stroke exists - just finished, upload it!
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
                        anyhow::bail!("current layer references nonexistant stroke collection")
                    };

                    // Pack and store it away
                    let stroke = builder.consume();
                    let points = crate::global::points();
                    let Some(point_collection) = points.insert(stroke) else {
                        anyhow::bail!("stroke data too large")
                    };
                    // Destructure immutable stroke and push it.
                    // Invokes an extra ID allocation, weh
                    collection_writer.push_back(
                        fuzzpaint_core::state::StrokeBrushSettings { is_eraser, ..brush },
                        point_collection,
                    );

                    Ok(())
                })
            }) {
                builder.clear();
                log::warn!("failed to insert stroke: {e:?}");
            }
        }
    }
    render_output.render_as = if builder.is_empty() {
        render_output.cursor = Some(crate::gizmos::CursorOrInvisible::Icon(
            winit::window::CursorIcon::Crosshair,
        ));
        super::RenderAs::None
    } else {
        let base_size = brush.spacing_px.get();
        let size_factor = brush.size_mul.get() - base_size;
        let last_pos = builder.position.last().unwrap();
        let last_size = if let Some(pressure) = builder.pressure.last() {
            pressure * size_factor + base_size
        } else {
            brush.size_mul.get()
        };

        let brush_tip = crate::gizmos::Gizmo {
            visual: crate::gizmos::Visual {
                mesh: crate::gizmos::MeshMode::Shape(crate::gizmos::RenderShape::Ellipse {
                    origin: ultraviolet::Vec2 {
                        x: last_pos[0],
                        y: last_pos[1],
                    },
                    radii: ultraviolet::Vec2 {
                        x: last_size / 2.0,
                        y: last_size / 2.0,
                    },
                    rotation: 0.0,
                }),
                texture: crate::gizmos::TextureMode::Solid([0, 0, 0, 200]),
            },
            ..Default::default()
        };
        render_output.cursor = Some(crate::gizmos::CursorOrInvisible::Invisible);
        super::RenderAs::InlineGizmos(
            [
                make_trail(
                    builder,
                    base_size,
                    size_factor,
                    if is_eraser {
                        None
                    } else {
                        Some(brush.color_modulate)
                    },
                ),
                brush_tip,
            ]
            .into_iter()
            .collect(),
        )
    }
}
fn make_trail(
    stroke: &StrokeBuilder,
    min_size: f32,
    size_factor: f32,
    color: Option<fuzzpaint_core::util::Color>,
) -> crate::gizmos::Gizmo {
    use crate::gizmos::{transform::Transform, Gizmo, MeshMode, TextureMode, Visual};

    // Make trail:
    let mut points = Vec::with_capacity(stroke.len());
    // Fill in positions at 100% size
    points.extend(
        stroke
            .position
            .iter()
            .map(|&pos| crate::gizmos::renderer::WideLineVertex {
                pos,
                // We use gizmo global color for this
                color: [255; 4],
                tex_coord: 0.0,
                width: min_size + size_factor,
            }),
    );

    // Go back to fill in sizes if known
    if !stroke.pressure.is_empty() {
        points
            .iter_mut()
            .zip(stroke.pressure.iter())
            .for_each(|(point, pressure)| {
                point.width = pressure.mul_add(size_factor, min_size);
            });
    }

    let texture = match color.map(|c| c.as_array()) {
        Some([r, g, b, a]) => {
            // unmultiply
            let color = if a.abs() > 0.001 {
                [r / a, g / a, b / a, a]
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
        }
        None => TextureMode::AntTrail,
    };

    Gizmo {
        visual: Visual {
            mesh: MeshMode::WideLineStrip(points.into()),
            texture,
        },
        transform: Transform::inherit_all(),
        ..Default::default()
    }
}

pub struct Brush {
    stroke: StrokeBuilder,
}
pub struct Eraser {
    stroke: StrokeBuilder,
}

impl super::MakePenTool for Brush {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Brush {
            stroke: StrokeBuilder::default(),
        }))
    }
}
impl super::MakePenTool for Eraser {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Eraser {
            stroke: StrokeBuilder::default(),
        }))
    }
}

#[async_trait::async_trait]
impl super::PenTool for Brush {
    fn exit(&mut self) {
        self.stroke.clear();
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
            &mut self.stroke,
            view_info,
            stylus_input,
            render_output,
        );
    }
}

#[async_trait::async_trait]
impl super::PenTool for Eraser {
    fn exit(&mut self) {
        self.stroke.clear();
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
            &mut self.stroke,
            view_info,
            stylus_input,
            render_output,
        );
    }
}

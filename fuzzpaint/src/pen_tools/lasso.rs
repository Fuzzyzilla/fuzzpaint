/// A point collection who's `push` folds "close enough" points all into a single point.
/// Works based on the delta angle between previously seen and new points. Currently this is a hardcoded threshold.
#[derive(Default, Clone)]
struct TolerantCurve {
    points: Vec<ultraviolet::Vec2>,
    dir: Option<Dir>,
}
#[derive(Clone, Copy)]
struct Dir {
    // Last point to be within COS_TOLERANCE_ANGLE of points.last()
    last_good_point: ultraviolet::Vec2,

    // vector from points.last() to the (unsaved) point seen immediately after
    // If the of the newly recorded point from that frozen point is within margin of this dir,
    // it is the new last_good_point.
    // Otherwise, last_good_point gets pushed, and this new point becomes the new last_good_point and a new
    // dir is calculated.
    next_dir: ultraviolet::Vec2,
}
impl TolerantCurve {
    /// `cos(5deg)`
    const COS_TOLERANCE_ANGLE: f32 = 0.996_194_7;
    pub fn push(&mut self, point: ultraviolet::Vec2) {
        // First point!
        if self.points.is_empty() {
            self.points.push(point);
            self.dir = None;
            return;
        }
        // Last frozen point. Unwrap ok because of check above.
        let frozen = self.points.last().copied().unwrap();
        let mut dir = point - frozen;
        // Unreasonably small delta, skip
        if dir.mag_sq() < 0.00001 {
            return;
        }
        dir.normalize();

        if let Some(saved_dir) = self.dir.as_mut() {
            // We have a dir to compare with!
            if saved_dir.next_dir.dot(dir) > Self::COS_TOLERANCE_ANGLE {
                // Close enough to tolerance, use this as the new point
                saved_dir.last_good_point = point;
            } else {
                // Too far off angle!
                // Freeze last good point, set a new dir.
                self.points.push(saved_dir.last_good_point);
                saved_dir.last_good_point = point;
                saved_dir.next_dir = dir;
            }
        } else {
            // First point after a frozen point!

            // Set the dir and last_good_point
            self.dir = Some(Dir {
                next_dir: dir,
                last_good_point: point,
            });
        }
    }
    pub fn len(&self) -> usize {
        // if there's a pending point it will *always* add one more.
        self.points.len() + usize::from(self.dir.is_some())
    }
    pub fn into_unclosed_vec(self) -> Vec<ultraviolet::Vec2> {
        let Self { mut points, dir } = self;
        // Accept the last good point
        if let Some(dir) = dir {
            points.push(dir.last_good_point);
        }

        points
    }
    /// `into_unclosed_vec` except pushes the first point onto the end to form a closed loop if there are 2 or
    /// more points.
    pub fn into_closed_vec(self) -> Vec<ultraviolet::Vec2> {
        let mut unclosed = self.into_unclosed_vec();
        if unclosed.len() < 2 {
            unclosed
        } else {
            // Unwrap ok - we just checked len :P
            unclosed.push(unclosed.first().copied().unwrap());
            unclosed
        }
    }
}

fn make_trail(curve: &TolerantCurve) -> crate::gizmos::Gizmo {
    if curve.len() < 3 {
        // No render
        crate::gizmos::Gizmo::default()
    } else {
        // todo: horribly inefficient lol.
        let curve = curve.clone().into_closed_vec();
        // plus two due to lines adjacency!
        let mut points = Vec::with_capacity(curve.len() + 2);
        // push dummy to start at idx 1
        points.push(bytemuck::Zeroable::zeroed());
        points.extend(
            curve
                .into_iter()
                .map(|point| crate::gizmos::renderer::WideLineVertex {
                    pos: point.into(),
                    color: [255; 4],
                    tex_coord: 0.0,
                    width: 2.0,
                }),
        );

        // No panics. Guarded by curve.len() >= 3
        points[0] = *points.last().unwrap();
        points.push(points[1]);

        let mesh = crate::gizmos::MeshMode::WideLineStrip(points.into());

        crate::gizmos::Gizmo {
            visual: crate::gizmos::Visual {
                mesh,
                texture: crate::gizmos::TextureMode::AntTrail,
            },
            transform: crate::gizmos::transform::GizmoTransform::inherit_all(),
            ..Default::default()
        }
    }
}

pub struct Lasso {
    // We use a tolerance-based curve because of the high complexity
    // of searching for hits. Reducing the count of points will make it
    // much much much faster :3
    in_progress_hoop: Option<TolerantCurve>,
    is_down: bool,
}

impl super::MakePenTool for Lasso {
    fn new_from_renderer(
        _: &std::sync::Arc<crate::render_device::RenderContext>,
    ) -> anyhow::Result<Box<dyn super::PenTool>> {
        Ok(Box::new(Lasso {
            in_progress_hoop: None,
            is_down: false,
        }))
    }
}
#[async_trait::async_trait]
impl super::PenTool for Lasso {
    fn exit(&mut self) {
        self.is_down = false;
    }
    async fn process(
        &mut self,
        view_info: &super::ViewInfo,
        stylus_input: crate::stylus_events::StylusEventFrame,
        _actions: &crate::actions::ActionFrame,
        _tool_output: &mut super::ToolStateOutput,
        render_output: &mut super::ToolRenderOutput,
    ) {
        let Some(transform) = view_info.calculate_transform() else {
            return;
        };
        for input in stylus_input.iter() {
            // If new press, delete old.
            // if held, continue old.
            // Otherwise, ignore and keep old unchanged.
            let hoop = match (self.is_down, input.pressed) {
                (false, true) => {
                    self.in_progress_hoop = Some(TolerantCurve::default());
                    self.in_progress_hoop.as_mut()
                }
                (true, true) => self.in_progress_hoop.as_mut(),
                (_, false) => None,
            };
            self.is_down = input.pressed;

            if let Some(hoop) = hoop {
                let Ok(proj) = transform.unproject(cgmath::Point2 {
                    x: input.pos.0,
                    y: input.pos.1,
                }) else {
                    return;
                };
                hoop.push(ultraviolet::Vec2 {
                    x: proj.x,
                    y: proj.y,
                });
            }
        }
        if let Some(hoop) = self.in_progress_hoop.as_ref() {
            render_output.render_as = super::RenderAs::InlineGizmos([make_trail(hoop)].into());
        }
    }
}

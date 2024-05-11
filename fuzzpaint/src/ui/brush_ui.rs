use fuzzpaint_core::brush::{Brush, Texture, UniqueID};

pub struct CreationOutput {
    pub texture_data: Option<Vec<u8>>,
    pub brush: Brush,
}
pub struct CreationModal {
    // Imported texture + handle, egui tex frees on drop!
    texture_data: Option<Vec<u8>>,
    texture: Option<egui::TextureHandle>,
}
impl Default for CreationModal {
    fn default() -> Self {
        let () = ();
        Self {
            texture_data: None,
            texture: None,
        }
    }
}
impl super::Modal for CreationModal {
    type Cancel = ();
    type Confirm = CreationOutput;
    type Error = ();
    const NAME: &'static str = "Create Brush";
    fn do_ui(
        &mut self,
        ui: &mut egui::Ui,
    ) -> super::modal::Response<Self::Cancel, Self::Confirm, Self::Error> {
        ui.label("OwO");
        if ui.button("Leave me be, foul beeste.").clicked() {
            super::modal::Response::Cancel(())
        } else {
            super::modal::Response::Continue
        }
    }
}

/// Should be in same layout as specified in `archetype`!
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
#[repr(C)]
struct DemoStrokePoint {
    position: [f32; 2],
    arc_length: f32,
    pressure: f32,
}

/// Starting at the left, a squiggle is drawn with varied axes. x and y are normalized
/// to `[-width/2, width/2]` and `[-height/2, height/2]` respectively.
#[must_use]
fn make_demo_stroke(width: f32, height: f32) -> Vec<DemoStrokePoint> {
    const NUM_SAMPLES: u8 = 15;

    let xt = |t: f32| -> f32 {
        let offs = 1.25 * t + 0.5;

        let x = offs.powi(3) - 1.5 * offs * offs + 0.25;

        x * 0.5 * width
    };
    let yt = |t: f32| -> f32 { -4.0 * t.sin() * (-t * t + 1.0) * (-3.0 * t.abs()).exp() * height };
    /*
    let plot = |x: f32| -> f32 {
        // Scale factor to make y value range from [-0.5, 0.5]
        const SCALE_FACTOR: f32 = 0.5 / 0.82;

        let abs = x.abs();

        let sin_part = ((10.0 * x) / (2.0 * abs).sqrt()).sin();

        let decay_part = (-5.0 * abs).exp();

        sin_part * decay_part * SCALE_FACTOR
    };*/

    let mut points = Vec::<DemoStrokePoint>::with_capacity(NUM_SAMPLES.into());

    for i in 0..=NUM_SAMPLES {
        // [-1, 1]
        let t = (f32::from(2 * i) / f32::from(NUM_SAMPLES) - 1.0);

        // Go from 0 - 1 over time.
        let pressure = (t / 2.0 + 0.5).sqrt();
        let x = xt(t);
        let y = yt(t);

        let arc_length = points.last().map_or(0.0, |point| {
            let delta = [x - point.position[0], y - point.position[1]];

            let dist = (delta[0] * delta[0] + delta[1] * delta[1]).max(0.0).sqrt();

            point.arc_length + dist
        });

        points.push(DemoStrokePoint {
            position: [x, y],
            arc_length,
            pressure,
        });
    }

    points
}

/// Translates all the vertices in the mesh by a fixed amount in-place.
fn translate_mesh(mesh: &mut egui::Mesh, by: [f32; 2]) {
    for vert in &mut mesh.vertices {
        // We hope the optimizer vectorizes this!
        vert.pos += egui::vec2(by[0], by[1]);
    }
}

/// Fill the given rectangle with a tessellated demo stroke. Each stamp will use the texture specified with the full UV rect specified.
#[must_use]
fn tessellate(
    texture: egui::TextureId,
    uv: egui::Rect,
    color: egui::Color32,
    rect: egui::Rect,
    spacing: f32,
    radius: f32,
) -> egui::Mesh {
    assert!(spacing > f32::EPSILON);
    let rect = rect.shrink(radius);
    let points = make_demo_stroke(rect.width(), rect.height());
    let min_radius = spacing / 2.0;
    let mut mesh = egui::Mesh::with_texture(texture);

    // We need a rand, rather than pulling in a whole dep.
    // This should really be (along with the GPU side one) a repeatable non-hardware-influenced integer based one.
    let rand = |[x, y]: [f32; 2]| -> f32 {
        let dot = x * 12.9898 + y * 78.233;
        (dot.sin() * 43758.547).fract()
    };
    let interp = |from: DemoStrokePoint, to: DemoStrokePoint, t: f32| -> DemoStrokePoint {
        DemoStrokePoint {
            position: [
                egui::lerp(from.position[0]..=to.position[0], t),
                egui::lerp(from.position[1]..=to.position[1], t),
            ],
            arc_length: egui::lerp(from.arc_length..=to.arc_length, t),
            pressure: egui::lerp(from.pressure..=to.pressure, t),
        }
    };

    let Some(total_len) = points.last().map(|point| point.arc_length) else {
        // 0 len, empty mesh.
        return mesh;
    };

    let num_stamps = (total_len / spacing) as usize;

    // A quad per stamp.
    mesh.vertices.reserve_exact(num_stamps * 4);
    mesh.indices.reserve_exact(num_stamps * 6);

    // Pushes a quad derived from the given point onto `mesh`.
    let mut write_stamp = |point: DemoStrokePoint| {
        let minor_radius = min_radius + point.pressure * (radius - min_radius);
        // Diagonal size, such that a circular texture shows with `minor_radius`
        let major_radius = minor_radius * std::f32::consts::SQRT_2;
        let angle = rand(point.position) * std::f32::consts::TAU;
        let (sin, cos) = angle.sin_cos();
        let sin = sin * major_radius;
        let cos = cos * major_radius;

        let center = egui::pos2(point.position[0], point.position[1]);

        let base_index = mesh.vertices.len() as u32;
        // A diamond shape, rotated by the random angle. (Diamond instead of square is easier math lol, just +45 degrees ccw from the square.)
        // Vertices start at the "left" (-X) vertex, counterclockwise.
        mesh.vertices.extend_from_slice(&[
            egui::epaint::Vertex {
                uv: uv.left_top(),
                pos: center + egui::vec2(-cos, sin),
                color,
            },
            egui::epaint::Vertex {
                uv: uv.left_bottom(),
                pos: center + egui::vec2(sin, cos),
                color,
            },
            egui::epaint::Vertex {
                uv: uv.right_bottom(),
                pos: center + egui::vec2(cos, -sin),
                color,
            },
            egui::epaint::Vertex {
                uv: uv.right_top(),
                pos: center + egui::vec2(-sin, -cos),
                color,
            },
        ]);
        mesh.indices.extend_from_slice(&[
            base_index,
            base_index + 1,
            base_index + 2,
            base_index + 2,
            base_index + 3,
            base_index,
        ]);
    };

    let mut current_arclen = 0.0;

    points.windows(2).for_each(|points| {
        let &[before, after] = points else {
            unreachable!()
        };

        assert!(after.arc_length.is_finite());

        while current_arclen <= after.arc_length {
            let t = (current_arclen - before.arc_length) / (after.arc_length - before.arc_length);

            let mut point = interp(before, after, t);
            // Points are in a [-width, width],... space, shift them into the `rect`!
            point.position[0] += rect.center().x;
            point.position[1] += rect.center().y;

            write_stamp(point);

            current_arclen += spacing;
        }
    });

    mesh
}

pub fn test(ui: &mut egui::Ui) {
    let width = ui.available_width();
    let height = width / 3.0;

    let rect = egui::Rect::from_min_size(ui.next_widget_position(), egui::vec2(width, height));
    let painter = ui.painter_at(rect);

    /*let mesh = make_demo_stroke(rect.width(), rect.height());
    painter.add(egui::Shape::line(
        mesh.iter()
            .map(|point| {
                let pos = egui::Vec2::from(point.position);
                rect.center() + pos
            })
            .collect(),
        egui::Stroke {
            color: egui::Color32::BLACK,
            width: 2.0,
        },
    ));*/
    painter.add(egui::Shape::Mesh(tessellate(
        egui::TextureId::default(),
        egui::Rect::from_min_size(egui::epaint::WHITE_UV, egui::Vec2::ZERO),
        egui::Color32::BLACK.linear_multiply(0.2),
        rect,
        2.0,
        10.0,
    )));
    painter.rect_stroke(
        rect,
        5.0,
        egui::Stroke {
            color: egui::Color32::BLACK,
            width: 2.0,
        },
    );
}

pub struct Preloaded {
    id: UniqueID,
    texture: egui::TextureHandle,
    vertices: egui::Mesh,
}

/// Provides a brush selection drawer with many brushes loaded dynamically.
pub struct Bin {}

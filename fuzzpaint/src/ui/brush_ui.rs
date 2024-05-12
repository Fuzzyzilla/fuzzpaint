use fuzzpaint_core::brush::{Brush, Texture, UniqueID};

use super::ResponseExt;

const FULL_UV: egui::Rect = egui::Rect {
    min: egui::Pos2::ZERO,
    max: egui::Pos2 { x: 1.0, y: 1.0 },
};

#[derive(Copy, Clone, Default, PartialEq, Eq)]
enum CreationTab {
    #[default]
    Settings,
    Texture,
}

pub struct CreationOutput {
    pub texture_data: Option<Vec<u8>>,
    pub brush: Brush,
}
pub struct CreationModal {
    tab: CreationTab,
    // Imported texture handle, frees on drop!
    texture: Option<egui::TextureHandle>,
    uv_rect: egui::Rect,
    name: String,
    spacing_proportion: f32,
}
impl Default for CreationModal {
    fn default() -> Self {
        Self {
            tab: CreationTab::default(),
            texture: None,
            uv_rect: FULL_UV,
            name: "New Brush".to_owned(),
            spacing_proportion: 5.0,
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
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, CreationTab::Settings, "Settings");
            ui.selectable_value(&mut self.tab, CreationTab::Texture, "Texture");
        });
        ui.separator();

        match self.tab {
            CreationTab::Settings => {
                ui.text_edit_singleline(&mut self.name);
                ui.add(
                    egui::Slider::new(&mut self.spacing_proportion, 2.0..=100.0)
                        .text("Spacing")
                        .clamp_to_range(true)
                        .suffix("%"),
                );
            }
            CreationTab::Texture => {
                if ui.button(super::GROUP_ICON).clicked() {
                    if let Some(file) = rfd::FileDialog::default().pick_file() {
                        let try_load = || -> anyhow::Result<egui::TextureHandle> {
                            // `image` crate is probably not the choice here. It sweeps a lot of details under the rug and doesn't
                            // exactly do those details justice lol (colorspaces are wayy off)
                            let image = image::open(file)?.to_luma32f();
                            let manager = ui.ctx().tex_manager();
                            let mut write = manager.write();

                            let size = [image.width() as usize, image.height() as usize];

                            // Create a reference-counted image out of it, refs = 1
                            let texture_id = write.alloc(
                                "Preview brush texture".to_owned(),
                                egui::ImageData::Font(egui::FontImage {
                                    pixels: image.to_vec(),
                                    size,
                                }),
                                egui::TextureOptions {
                                    magnification: egui::TextureFilter::Nearest,
                                    minification: egui::TextureFilter::Linear,
                                    wrap_mode: egui::TextureWrapMode::ClampToEdge,
                                },
                            );

                            drop(write);

                            // This handle takes the only existing ref, dropping it destroys the image.
                            Ok(egui::TextureHandle::new(manager, texture_id))
                        };

                        match try_load() {
                            Ok(image) => self.texture = Some(image),
                            Err(err) => log::error!("Failed to load image: {err}"),
                        }
                    }
                }

                if let Some(texture) = self.texture.as_ref() {
                    let width = ui.available_width();

                    uv_picker(
                        ui,
                        egui::Vec2::splat(width),
                        &mut self.uv_rect,
                        FULL_UV,
                        texture.id(),
                    );
                }
            }
        }
        if let Some(texture) = self.texture.as_ref() {
            let width = ui.available_width();
            let height = width / 3.0;
            let size = egui::vec2(width, height);

            let (response, painter) = ui.allocate_painter(
                size,
                egui::Sense {
                    click: false,
                    drag: false,
                    focusable: false,
                },
            );

            let mesh = tessellate(
                texture.id(),
                self.uv_rect,
                egui::Color32::WHITE,
                response.rect,
                self.spacing_proportion / 100.0 * 10.0,
                10.0,
            );
            painter.rect_filled(response.rect, 0.0, egui::Color32::BLACK);
            painter.add(egui::Shape::mesh(mesh));
        }
        ui.separator();
        if ui.button("Cancel").clicked_or_escape() {
            super::modal::Response::Cancel(())
        } else {
            super::modal::Response::Continue
        }
    }
}

enum RGBAChannel {
    R,
    G,
    B,
    A,
}
enum RGBChannel {
    R,
    G,
    B,
}
enum LAChannel {
    L,
    A,
}
enum LChannel {
    L,
}

struct Swizzle<Channel> {
    channel: Channel,
    invert: bool,
}

struct ImageManager {
    image: image::DynamicImage,
}
/// Provides many buttons letting the user customize how their image is to be interpreted.
/// When a change is made that effects the image, it will be destroyed and rebuilt and the new handle will be
/// left in it's place. Returns `true` if the handle changed in this way.
fn image_mode(
    ui: &mut egui::Ui,
    image: &image::DynamicImage,
    handle: &mut egui::TextureHandle,
) -> bool {
    false
}

/// A picker showing an image background and allowing the user to pick a UV rectangle from it.
fn uv_picker(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    uv: &mut egui::Rect,
    max_uv: egui::Rect,
    texture: egui::TextureId,
) -> egui::Response {
    let (mut response, painter) = ui.allocate_painter(
        size,
        egui::Sense {
            click: false,
            drag: true,
            focusable: true,
        },
    );
    let rect = response.rect;
    if response.drag_started() {
        uv.min = egui::Pos2::ZERO
            + (response.interact_pointer_pos().unwrap() - rect.left_top()) / rect.size();
        uv.min = uv.min.clamp(max_uv.min, max_uv.max);
        uv.max = uv.min.clamp(max_uv.min, max_uv.max);
        response.mark_changed();
    } else if response.dragged() {
        uv.max = egui::Pos2::ZERO
            + (response.interact_pointer_pos().unwrap() - rect.left_top()) / rect.size();
        uv.max = uv.max.clamp(max_uv.min, max_uv.max);
        response.mark_changed();
    }

    let mut mesh = egui::Mesh::with_texture(texture);
    mesh.add_rect_with_uv(rect, FULL_UV, egui::Color32::WHITE);

    painter.add(egui::Shape::Mesh(mesh));

    // Where the current UV area is in UI space.
    let mut visual_uv_rect = egui::Rect::from_min_max(
        egui::pos2(
            egui::remap(uv.min.x, 0.0..=1.0, rect.min.x..=rect.max.x),
            egui::remap(uv.min.y, 0.0..=1.0, rect.min.y..=rect.max.y),
        ),
        egui::pos2(
            egui::remap(uv.max.x, 0.0..=1.0, rect.min.x..=rect.max.x),
            egui::remap(uv.max.y, 0.0..=1.0, rect.min.y..=rect.max.y),
        ),
    );

    // Round to px so that tiny gaps don't show up (very obvious lol)
    visual_uv_rect.min = painter.round_pos_to_pixels(visual_uv_rect.min);
    visual_uv_rect.max = painter.round_pos_to_pixels(visual_uv_rect.max);

    // Inverted UV rect is fine, but avoid visual artifacting with the below logic!
    if visual_uv_rect.min.x > visual_uv_rect.max.x {
        std::mem::swap(&mut visual_uv_rect.min.x, &mut visual_uv_rect.max.x);
    }
    if visual_uv_rect.min.y > visual_uv_rect.max.y {
        std::mem::swap(&mut visual_uv_rect.min.y, &mut visual_uv_rect.max.y);
    }

    // Draw an outline for the selected rect.
    painter.rect_stroke(
        visual_uv_rect,
        0.0,
        egui::Stroke {
            color: egui::Color32::BLACK,
            width: 1.0,
        },
    );
    // Darken the region outside the selection.
    let ghost_color = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200);
    // Left margin
    painter.rect_filled(rect.with_max_x(visual_uv_rect.min.x), 0.0, ghost_color);
    // right margin
    painter.rect_filled(rect.with_min_x(visual_uv_rect.max.x), 0.0, ghost_color);
    // Covers top and bottom region.
    let vertical_area = rect
        .with_min_x(visual_uv_rect.min.x)
        .with_max_x(visual_uv_rect.max.x);
    // Top center margin
    painter.rect_filled(
        vertical_area.with_max_y(visual_uv_rect.min.y),
        0.0,
        ghost_color,
    );
    // Bottom center margin
    painter.rect_filled(
        vertical_area.with_min_y(visual_uv_rect.max.y),
        0.0,
        ghost_color,
    );

    response
}

/// Changes the texture UVs of a tessellated demo stroke in-place.
fn change_tessellated_uv(mesh: &mut egui::Mesh, uv: egui::Rect) {
    // Mesh consists of many quads, all in order.
    mesh.vertices.chunks_exact_mut(4).for_each(|verts| {
        let [a, b, c, d] = verts else {
            unreachable!();
        };
        a.uv = uv.left_top();
        b.uv = uv.left_bottom();
        c.uv = uv.right_bottom();
        d.uv = uv.right_top();
    });
}
/// Remap a mesh from the given size to a new size in-place.
fn resize(mesh: &mut egui::Mesh, from: egui::Rect, to: egui::Rect) {}

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
    const NUM_SAMPLES: u8 = 31;

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

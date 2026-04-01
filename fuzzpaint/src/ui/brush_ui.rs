use fuzzpaint_core::brush::{Brush, Texture};
use fuzzpaint_types::resource::UniqueID;
use hashbrown::HashMap;

use super::ResponseExt;

const FULL_UV: egui::Rect = egui::Rect {
    min: egui::Pos2::ZERO,
    max: egui::Pos2 { x: 1.0, y: 1.0 },
};

#[derive(strum::IntoStaticStr, strum::EnumIter, Copy, Clone, Default, PartialEq, Eq)]
enum CreationTab {
    #[default]
    Settings,
    Texture,
    Dynamics(DynamicDestination, DynamicSourceNormalized),
}
#[derive(strum::IntoStaticStr, strum::EnumIter, Debug, PartialEq, Eq, Clone, Copy, Default)]
enum DynamicDestination {
    #[default]
    Size,
    Rotation,
    Offset,
    Flow,
    Spacing,
    // Hue,
    // Lightness,
}
/// Sources with a constant min an max, mapped to [0, 1].
#[derive(
    strum::IntoStaticStr, strum::EnumIter, Debug, PartialEq, Eq, Clone, Copy, Default, Hash,
)]
enum DynamicSourceNormalized {
    #[default]
    Pressure,
    Roll,
    TiltX,
    TiltY,
    Altitude,
    Azimuth,
    Direction,
    StampRandom,
    StrokeRandom,
}
impl DynamicSourceNormalized {
    fn domain_labels(self) -> &'static [&'static str] {
        match self {
            // Unitless, [0, 1]
            Self::Pressure | Self::StampRandom | Self::StrokeRandom => &["Min", "Max"],
            Self::Roll => &["0deg", "360deg"],
            // [-90, 90deg], but more descriptive of their directions. Negative
            // is towards top/left.
            Self::TiltX => &["Left", "Center", "Right"],
            Self::TiltY => &["Forward", "Center", "Back"],
            // [0, 90deg], but more descriptive of the directions:
            Self::Altitude => &["Horizontal", "Vertical"],
            // any angle, in canvas space. 0 = Right, increasing clockwise.
            // (This is a consequence of the y axis increasing downwards!)
            Self::Azimuth | Self::Direction => &["Right", "Down", "Left", "Up", "Right"],
        }
    }
    fn range(self) -> [f32; 2] {
        match self {
            Self::Pressure | Self::StampRandom | Self::StrokeRandom => [0.0, 1.0],
            Self::Roll | Self::Azimuth | Self::Direction => [0.0, 360.0],
            Self::TiltX | Self::TiltY => [-90.0, 90.0],
            Self::Altitude => [0.0, 90.0],
        }
    }
    fn description(self) -> &'static str {
        match self {
            DynamicSourceNormalized::Pressure => "How hard the stylus is pressed into the surface.",
            DynamicSourceNormalized::Roll => "Rotation of the stylus along it's own axis",
            DynamicSourceNormalized::TiltX => "Left-right tilt from vertical.",
            DynamicSourceNormalized::TiltY => "Forward-backward tilt from vertical.",
            DynamicSourceNormalized::Altitude => "Verticality of the pen relative to the surface.",
            DynamicSourceNormalized::Azimuth => "Direction of tilt.",
            DynamicSourceNormalized::Direction => "Stroke movement direction.",
            DynamicSourceNormalized::StampRandom => "A random value per-stamp.",
            DynamicSourceNormalized::StrokeRandom => "A random value per-stroke.",
        }
    }
}
/// Unnormalized Units of `[length] * [something]`, including where `something =
/// 1`. These are separated because the different kinds of DPI relations
/// influence how these are calculated depending on which length unit is used.
#[derive(strum::IntoStaticStr, strum::EnumIter, Debug, PartialEq, Eq, Clone, Copy)]
enum DynamicSourceUnnormalizedLengthNumerator {
    Speed,
    Distance,
}
impl DynamicSourceUnnormalizedLengthNumerator {
    /// Just the end of the unit, without the `length` part.
    fn unit_suffix(self) -> &'static str {
        match self {
            Self::Speed => "/s",
            Self::Distance => "",
        }
    }
}
#[derive(Clone, Copy, strum::EnumIter, Default)]
pub enum PhysicalUnit {
    #[default]
    Centimeter,
    Inch,
    Point,
}
impl PhysicalUnit {
    fn per_centimeter(self) -> f32 {
        match self {
            Self::Point => 28.346_457,
            Self::Inch => 2.54,
            Self::Centimeter => 1.0,
        }
    }
    fn unit(self) -> &'static str {
        match self {
            Self::Point => "pt",
            Self::Inch => "in",
            Self::Centimeter => "cm",
        }
    }
}
#[derive(Clone, Copy, strum::EnumIter)]
pub enum LengthUnit {
    /// Logical pixels, scale-factor aware. (e.g., if you render at twice the
    /// scale factor, this will scale up 2x.)
    LogicalPx,
    /// Screen pixels, non-scale-factor aware. (e.g., if you render at twice the
    /// scale factor, this will not scale up, thus becoming proportionally
    /// smaller)
    PhysicalPx,
    /// Depends on the DPI of the document. Just centimeters times a constant.
    Physical(PhysicalUnit),
}
impl LengthUnit {
    fn unit(self) -> &'static str {
        match self {
            Self::LogicalPx => "px",
            Self::PhysicalPx => "ppx",
            Self::Physical(unit) => unit.unit(),
        }
    }
}
/// Unnormalized Units that do not involve length.
#[derive(strum::IntoStaticStr, strum::EnumIter, Debug, PartialEq, Eq, Clone, Copy)]
enum DynamicSourceUnnormalized {
    Time,
}
impl DynamicSourceUnnormalized {
    fn unit(self) -> &'static str {
        match self {
            Self::Time => "s",
        }
    }
}
struct CurveWithMax {
    max: f32,
    curve: CurveNormalized,
}

#[derive(Clone, Copy)]
struct CurvePoint {
    norm_x: f32,
    norm_y: f32,
}
struct CurveNormalized {
    min_y: f32,
    max_y: f32,
    // Always ordered by norm_x, always at least two elements. The first element
    // will have norm_x = 0.0, the last will have norm_x = 1.0.
    points: Vec<CurvePoint>,
    dragged_idx: Option<usize>,
}
impl CurveNormalized {
    const POINT_RADIUS_PX: f32 = 7.0;
    fn new(min: f32, max: f32) -> Self {
        Self {
            min_y: min,
            max_y: max,
            points: vec![
                CurvePoint {
                    norm_x: 0.0,
                    norm_y: 0.0,
                },
                CurvePoint {
                    norm_x: 1.0,
                    norm_y: 1.0,
                },
            ],
            dragged_idx: None,
        }
    }
    fn insert_and_drag(&mut self, norm_x: f32, norm_y: f32) {
        match self
            .points
            .binary_search_by(|point| point.norm_x.total_cmp(&norm_x))
        {
            Ok(found_idx) => {
                self.points[found_idx].norm_y = norm_y;
                self.dragged_idx = Some(found_idx);
            }
            Err(would_be_idx) => {
                self.points
                    .insert(would_be_idx, CurvePoint { norm_x, norm_y });
                self.dragged_idx = Some(would_be_idx);
            }
        }
        let dragged_idx = self.dragged_idx.unwrap();
        if dragged_idx == 0 {
            self.points[dragged_idx].norm_x = 0.0;
        } else if dragged_idx == self.points.len() - 1 {
            self.points[dragged_idx].norm_x = 1.0;
        }
    }
    fn show(
        &mut self,
        x_unit: DynamicSourceNormalized,
        y_unit: &str,
        ui: &mut egui::Ui,
    ) -> egui::Response {
        let width = ui.available_width();
        let (rect, mut response) =
            ui.allocate_at_least(egui::Vec2::splat(width), egui::Sense::click_and_drag());
        if response.is_pointer_button_down_on() {
            let pos = response.interact_pointer_pos().unwrap();
            let mut norm_pos = (pos - rect.min) / rect.size();
            norm_pos = norm_pos.clamp(egui::Vec2::ZERO, egui::Vec2::ONE);
            norm_pos.y = 1.0 - norm_pos.y;

            if let Some(dragged_idx) = self.dragged_idx {
                // Remove, update, then re-insert. This way, the slice remains
                // sorted.
                let _ = self.points.remove(dragged_idx);
                self.insert_and_drag(norm_pos.x, norm_pos.y);
            } else {
                // New click! Find which is grabbed.
                for (i, point) in self.points.iter().enumerate() {
                    let delta = egui::Vec2::new(point.norm_x, point.norm_y) - norm_pos;
                    if (delta * rect.size()).length_sq()
                        < Self::POINT_RADIUS_PX * Self::POINT_RADIUS_PX
                    {
                        // Grabbed this one!
                        self.dragged_idx = Some(i);
                        break;
                    }
                }
                if self.dragged_idx.is_none() {
                    // Didn't grab any, add one!
                    self.insert_and_drag(norm_pos.x, norm_pos.y);
                }
            }
            response.mark_changed();
        } else {
            self.dragged_idx = None;
        }
        if ui.will_discard() {
            // Skip drawing, we've already done all the layout egui needs :3
            return response;
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
        for (i, point) in self.points.iter().enumerate() {
            let pos = rect.min + egui::Vec2::new(point.norm_x, 1.0 - point.norm_y) * rect.size();
            let is_dragged = self.dragged_idx == Some(i);
            let color = ui.visuals().text_color();
            if is_dragged {
                painter.circle_filled(pos, Self::POINT_RADIUS_PX, color);
            } else {
                painter.circle_stroke(pos, Self::POINT_RADIUS_PX, egui::Stroke::new(2.0, color));
            }
        }
        {
            let text_height = ui.text_style_height(&egui::TextStyle::Small);
            let font_id = egui::FontId::monospace(text_height);
            let text_color = ui.visuals().weak_text_color();
            painter.text(
                rect.right_top(),
                egui::Align2::RIGHT_TOP,
                format!("{}{y_unit}", self.max_y),
                font_id.clone(),
                text_color,
            );
            painter.text(
                rect.right_bottom(),
                egui::Align2::RIGHT_BOTTOM,
                format!("{}{y_unit}", self.min_y),
                font_id.clone(),
                text_color,
            );
            let labels = x_unit.domain_labels();
            for (i, label) in labels.iter().enumerate() {
                let norm_x = i as f32 / (labels.len() - 1) as f32;
                let pos = egui::Pos2::new(norm_x * rect.width() + rect.min.x, rect.bottom());
                let anchor = if i == 0 {
                    egui::Align2::LEFT_BOTTOM
                } else if i == labels.len() - 1 {
                    egui::Align2::RIGHT_BOTTOM
                } else {
                    painter.line_segment(
                        [
                            egui::Pos2::new(pos.x, rect.bottom()),
                            egui::Pos2::new(pos.x, rect.top()),
                        ],
                        // more semantically correct would be weak_bg, but
                        // for some reason it's invisible on extreme_bg
                        egui::Stroke::new(1.0, ui.visuals().window_fill),
                    );
                    egui::Align2::CENTER_BOTTOM
                };
                painter.text(pos, anchor, label, font_id.clone(), text_color);
            }
            if let Some(hover) = response.hover_pos() {
                let norm_pos = if let Some(dragged_idx) = self.dragged_idx {
                    let dragged = self.points[dragged_idx];
                    egui::Vec2::new(dragged.norm_x, dragged.norm_y)
                } else {
                    (hover - rect.min) / rect.size()
                };
                let [min, max] = x_unit.range();
                let human_readable_x_value = norm_pos.x * (max - min) + min;
                painter.text(
                    rect.left_top(),
                    egui::Align2::LEFT_TOP,
                    format!("{human_readable_x_value}"),
                    font_id.clone(),
                    text_color,
                );
            }
        }
        let points = self
            .points
            .iter()
            .map(|point| rect.min + egui::Vec2::new(point.norm_x, 1.0 - point.norm_y) * rect.size())
            .collect::<Vec<_>>();
        painter.line(points, egui::Stroke::new(2.0, ui.visuals().text_color()));

        response
    }
}

#[derive(Copy, Clone, strum::EnumIter, Default)]
pub enum DynamicCombinator {
    #[default]
    Multipy,
    Add,
    Min,
    Max,
    Width,
    Average,
}
#[derive(Default)]
pub struct CurveSetNormalized {
    curves: HashMap<DynamicSourceNormalized, CurveNormalized>,
}
impl CurveSetNormalized {
    fn show(&mut self, selected_source: &mut DynamicSourceNormalized, ui: &mut egui::Ui) {
        egui::Panel::left("dynamic_sources_panel")
            .resizable(false)
            .show_inside(ui, |ui| {
                for source in <DynamicSourceNormalized as strum::IntoEnumIterator>::iter() {
                    ui.horizontal(|ui| {
                        let exists = self.curves.contains_key(&source);
                        let mut checked = exists;
                        ui.checkbox(&mut checked, ());
                        if checked != exists {
                            if checked {
                                self.curves.insert(source, CurveNormalized::new(0.0, 1.0));
                                *selected_source = source;
                            } else {
                                self.curves.remove(&source);
                            }
                        }
                        ui.selectable_value(selected_source, source, <&'static str>::from(source))
                            .on_hover_text(source.description());
                    });
                }
            });

        ui.label(selected_source.description());
        if let Some(curve) = self.curves.get_mut(selected_source) {
            curve.show(*selected_source, "", ui);
        }
    }
}
pub struct NormalizedDynamic {
    base: f32,
    combinator: DynamicCombinator,
    curves: CurveSetNormalized,
}
pub struct LengthDynamic {
    unit: LengthUnit,
    base: f32,
    combinator: DynamicCombinator,
}
pub enum SpacingMode {
    /// Expressed as a ratio of the resulting `size` after dynamics.
    /// i.e. 1.0 = stamps perfectly side-by-side, 0.5 = 50% overlap, etc.
    Ratio(f32),
    /// Expressed as a regular dynamic.
    Custom(LengthDynamic),
}
pub struct Dynamics {
    size: LengthDynamic,
    rotation: NormalizedDynamic,
    offset: LengthDynamic,
    flow: NormalizedDynamic,
    spacing: SpacingMode,
}
pub enum StampKind {
    Circle(CircleStamp),
    Texture(TextureStamp),
}
impl Default for StampKind {
    fn default() -> Self {
        Self::Circle(CircleStamp::default())
    }
}
impl StampKind {
    fn show(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_salt(ui.id())
            .selected_text(match self {
                StampKind::Circle(_) => "Circle",
                StampKind::Texture(_) => "Texture",
            })
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(matches!(self, StampKind::Circle(_)), "Circle")
                    .clicked()
                {
                    *self = StampKind::Circle(CircleStamp::default());
                }
                if ui
                    .selectable_label(matches!(self, StampKind::Texture(_)), "Texture")
                    .clicked()
                {
                    *self = StampKind::Texture(TextureStamp::default());
                }
            });
        match self {
            StampKind::Circle(circle) => {
                circle.show(ui);
            }
            StampKind::Texture(texture) => {
                texture.show(ui);
            }
        }
    }
}
pub struct TextureStamp {
    // Imported texture handle, frees on drop!
    texture: Option<egui::TextureHandle>,
    uv_rect: egui::Rect,
}
impl TextureStamp {
    fn show(&mut self, ui: &mut egui::Ui) {
        if ui.button(super::GROUP_ICON).clicked() {
            if let Some(file) = rfd::FileDialog::default().pick_file() {
                let try_load = || -> anyhow::Result<egui::TextureHandle> {
                    // `image` crate is probably not the choice here. It sweeps
                    // a lot of details under the rug, like colorspaces.
                    let image = image::open(file)?.to_rgba8();
                    let manager = ui.tex_manager();
                    let mut write = manager.write();

                    let size = [image.width() as usize, image.height() as usize];

                    // Create a reference-counted image out of it, refs = 1
                    let texture_id = write.alloc(
                        "Preview brush texture".to_owned(),
                        egui::ImageData::Color(
                            egui::ColorImage {
                                pixels: image
                                    .pixels()
                                    .map(|rgba| {
                                        egui::Color32::from_rgba_unmultiplied(
                                            rgba.0[0], rgba.0[1], rgba.0[2], rgba.0[3],
                                        )
                                    })
                                    .collect(),
                                size,
                                source_size: egui::Vec2 {
                                    x: size[0] as f32,
                                    y: size[1] as f32,
                                },
                            }
                            .into(),
                        ),
                        egui::TextureOptions {
                            magnification: egui::TextureFilter::Nearest,
                            minification: egui::TextureFilter::Linear,
                            wrap_mode: egui::TextureWrapMode::ClampToEdge,
                            mipmap_mode: None,
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
impl Default for TextureStamp {
    fn default() -> Self {
        Self {
            texture: None,
            uv_rect: FULL_UV,
        }
    }
}
#[derive(Default)]
pub struct CircleStamp {
    smoothness: Option<f32>,
}
impl CircleStamp {
    const DEFUALT_SMOOTHNESS: f32 = 1.0;
    fn show(&mut self, ui: &mut egui::Ui) -> () {
        let mut smoothed = self.smoothness.is_some();
        ui.checkbox(&mut smoothed, "Smooth");
        if smoothed {
            self.smoothness = self.smoothness.or(Some(Self::DEFUALT_SMOOTHNESS));
        } else {
            self.smoothness = None;
        }

        let mut dont_care = Self::DEFUALT_SMOOTHNESS;
        let smoothness_mut = self.smoothness.as_mut().unwrap_or(&mut dont_care);
        ui.add_enabled(
            smoothed,
            egui::Slider::new(smoothness_mut, 0.0..=10.0).clamping(egui::SliderClamping::Never),
        );
    }
}

pub struct CreationOutput {
    pub texture_data: Option<Vec<u8>>,
    pub brush: Brush,
}
pub struct CreationModal {
    tab: CreationTab,
    name: String,
    stamp: StampKind,
    test_curves: CurveSetNormalized,
}
impl Default for CreationModal {
    fn default() -> Self {
        Self {
            tab: CreationTab::default(),
            name: "New Brush".to_owned(),
            stamp: StampKind::default(),
            test_curves: CurveSetNormalized::default(),
        }
    }
}
/*
impl super::Modal for CreationModal {
    type Cancel = ();
    type Confirm = CreationOutput;
    type Error = ();
    const NAME: &'static str = "Create Brush";
    fn do_ui(
        &mut self,
        ui: &mut egui::Ui,
    ) -> super::modals::Response<Self::Cancel, Self::Confirm, Self::Error> {
        ui.horizontal(|ui| {
            for tab in <CreationTab as strum::IntoEnumIterator>::iter() {
                // Can't use selectable label here, as it incorrectly checks the
                // fields of the enum for equality too!
                let same_discriminant =
                    std::mem::discriminant(&tab) == std::mem::discriminant(&self.tab);
                if ui
                    .selectable_label(same_discriminant, <&'static str>::from(tab))
                    .clicked()
                {
                    self.tab = tab;
                }
            }
        });
        ui.separator();
        let cancel_response = egui::panel::TopBottomPanel::new(
            egui::panel::TopBottomSide::Bottom,
            egui::Id::new("brush-cancel-panel"),
        )
        .show_inside(ui, |ui| {
            if ui.button("Cancel").clicked_or_escape() {
                super::modals::Response::Cancel(())
            } else {
                super::modals::Response::Continue
            }
        })
        .inner;
        if !matches!(cancel_response, super::modals::Response::Continue) {
            return cancel_response;
        }

        match self.tab {
            CreationTab::Settings => {
                ui.text_edit_singleline(&mut self.name);
            }
            CreationTab::Texture => {
                self.stamp.show(ui);
            }
            CreationTab::Dynamics(mut selected_dynamic, mut selected_source) => {
                egui::SidePanel::new(
                    egui::panel::Side::Left,
                    egui::Id::new("selected_dynamic_panel"),
                )
                .frame(egui::Frame::new().fill(ui.visuals().extreme_bg_color))
                .resizable(false)
                .show_inside(ui, |ui| {
                    for dynamic in <DynamicDestination as strum::IntoEnumIterator>::iter() {
                        ui.selectable_value(
                            &mut selected_dynamic,
                            dynamic,
                            <&'static str>::from(dynamic),
                        );
                    }
                });
                self.test_curves.show(&mut selected_source, ui);
                self.tab = CreationTab::Dynamics(selected_dynamic, selected_source);
            }
        }
        /*
        if let Some(texture) = self.texture.as_ref() {
            let width = ui.available_width();
            let height = width / 3.0;
            let size = egui::vec2(width, height);

            let (response, painter) = ui.allocate_painter(size, egui::Sense::empty());

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
        }*/
        super::modals::Response::Continue
    }
}
    */

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
    _ui: &mut egui::Ui,
    _image: &image::DynamicImage,
    _handle: &mut egui::TextureHandle,
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
    let (mut response, painter) =
        ui.allocate_painter(size, egui::Sense::DRAG | egui::Sense::FOCUSABLE);
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

    painter.add(egui::Shape::Mesh(mesh.into()));

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

    use egui::emath::GuiRounding;
    // Round to px so that tiny gaps don't show up (very obvious lol)
    visual_uv_rect.min = visual_uv_rect
        .min
        .round_to_pixels(painter.pixels_per_point());
    visual_uv_rect.max = visual_uv_rect
        .max
        .round_to_pixels(painter.pixels_per_point());

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
        egui::StrokeKind::Outside,
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
fn resize(_mesh: &mut egui::Mesh, _from: egui::Rect, _to: egui::Rect) {}

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
        let t = f32::from(2 * i) / f32::from(NUM_SAMPLES) - 1.0;

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
    painter.add(egui::Shape::Mesh(
        tessellate(
            egui::TextureId::default(),
            egui::Rect::from_min_size(egui::epaint::WHITE_UV, egui::Vec2::ZERO),
            egui::Color32::BLACK.linear_multiply(0.2),
            rect,
            2.0,
            10.0,
        )
        .into(),
    ));
    painter.rect_stroke(
        rect,
        5.0,
        egui::Stroke {
            color: egui::Color32::BLACK,
            width: 2.0,
        },
        egui::StrokeKind::Outside,
    );
}

pub struct Preloaded {
    id: UniqueID,
    texture: egui::TextureHandle,
    vertices: egui::Mesh,
}

/// Provides a brush selection drawer with many brushes loaded dynamically.
pub struct Bin {}

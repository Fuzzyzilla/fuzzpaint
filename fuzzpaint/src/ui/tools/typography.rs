use fuzzpaint_types::dpi;
use fuzzpaint_types::text;
/// Adapter that allows [`egui_editable_combobox`] to match on any of the names
/// a font has (it's postscript identifier and all of its human-readable names
/// in all languages).
struct MatchableFont<'a>(&'a fontdb::FaceInfo);
impl egui_editable_combobox::ValueOption<String> for MatchableFont<'_> {
    fn filter_by_text(
        &self,
        text: &str,
        _state: egui_editable_combobox::FilterState,
    ) -> egui_editable_combobox::FilterResult {
        use egui_editable_combobox::FilterResult;
        let mut res = FilterResult::from_case_insensitive_substring(&self.0.post_script_name, text);
        for (name, _lang) in &self.0.families {
            if let FilterResult::Exact = res {
                break;
            }
            match FilterResult::from_case_insensitive_substring(name, text) {
                FilterResult::Exact => res = FilterResult::Exact,
                FilterResult::Partial => res = FilterResult::Partial,
                FilterResult::None => (),
            }
        }
        res
    }
    fn display(&self, _text: &str) -> impl egui::IntoAtoms<'_> {
        self.0.post_script_name.as_str()
    }
    fn into_value(self, _text: &str) -> String {
        self.0.post_script_name.clone()
    }
    fn equals_value(&self, value: &String, _text: &str) -> bool {
        &self.0.post_script_name == value
    }
}
pub struct Typography {
    db: fontdb::Database,

    selected_face: fontdb::ID,
    size: dpi::Length<f32>,
    align: text::TextAlign,
    line_spacing_multiplier: f32,
    letter_spacing_multiplier: f32,

    bold: bool,
    italic: bool,
}
impl Typography {
    pub fn new() -> Self {
        let db = {
            let start = std::time::Instant::now();

            let mut db = fontdb::Database::new();
            db.load_system_fonts();

            let end = std::time::Instant::now();
            log::info!("loaded font directory in {}ms", (end - start).as_millis());
            db
        };

        Self {
            db,
            // Gets converted to a more semantic default during UI loop.
            selected_face: fontdb::ID::dummy(),

            size: dpi::Length::physical(12.0, dpi::PhysicalUnit::Point),
            align: text::TextAlign::Left,

            line_spacing_multiplier: 1.0,
            letter_spacing_multiplier: 1.0,

            bold: false,
            italic: false,
        }
    }
    /// Get the ID of the default face, or Dummy if no faces.
    pub fn default_face_id(&self) -> fontdb::ID {
        // Use the default sans serif face.
        self.db
            .query(&fontdb::Query {
                families: &[fontdb::Family::SansSerif],
                weight: fontdb::Weight::NORMAL,
                stretch: fontdb::Stretch::Normal,
                style: fontdb::Style::Normal,
            })
            // Otherwise, use the (completely arbitrary) first face.
            .unwrap_or_else(|| self.db.faces().next().map_or(fontdb::ID::dummy(), |f| f.id))
    }
    pub fn show(&mut self, ui: &mut egui::Ui) {
        let viewport = ui.max_rect();
        egui::Window::new("text-settings")
            .title_bar(false)
            .frame(egui::Frame {
                fill: ui.style().visuals.extreme_bg_color,
                ..egui::Frame::window(ui.style())
            })
            .fade_in(true)
            .constrain_to(ui.max_rect())
            .anchor(egui::Align2::CENTER_TOP, [0.0, 16.0])
            .default_size([0.0, 0.0])
            .resizable(false)
            .show(ui, |ui| {
                // Horizontal-fill, with wrap if too large to fit. Normal
                // horizontal_wrapped doesn't re-expand after the space gets
                // larger.
                ui.allocate_ui_with_layout(
                    // Bias to smaller. For some reason, the wrapping only
                    // occurs much after it exhausts all the space!
                    egui::Vec2::new((viewport.width() - 200.0).max(0.0), 0.0),
                    egui::Layout::left_to_right(egui::Align::Center).with_main_wrap(true),
                    |ui| {
                        self.internal_show(ui);
                    },
                );
            });
    }
    fn internal_show(&mut self, ui: &mut egui::Ui) {
        use material_icons::Icon;
        let default_face_id = self.default_face_id();

        let icon_widget_text = |text: Icon| -> egui::WidgetText {
            egui::WidgetText::RichText(
                egui::RichText::new(text.to_string())
                    .font(egui::FontId::new(
                        16.0,
                        crate::ui::GOOGLE_MATERIAL_ICONS_FAMILY.clone(),
                    ))
                    .into(),
            )
        };

        let selected_face_info = if let Some(face) = self.db.face(self.selected_face) {
            Some(face)
        } else {
            self.selected_face = default_face_id;
            // Try again (may still be none, if empty.)
            self.db.face(self.selected_face)
        };
        {
            let mut readable_name = selected_face_info.map_or("[empty]".to_owned(), |face| {
                face.families.first().map_or_else(
                    || face.post_script_name.clone(),
                    |(name, _lang)| name.clone(),
                )
            });

            // To check if the name jumps to a new one (where we
            // then update the id).
            let orig_name = readable_name.clone();

            ui.style_mut().spacing.text_edit_width *= 0.5;
            let response = egui_editable_combobox::EditableComboBox::new(ui.id().with("font"))
                .show(ui, &mut readable_name, self.db.faces().map(MatchableFont))
                // Show the full postscript name
                .on_hover_text(selected_face_info.map_or("[empty]", |face| &face.post_script_name));

            if response.changed() && readable_name != orig_name {
                // Find the face with this postscript name, or
                // set to dummy if it doesn't exist.
                self.selected_face = self
                    .db
                    .faces()
                    .find(|f| f.post_script_name == readable_name)
                    .map_or(
                        // Next iter will fix this.
                        fontdb::ID::dummy(),
                        |f| f.id,
                    );
            }

            if ui
                .add(egui::Button::new(crate::ui::PLUS_ICON.to_string()).frame_when_inactive(false))
                .on_hover_text("Import fonts...")
                .clicked()
                // Import files or scan whole directories would be nice...
                && let Some(files) = rfd::FileDialog::new()
                    .add_filter("Font files (ttf, ttc, otf, otc, tte, dfont)", &["ttf", "ttc", "otf", "otc", "tte", "dfont"])
                    .add_filter("All files", &[String::new()])
                    .set_can_create_directories(false)
                    .set_title("Import fonts")
                    .pick_files()
            {
                for file in files {
                    if let Err(e) = self.db.load_font_file(&file) {
                        log::error!("failed to open {}: {e}", file.display());
                    }
                }
            }
        }

        ui.separator();

        ui.horizontal(|ui| {
            ui.add(egui::Label::new(icon_widget_text(Icon::FormatSize)).selectable(false))
                .on_hover_text("Font size");
            ui.add(
                egui::DragValue::new(self.size.value_mut())
                    .clamp_existing_to_range(true)
                    .range(0.1..=2048.0),
            );
            egui::ComboBox::from_id_salt(ui.id().with("size-unit"))
                .width(ui.style().spacing.combo_width / 4.0)
                .selected_text(self.size.unit().to_string())
                .show_ui(ui, |_| ());
        });

        ui.separator();

        ui.horizontal(|ui| {
            for (icon, tooltip, enabled) in [
                (Icon::FormatBold, "Bold", &mut self.bold),
                (Icon::FormatItalic, "Italic", &mut self.italic),
                // Do we want this?
                // (Icon::FormatUnderlined, "Underline"),
            ] {
                if ui
                    .add(
                        egui::Button::selectable(*enabled, icon_widget_text(icon))
                            .frame_when_inactive(*enabled),
                    )
                    .on_hover_text(tooltip)
                    .clicked()
                {
                    *enabled = !*enabled;
                }
            }
        });

        ui.separator();

        ui.horizontal(|ui| {
            for (icon, value) in [
                (Icon::FormatAlignLeft, text::TextAlign::Left),
                (Icon::FormatAlignCenter, text::TextAlign::Center),
                (Icon::FormatAlignRight, text::TextAlign::Right),
                (Icon::FormatAlignJustify, text::TextAlign::Justify),
            ] {
                if ui
                    .add(
                        egui::Button::selectable(self.align == value, icon_widget_text(icon))
                            // Show frame when selected.
                            .frame_when_inactive(self.align == value),
                    )
                    .on_hover_text(value.to_string())
                    .clicked()
                {
                    self.align = value;
                }
            }
        });

        ui.separator();

        ui.horizontal(|ui| {
            let mut is_first = true;
            for (icon, tooltip, value) in [
                (
                    // Material icons has a more semantically
                    // correct one, `FormatLineSpacing`, but we
                    // use this to match the missing character
                    // spacing icon.
                    Icon::VerticalDistribute,
                    "Line spacing",
                    &mut self.line_spacing_multiplier,
                ),
                (
                    // Material symbols has a more semantically
                    // currect one, icons doesn't.
                    Icon::HorizontalDistribute,
                    "Letter spacing",
                    &mut self.letter_spacing_multiplier,
                ),
            ] {
                if !is_first {
                    ui.separator();
                }
                is_first = false;
                ui.add(egui::Label::new(icon_widget_text(icon)).selectable(false))
                    .on_hover_text(tooltip);
                ui.add(
                    egui::DragValue::new(value)
                        .custom_formatter(|v, _| format!("{:.1}%", v * 100.0))
                        .speed(0.01)
                        .clamp_existing_to_range(false)
                        // This is a multiplier, not a percent!
                        // Negative or zero spacing is
                        // well-defined and useful. I prombis.
                        // :3
                        .range(-10.0..=10.0),
                );
            }
        });
    }
}

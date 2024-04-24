use egui::Color32;
use fuzzpaint_core::{color::Color as FColor, util::FiniteF32};

const GROW_FACTOR: f32 = 1.5;

/// If the contrast between two colors is too low, choose a stroke color to contrast both.
fn contrasting_stroke(
    background: impl Into<egui::Rgba>,
    foreground: impl Into<egui::Rgba>,
    contrast_target: f32,
) -> Option<egui::Rgba> {
    // Hella scuffed implementation. :V
    let background: egui::Rgba = background.into();
    let foreground: egui::Rgba = foreground.into();

    let back_intensity = background.intensity();
    let fore_intensity = foreground.intensity();

    let diff = back_intensity - fore_intensity;
    // difference vector from fore -> back
    let diff_dir = diff.signum();
    let abs_diff = (back_intensity - fore_intensity).abs();

    let intensity = if abs_diff < contrast_target {
        // Not enough contrast!
        // Try to choose a color which contrasts with *both* fore and background

        // We will try several times, in order:
        // contrast from fg away from bg, if in 0..1 range.
        // contrast from bg away from fg, if in 0..1 range.
        // negative fg
        // negative bg
        // Fallback on clamped negative fg

        let intensities = [
            fore_intensity - diff_dir * contrast_target,
            back_intensity + diff_dir * contrast_target,
            1.0 - fore_intensity,
            1.0 - back_intensity,
        ];

        // Choose first that is a valid color.
        // Otherwise, force choose the opposite of foreground.
        Some(
            intensities
                .into_iter()
                .find(|i| (0.0..=1.0f32).contains(i))
                .unwrap_or_else(|| (1.0 - fore_intensity).clamp(0.0, 1.0)),
        )
    } else {
        // Not needed
        None
    };

    // HACK. this ctor is not *perceptual* intensity, and it shows.
    // In light mode it's practically invisible. So, we square the intensity...
    intensity.map(|i| egui::Rgba::from_luminance_alpha(i * i, 1.0))
}
fn grayscale_contrasting(
    foreground: impl Into<egui::Rgba>,
    background: impl Into<egui::Rgba>,
) -> egui::Color32 {
    // Hella scuffed implementation. :V
    let color: egui::Rgba = foreground.into();
    let background: egui::Rgba = background.into();
    let color = background.multiply(1.0 - color.a()) + color;

    let intensity = color.intensity();
    if intensity > 0.5 {
        egui::Color32::BLACK
    } else {
        egui::Color32::WHITE
    }
}

/// Square button-like element with a solid color that does... nothing except report clicks :P
#[derive(Copy, Clone)]
pub struct ColorSquare {
    pub color: FColor,
    pub size: f32,
    pub selected: bool,
    pub palette_idx: Option<fuzzpaint_core::color::PaletteIndex>,
}
impl egui::Widget for ColorSquare {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let enabled = ui.is_enabled();
        let (rect, this) = ui.allocate_exact_size(
            egui::Vec2::splat(self.size),
            egui::Sense {
                click: enabled,
                drag: false,
                focusable: enabled,
            },
        );
        this.widget_info(|| egui::WidgetInfo {
            typ: egui::WidgetType::Button,
            enabled,
            // Todo: text description of color
            label: None,
            current_text_value: None,
            prev_text_value: None,
            selected: None,
            value: None,
            text_selection: None,
        });

        // false if all in the normal range of colors
        let out_of_gammut = self
            .color
            .as_slice()
            .iter()
            .any(|&v| !(FiniteF32::ZERO..=FiniteF32::ONE).contains(&v));
        let color_arr = self.color.as_array();
        let color = egui::Rgba::from_rgba_premultiplied(
            color_arr[0],
            color_arr[1],
            color_arr[2],
            color_arr[3],
        );
        // Red border if out-of-gamut. Negative border if hovered. Default contrasting border
        let stroke_color = match (out_of_gammut, this.hovered()) {
            (true, _) => Color32::DARK_RED,
            // Use maximally-contrasty color (not correct impl lol)
            (false, true) => egui::Rgba::from_gray(1.0 - color.intensity()).into(),
            (false, false) => {
                const MIN_CONTRAST: f32 = 0.3;
                // Check contrast against assumed (FIXME) `canvas` style background, outline with
                // contrast-y color if not high enough.
                contrasting_stroke(ui.style().visuals.extreme_bg_color, color, MIN_CONTRAST)
                    .map_or(ui.style().visuals.extreme_bg_color, Into::into)
            }
        };
        // Increase in size and show ontop when hovered.
        let (expansion, layer) = if this.hovered() || self.selected {
            // Additive expansion, not multiplicative - subtract one
            (
                self.size * (GROW_FACTOR - 1.0),
                egui::LayerId {
                    // Wont work when this button is added to foreground already :V
                    order: egui::Order::Foreground,
                    ..ui.layer_id()
                },
            )
        } else {
            // No grow, same layer
            (0.0, ui.layer_id())
        };
        let expansion =
            ui.ctx()
                .animate_value_with_time(this.id, expansion, ui.style().animation_time);
        let painter = ui.painter().clone().with_layer_id(layer);
        painter.rect(
            rect.expand(expansion),
            0.0,
            color,
            egui::Stroke {
                color: stroke_color,
                width: 1.0,
            },
        );

        this
    }
}

// Where palette memory resides. Local uses the local widget ID to store data,
// `Global` will share history between all palettes set to it.
pub enum HistoryScope {
    Local,
    Global,
}

#[derive(Clone, Default)]
#[repr(transparent)]
struct ColorPaletteQueue(
    // front = new, back = old
    std::collections::VecDeque<FColor>,
);
impl ColorPaletteQueue {
    /// Move or insert the color to the most recent.
    fn hoist(&mut self, color: FColor) {
        self.remove(color);
        // Add as most recent
        self.0.push_front(color);
    }
    fn max_len(&mut self, max: usize) {
        if self.0.len() > max {
            let _ = self.0.drain(max..);
        }
    }
    fn remove(&mut self, color: FColor) {
        // Delete this color from mem
        // We care not for the numeric value of these floats!
        self.0
            .retain(|c| bytemuck::bytes_of(c) != bytemuck::bytes_of(&color));
    }
}
// Thin wrapper, inherit iters and such.
impl std::ops::Deref for ColorPaletteQueue {
    type Target = std::collections::VecDeque<FColor>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(Clone, Default)]
struct ColorPaletteState {
    pinned: ColorPaletteQueue,
    history: ColorPaletteQueue,
}
impl ColorPaletteState {
    fn hoist(&mut self, color: FColor) {
        // We do *not* reorder pinned if it already contains!
        if self.pinned.contains(&color) {
            return;
        }
        self.history.hoist(color);
    }
    fn pin(&mut self, color: FColor) {
        self.history.remove(color);
        self.pinned.hoist(color);
    }
    fn unpin(&mut self, color: FColor) {
        self.pinned.remove(color);
        self.history.hoist(color);
    }
}

/// Widget that remembers color history, displaying a compact grid of previous selected colors
/// that can be re-applied.
pub struct ColorPalette<'color> {
    color: &'color mut FColor,
    max_history: Option<usize>,
    history_scope: HistoryScope,
    in_flux: bool,
    id: Option<egui::Id>,
}
impl<'color> ColorPalette<'color> {
    #[must_use]
    pub fn new(color: &'color mut FColor) -> Self {
        Self {
            color,
            max_history: None,
            history_scope: HistoryScope::Local,
            in_flux: false,
            id: None,
        }
    }
    /// While this is set, the color will not be added to the palette
    /// until `in_flux` becomes false.
    #[must_use]
    pub fn in_flux(self, in_flux: bool) -> Self {
        Self { in_flux, ..self }
    }
    #[must_use]
    pub fn max_history(self, max: usize) -> Self {
        Self {
            max_history: Some(max),
            ..self
        }
    }
    #[must_use]
    pub fn scope(self, scope: HistoryScope) -> Self {
        Self {
            history_scope: scope,
            ..self
        }
    }
    /// When paired with `HistoryScope::Local`, provides the ID to use for memory storage.
    #[must_use]
    pub fn id_source(self, id: impl std::hash::Hash) -> Self {
        Self {
            id: Some(egui::Id::new(id)),
            ..self
        }
    }
}
impl egui::Widget for ColorPalette<'_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        const BTN_BASE_SIZE: f32 = 12.0;
        let width = ui.available_width();
        egui::Frame::canvas(ui.style())
            .shadow(egui::epaint::Shadow::NONE)
            .stroke(egui::Stroke::NONE)
            .rounding(0.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    // Expand to width, dynamic height
                    .auto_shrink([false, true])
                    // Prevent from getting too tall
                    .max_height(width / 2.0)
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                    )
                    // show_viewport can be used to improve perf
                    // We don't have *that* large of history tho, WONTFIX for now
                    .show(ui, |ui| {
                        let palette_id = match (self.history_scope, self.id) {
                            (HistoryScope::Global, _) => egui::Id::NULL,
                            (HistoryScope::Local, None) => {
                                // Get our own ID, not stable but it wasn't requested to be.
                                ui.id()
                            }
                            (HistoryScope::Local, Some(id)) => id,
                        };
                        let palette = ui.data_mut(|mem| {
                            let palette =
                                mem.get_temp_mut_or_default::<ColorPaletteState>(palette_id);
                            if !self.in_flux {
                                palette.hoist(*self.color);
                            }
                            if let Some(max) = self.max_history {
                                palette.history.max_len(max);
                            }
                            palette.clone()
                        });
                        let size = super::justify_mut(
                            ui,
                            super::JustifyAxis::Horizontal,
                            BTN_BASE_SIZE,
                            2.0,
                        );
                        ui.style_mut().spacing.interact_size.y = size;

                        // Both are showing, add an hrule to separate
                        let needs_separator =
                            !palette.history.is_empty() && !palette.pinned.is_empty();

                        let mut new_unpins = smallvec::SmallVec::<[FColor; 1]>::new();
                        if !palette.pinned.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                for color in palette.pinned.0 {
                                    let color_btn = ColorSquare {
                                        color,
                                        size,
                                        selected: false,
                                        palette_idx: None,
                                    };
                                    let response = ui.add(color_btn);
                                    response.context_menu(|ui| {
                                        if ui.small_button("Unpin").clicked() {
                                            new_unpins.push(color);
                                            ui.close_menu();
                                        }
                                    });
                                    if response.clicked() {
                                        *self.color = color_btn.color;
                                    }
                                }
                            });
                        }
                        if needs_separator {
                            ui.separator();
                        }
                        let mut new_pins = smallvec::SmallVec::<[FColor; 1]>::new();
                        if !palette.history.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                for color in palette.history.0 {
                                    let color_btn = ColorSquare {
                                        color,
                                        size,
                                        selected: false,
                                        palette_idx: None,
                                    };
                                    let response = ui.add(color_btn);
                                    response.context_menu(|ui| {
                                        if ui.small_button("Pin").clicked() {
                                            new_pins.push(color);
                                            ui.close_menu();
                                        }
                                    });
                                    if response.clicked() {
                                        *self.color = color_btn.color;
                                    }
                                }
                            });
                        }

                        // Update palette pins
                        if !new_unpins.is_empty() || !new_pins.is_empty() {
                            ui.data_mut(|mem| {
                                let palette =
                                    mem.get_temp_mut_or_default::<ColorPaletteState>(palette_id);

                                for unpin in new_unpins {
                                    palette.unpin(unpin);
                                }
                                for pin in new_pins {
                                    palette.pin(pin);
                                }
                            });
                        }
                    });
            })
            .response
            .on_hover_cursor(egui::CursorIcon::Crosshair)
    }
}

use egui::Color32;
use either::Either;
use fuzzpaint_core::{
    color::{Color as FColor, ColorOrPalette, PaletteIndex},
    util::FiniteF32,
};

const GROW_FACTOR: f32 = 1.25;

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
    let contrasting = (intensity + 0.7) % 1.0;
    egui::Color32::from_gray((contrasting * 255.999) as u8)
}

/// Square button-like element with a solid color that does nothing except report interactions and grow on hover.
#[derive(Copy, Clone)]
pub struct ColorSquare {
    pub color: FColor,
    /// Display as selected, larger and highlighted. Also triggered on hover and focus.
    pub selected: bool,
    pub icon: Option<char>,
}
impl Default for ColorSquare {
    fn default() -> Self {
        Self {
            color: FColor::TRANSPARENT,
            selected: false,
            icon: None,
        }
    }
}
impl egui::Widget for ColorSquare {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let enabled = ui.is_enabled();
        let size = ui.style().spacing.interact_size.min_elem();
        let (rect, this) = ui.allocate_exact_size(
            egui::Vec2::splat(size),
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
        let selected = this.hovered() || self.selected || this.has_focus();
        // Red border if out-of-gamut. Negative border if hovered. Default contrasting border
        let stroke_color = match (out_of_gammut, selected) {
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
        let (expansion, layer) = if selected {
            // Additive expansion, not multiplicative - subtract one
            (
                size * (GROW_FACTOR - 1.0),
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
        if let Some(icon) = self.icon {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                icon,
                egui::FontId::default(),
                grayscale_contrasting(color, ui.style().visuals.extreme_bg_color),
            );
        }

        this
    }
}

/// An icon of identical layout to [`ColorSquare`] that provides a simple icon.
pub struct IconSquare {
    icon: char,
}
impl egui::Widget for IconSquare {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let enabled = ui.is_enabled();
        let size = ui.style().spacing.interact_size.min_elem();
        let (rect, this) = ui.allocate_exact_size(
            egui::Vec2::splat(size),
            egui::Sense {
                click: enabled,
                drag: false,
                focusable: false,
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

        let painter = ui.painter();
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            self.icon,
            egui::FontId::default(),
            ui.style().visuals.text_color(),
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
    std::collections::VecDeque<ColorOrPalette>,
);
impl ColorPaletteQueue {
    /// Move or insert the color to the most recent.
    fn hoist(&mut self, color: ColorOrPalette) {
        self.remove(color);
        // Add as most recent
        self.0.push_front(color);
    }
    fn max_len(&mut self, max: usize) {
        if self.0.len() > max {
            let _ = self.0.drain(max..);
        }
    }
    fn remove(&mut self, color: ColorOrPalette) {
        // Delete this color from mem
        // We care not for the numeric value of these floats!
        self.0
            .retain(|c| bytemuck::bytes_of(c) != bytemuck::bytes_of(&color));
    }
}
// Thin wrapper, inherit iters and such.
impl std::ops::Deref for ColorPaletteQueue {
    type Target = std::collections::VecDeque<ColorOrPalette>;
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
    fn hoist(&mut self, color: ColorOrPalette) {
        self.history.hoist(color);
        // We do *not* reorder pinned!
    }
    fn pin(&mut self, color: ColorOrPalette) {
        self.history.hoist(color);
        self.pinned.hoist(color);
    }
    fn unpin(&mut self, color: ColorOrPalette) {
        self.history.hoist(color);
        self.pinned.remove(color);
    }
}

/// Widget that remembers color history, displaying a compact grid of previous selected colors
/// that can be re-applied.
pub struct ColorPalette<'a, Writer> {
    color: &'a mut ColorOrPalette,
    palette: &'a mut fuzzpaint_core::state::palette::writer::Writer<'a, Writer>,
    max_history: Option<usize>,
    history_scope: HistoryScope,
    in_flux: bool,
    /// Swap top two colors in the history.
    do_swap: bool,
    id: Option<egui::Id>,
}
impl<'a, Writer> ColorPalette<'a, Writer> {
    #[must_use]
    pub fn new(
        color: &'a mut ColorOrPalette,
        palette: &'a mut fuzzpaint_core::state::palette::writer::Writer<'a, Writer>,
    ) -> Self {
        Self {
            color,
            palette,
            max_history: None,
            history_scope: HistoryScope::Local,
            in_flux: false,
            do_swap: false,
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
    /// Swap the first and second most recent colors.
    #[must_use]
    pub fn swap(self, swap: bool) -> Self {
        Self {
            do_swap: swap,
            ..self
        }
    }
}
// Ewwwww.. Traits that make the undo/redo system tick, that usually need not be seen by mortal eyes, but alas here we are...
impl<
        Writer: fuzzpaint_core::queue::writer::CommandWrite<
            fuzzpaint_core::state::palette::commands::Command,
        >,
    > egui::Widget for ColorPalette<'_, Writer>
{
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let mut changed = false;
        const BTN_BASE_SIZE: f32 = 12.0;
        let width = ui.available_width();

        let mut response = egui::Frame::canvas(ui.style())
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
                                // Pull the active color up in the list.
                                // Funny syntax!
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

                        let mut new_pins = smallvec::SmallVec::<[ColorOrPalette; 1]>::new();
                        ui.horizontal_wrapped(|ui| {
                            ui.add(IconSquare {
                                icon: super::PALETTE_ICON
                            }).on_hover_text("Paletted colors")
                            .on_hover_text("Colors used from here allow you to modify the color definition after-the-fact and have these changes reflected everywhere it is used.");
                            // Current color, with indexed color collapsed into a real color.
                            let mut deref_color = self
                                .color
                                .get()
                                .left_or_else(|idx| self.palette.get(idx).unwrap_or(FColor::BLACK));

                            // if a palette write has been issued.
                            let mut replace_color = None;

                            for (idx, &color) in self.palette.iter() {
                                // Show each palette color, if clicked then select it.
                                let selected = self
                                    .color
                                    .get()
                                    .right()
                                    .is_some_and(|selected| selected == idx);

                                let color_square = ui.add(ColorSquare {
                                    color,
                                    icon: None,
                                    selected,
                                });

                                color_square.context_menu(|ui| {
                                    ui.label(format!("{idx:?}"));
                                    ui.separator();
                                    if ui.small_button("Pin").clicked() {
                                        new_pins.push(idx.into());
                                        ui.close_menu();
                                    }
                                    if ui.small_button("Replace contents with active color").clicked() {
                                        replace_color = Some((idx, deref_color));
                                        ui.close_menu();
                                    }
                                });

                                if color_square.clicked() {
                                    *self.color = idx.into();
                                    changed = true;
                                    deref_color = self.palette.get(idx).unwrap_or(deref_color);
                                }
                            }
                            // "Add" button at the end, with a plus symbol
                            if ui
                                .add(ColorSquare {
                                    color: deref_color,
                                    icon: Some(super::PLUS_ICON),
                                    ..Default::default()
                                })
                                .on_hover_text("Create a paletted color from active color")
                                .clicked()
                            {
                                *self.color = self.palette.insert(deref_color).into();
                            };

                            // Issue write, if requested.
                            if let Some((idx, color)) = replace_color {
                                let _ = self.palette.set(idx, color);
                            }
                        });
                        ui.separator();

                        // If both recent and pinned colors are shown similtaneously, add an hrule to separate
                        let needs_separator =
                            !palette.history.is_empty() && !palette.pinned.is_empty();
                        let mut new_unpins = smallvec::SmallVec::<[ColorOrPalette; 1]>::new();
                        if !palette.pinned.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                ui.add(IconSquare {
                                    icon: super::PIN_ICON
                                }).on_hover_text("Pinned colors").on_hover_text("Pin colors from your history to keep them here.");
                                for &color in &palette.pinned.0 {
                                    let is_paletted = color.is_palette();
                                    let color_btn = ColorSquare {
                                        color: color.get().left_or_else(|idx| {
                                            self.palette.get(idx).unwrap_or(FColor::TRANSPARENT)
                                        }),
                                        icon: is_paletted.then_some(super::PALETTE_ICON),
                                        // I have experimented with setting `selected` here based on whether it's current, but no
                                        // combination of logic makes it *feel* right. Always just looks like the selection is jumping around wildly x3
                                        ..Default::default()
                                    };
                                    let response = ui.add(color_btn);
                                    response.context_menu(|ui| {
                                        if ui.small_button("Unpin").clicked() {
                                            new_unpins.push(color);
                                            ui.close_menu();
                                        }
                                    });
                                    if response.clicked() {
                                        changed = true;
                                        *self.color = color;
                                    }
                                }
                            });
                        }
                        if needs_separator {
                            ui.separator();
                        }
                        if !palette.history.is_empty() {
                            ui.horizontal_wrapped(|ui| {

                            ui.add(IconSquare {
                                icon: super::HISTORY_ICON
                            }).on_hover_text("Recent colors");
                                for &color in &palette.history.0 {
                                    let is_paletted = color.is_palette();
                                    let color_btn = ColorSquare {
                                        color: color.get().left_or_else(|idx| {
                                            self.palette.get(idx).unwrap_or(FColor::TRANSPARENT)
                                        }),
                                        // Show if it's paletted or pinned.
                                        icon: is_paletted.then_some(super::PALETTE_ICON).or_else(|| palette.pinned.contains(&color).then_some(super::PIN_ICON)),
                                        ..Default::default()
                                    };
                                    let response = ui.add(color_btn);
                                    response.context_menu(|ui| {
                                        if ui.small_button("Pin").clicked() {
                                            new_pins.push(color);
                                            ui.close_menu();
                                        }
                                    });
                                    if response.clicked() {
                                        changed = true;
                                        *self.color = color;
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
                        // Perform swap
                        if self.do_swap {
                            ui.data_mut(|mem| {
                                let palette =
                                    mem.get_temp_mut_or_default::<ColorPaletteState>(palette_id);
                                let Some(&swap_with) = palette.history.get(1) else {
                                    // History too smol to swap :3
                                    return;
                                };
                                changed = true;
                                palette.hoist(swap_with);
                                *self.color = swap_with;
                        });
                        }
                    });
            })
            .response
            .on_hover_cursor(egui::CursorIcon::Crosshair);

        if changed {
            response.mark_changed()
        }

        response
    }
}

pub struct PickerResponse {
    // The color is actively being changed.
    pub in_flux: bool,
    pub response: egui::Response,
}

pub fn picker_dock(ctx: &egui::Context, hsva: &mut egui::ecolor::HsvaGamma) -> PickerResponse {
    egui::containers::Area::new("color-picker")
        .anchor(egui::Align2::LEFT_TOP, [0.0f32; 2])
        .show(ctx, |ui| {
            /// number of edge "rays" from the center of the picker arc to the edge.
            const MAIN_RAYS: usize = 16;
            /// Number of steps along each ray from the center to the edge.
            /// Step 0 is special in that it's a triangular section while the others are quads.
            const MAIN_STEPS: usize = 8;
            /// Number of segments of the hue arc. Should be a multiple of six to hit all hue peaks!
            const HUE_STEPS: usize = 24;
            /// Spacing between main SV picker and H picker
            const RADIAL_MARGIN: f32 = 10.0;
            /// Witdth of the H picker
            const HUE_WIDTH: f32 = 15.0;
            /// When the circle is small, shift it out of it's corner.
            const CONTRACTED_OFFSET: f32 = 10.0;
            const EXPANDED_RADIUS: f32 = 150.0;
            const PICKER_SIZE: f32 = 10.0;
            const CONTRACTED_RADIUS: f32 = 20.0;
            const STEP_DIST: f32 = 1.0 / MAIN_STEPS as f32;

            #[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
            enum ClickTarget {
                #[default]
                Hover,
                SaturationValue,
                Hue,
            };

            // Remember if we were expanded last frame, make the interact area much larger.
            let click_marker = ui.id().with("click");
            let last_target = ui.memory(|w| w.data.get_temp::<ClickTarget>(click_marker));

            let (rect, mut response) = ui.allocate_exact_size(
                // Be a small or large area depending on whether it's been opened yet.
                if last_target.is_some() {
                    [EXPANDED_RADIUS + RADIAL_MARGIN + HUE_WIDTH; 2].into()
                } else {
                    [CONTRACTED_OFFSET + CONTRACTED_RADIUS; 2].into()
                },
                egui::Sense::click_and_drag(),
            );

            let expanded_proportion = ctx.animate_bool(response.id, last_target.is_some());
            let not_closed = expanded_proportion > 0.0;
            let radius = egui::lerp(CONTRACTED_RADIUS..=EXPANDED_RADIUS, expanded_proportion);
            let origin_offset = egui::lerp(CONTRACTED_OFFSET..=0.0, expanded_proportion);
            let origin = rect.left_top() + [origin_offset; 2].into();

            let mesh_origin = rect.left_top();
            let mesh_radius = radius + origin_offset;

            // Store expand state for next frame.
            let click_target = if response.is_pointer_button_down_on() {
                // Start or continue interaction...
                match last_target {
                    Some(ClickTarget::Hover) | None => {
                        // A new interaction. Find out which type...
                        // Unwrap ok, we know are sure it's interacted.
                        let delta = response.interact_pointer_pos().unwrap() - origin;
                        if delta.length_sq() < EXPANDED_RADIUS * EXPANDED_RADIUS {
                            Some(ClickTarget::SaturationValue)
                        } else {
                            Some(ClickTarget::Hue)
                        }
                    }
                    // Interaction ongoing, continue it.
                    other => other,
                }
            } else if response.hovered() {
                Some(ClickTarget::Hover)
            } else {
                None
            };

            if click_target != last_target {
                ui.memory_mut(|w| {
                    if let Some(click_target) = click_target {
                        w.data
                            .insert_temp::<ClickTarget>(click_marker, click_target);
                    } else {
                        w.data.remove_temp::<ClickTarget>(click_marker);
                    }
                });
            }

            let painter = ui.painter();

            // Default mesh uses "no" texture, just vertex colors. (well, a white texture, same difference).
            let mesh = not_closed.then(|| {
                let mut mesh = egui::Mesh::default();

                //======================================= Main arc =======================================
                // Build the main body of the picker, a quarter circle which displays
                // lightness on it's radius and saturation on it's circumference.

                // Fixme: reservations too small by a tiny and baffling amount, making it substantially less performant lol.
                // First step has a single shared vertex for all. Every ray then has STEPS more verts on it.
                mesh.reserve_vertices(1 + MAIN_RAYS * MAIN_STEPS);
                // Inner circle (Step 0-1) has RAYS - 1 tris. All further step regions have 2 per ray pair.
                mesh.reserve_triangles((MAIN_RAYS - 1) + (MAIN_RAYS - 1) * (MAIN_STEPS - 1) * 2);

                // Origin point shared by all rays! Black under any hue.
                mesh.colored_vertex(mesh_origin, egui::Color32::BLACK);

                // Create vertices.
                for ray in 0..=MAIN_RAYS {
                    // [0, 1], how bright is this?
                    let saturation = ray as f32 / MAIN_RAYS as f32;
                    // Quarter turn, from x axis to y axis, across the rays.
                    let ray_dir = egui::Vec2::angled(saturation * std::f32::consts::FRAC_PI_2);
                    // Skip first step, it's the origin point we pushed first
                    for step in 1..=MAIN_STEPS {
                        // [0, 1], how saturated is this?
                        let value = step as f32 / MAIN_STEPS as f32;
                        mesh.colored_vertex(
                            mesh_origin + mesh_radius * ray_dir * (step as f32 * STEP_DIST),
                            egui::ecolor::HsvaGamma {
                                h: hsva.h,
                                // Sqrt accounts for perceptual nonlinearity due to area growing with square of radius,
                                // leading to most of the area being muddy grays. With my understanding, this means the
                                // *lightness* should be sqrt-ed, but this looks more correct to me. Weird!
                                s: (1.0 - saturation).sqrt(),
                                v: value,
                                a: 1.0,
                            }
                            .into(),
                        );
                    }
                }
                // Create triangles.
                for ray in 0..(MAIN_RAYS as u32) {
                    const VERTICES_PER_RAY: u32 = MAIN_STEPS as u32;

                    let next_ray = ray + 1;
                    // Inner circle, shared by all rays, filled with a triangle fan.
                    // Zero is the origin, need to math for the other two.
                    // Draw between the origin, and first steps of two adjactent rays, to make triangle fan.
                    mesh.add_triangle(
                        0,
                        1 + ray * VERTICES_PER_RAY,
                        1 + next_ray * VERTICES_PER_RAY,
                    );

                    // Outer sectors, space between the rays is filled by a quad.
                    for step in 0..(MAIN_STEPS as u32 - 1) {
                        let next_step = step + 1;

                        let a = 1 + ray * VERTICES_PER_RAY + step;
                        let b = 1 + ray * VERTICES_PER_RAY + next_step;
                        let c = 1 + next_ray * VERTICES_PER_RAY + step;
                        let d = 1 + next_ray * VERTICES_PER_RAY + next_step;

                        mesh.add_triangle(a, b, c);
                        mesh.add_triangle(b, c, d);
                    }
                }

                // =================================== Hue wheel =============================================
                let hue_base_vertex = mesh.vertices.len() as u32;
                for hue in 0..=HUE_STEPS {
                    let hue = hue as f32 / HUE_STEPS as f32;
                    let quarter_angle = hue * std::f32::consts::FRAC_PI_2;

                    let ray_dir = egui::Vec2::angled(quarter_angle);

                    let color = egui::ecolor::Hsva::new(hue, 1.0, 1.0, 1.0).into();

                    mesh.colored_vertex(
                        mesh_origin + ray_dir * (mesh_radius + RADIAL_MARGIN),
                        color,
                    );
                    mesh.colored_vertex(
                        mesh_origin + ray_dir * (mesh_radius + RADIAL_MARGIN + HUE_WIDTH),
                        color,
                    );
                }
                for hue in 0..(HUE_STEPS as u32) {
                    const VERTICES_PER_HUE: u32 = 2;
                    let next_hue = hue + 1;

                    let a = hue_base_vertex + hue * VERTICES_PER_HUE;
                    let b = hue_base_vertex + hue * VERTICES_PER_HUE + 1;
                    let c = hue_base_vertex + next_hue * VERTICES_PER_HUE;
                    let d = hue_base_vertex + next_hue * VERTICES_PER_HUE + 1;

                    mesh.add_triangle(a, b, c);
                    mesh.add_triangle(b, c, d);
                }
                mesh
            });
            let mut sv_radius = hsva.v * radius;
            let mut sv_angle = (1.0 - hsva.s.powi(2)) * std::f32::consts::FRAC_PI_2;
            let mut hue_angle = hsva.h * std::f32::consts::FRAC_PI_2;

            if let Some(click_point) = response.interact_pointer_pos() {
                match click_target {
                    Some(ClickTarget::Hover) | None => (),
                    Some(ClickTarget::SaturationValue) => {
                        response.mark_changed();
                        let mut delta = click_point - origin;
                        delta = delta.max(egui::Vec2::ZERO);

                        sv_radius = delta.length().min(radius);
                        sv_angle = delta.angle();

                        let selection_angle = sv_angle / std::f32::consts::FRAC_PI_2;

                        let selection_radius = sv_radius / radius;

                        *hsva = egui::ecolor::HsvaGamma {
                            s: (1.0 - selection_angle).sqrt(),
                            v: selection_radius,
                            ..*hsva
                        }
                    }
                    Some(ClickTarget::Hue) => {
                        response.mark_changed();
                        let mut delta = click_point - origin;
                        delta = delta.max(egui::Vec2::ZERO);
                        hue_angle = delta.angle();
                        hsva.h = hue_angle / std::f32::consts::FRAC_PI_2;
                    }
                }
            }

            // Show the picker mesh, if one was made.
            if let Some(mesh) = mesh {
                painter.add(egui::Shape::mesh(mesh));
                // Show a stroke around the picker meshes.
                let stroke = egui::Stroke {
                    color: egui::Color32::BLACK,
                    width: 2.0,
                };
                // Main picker
                painter.circle_stroke(mesh_origin, mesh_radius, stroke);
                // Inner for hue circle
                painter.circle_stroke(mesh_origin, mesh_radius + RADIAL_MARGIN, stroke);
                // outer
                painter.circle_stroke(mesh_origin, mesh_radius + RADIAL_MARGIN + HUE_WIDTH, stroke);

                painter.circle(
                    mesh_origin
                        + egui::Vec2::angled(hue_angle)
                            * (mesh_radius + RADIAL_MARGIN + HUE_WIDTH / 2.0),
                    PICKER_SIZE,
                    egui::ecolor::Hsva::new(hsva.h, 1.0, 1.0, 1.0),
                    stroke,
                );
            }
            // Show the selected color in a bubble.
            // When the selector is collapsed, this is what the user interacts with to open it, it moves out to
            // it's position on the wheel when expanded.
            painter.circle(
                origin + expanded_proportion * egui::Vec2::angled(sv_angle) * sv_radius,
                egui::lerp(CONTRACTED_RADIUS..=PICKER_SIZE, expanded_proportion),
                *hsva,
                egui::Stroke {
                    color: grayscale_contrasting(*hsva, egui::Rgba::WHITE),
                    width: 1.0,
                },
            );
            // Double not here makes sense, i swear--
            // This means fully closed, not animating.
            // Would be good to show an icon here, an eyedropper specifically.
            // There is no such glyph, and the palette icon is confusing given the same
            // iconography is used elsewhere for a different purpose. Uh oh!
            /*
            if !not_closed {
                painter.text(
                    origin,
                    egui::Align2::CENTER_CENTER,
                    super::PALETTE_ICON,
                    egui::FontId::default(),
                    grayscale_contrasting(*hsva, egui::Rgba::BLACK),
                );
            }*/
            PickerResponse {
                in_flux: matches!(
                    click_target,
                    Some(ClickTarget::Hue | ClickTarget::SaturationValue)
                ),
                response,
            }
        })
        .inner
}

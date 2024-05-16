pub struct Handle;

impl egui::Widget for Handle {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let interact_height = ui.style().spacing.interact_size.y;
        let size = egui::vec2(interact_height * 2.0 / 3.0, interact_height);
        let response = ui.allocate_response(size, egui::Sense::drag());

        // Paint six dots, a somewhat universal drag icon.
        let painter = ui.painter();

        let heigh_spacing = size.y / 3.0;
        let width_spacing = size.x / 2.0;

        let dot_size = size.y / 16.0;

        let base_position =
            response.rect.left_top() + egui::vec2(width_spacing, heigh_spacing) / 2.0;

        let color = if response.dragged() {
            ui.style().visuals.selection.bg_fill
        } else {
            ui.style().interact(&response).fg_stroke.color
        };

        for x in 0..=1u8 {
            for y in 0..=2u8 {
                let offset = egui::vec2(f32::from(x) * width_spacing, f32::from(y) * heigh_spacing);
                painter.circle_filled(base_position + offset, dot_size, color);
            }
        }

        if response.dragged() {
            response.on_hover_and_drag_cursor(egui::CursorIcon::Grabbing)
        } else if response.hovered() {
            response.on_hover_cursor(egui::CursorIcon::Grab)
        } else {
            response
        }
    }
}

pub struct DropSeparatorResponse {
    /// The item was active, enabled, and hovered.
    pub selected: bool,
    pub response: egui::Response,
}

/// A separator, simpliar to [`egui::widgets::Separator`], but with optional hover detection.
pub struct DropSeparator {
    /// The separator is listening for drags.
    pub active: bool,
    /// The separator should respond as selected regardless of interaction state.
    pub already_selected: bool,
}
impl DropSeparator {
    pub fn show(self, ui: &mut egui::Ui) -> DropSeparatorResponse {
        let width = ui.available_size_before_wrap().x;

        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(width, ui.style().spacing.interact_size.y / 2.0),
            if self.active {
                egui::Sense::hover()
            } else {
                egui::Sense {
                    click: false,
                    drag: false,
                    focusable: false,
                }
            },
        );

        let painter = ui.painter();

        let is_responsive = self.already_selected || self.active && ui.is_enabled();
        let selected = self.already_selected || is_responsive && response.contains_pointer();

        let stroke = if is_responsive {
            let base_stroke = ui.style().interact(&response).fg_stroke;
            egui::Stroke {
                width: if selected {
                    base_stroke.width * 2.5
                } else {
                    base_stroke.width
                },
                ..base_stroke
            }
        } else {
            ui.style().noninteractive().bg_stroke
        };

        painter.hline(rect.x_range(), rect.left_center().y, stroke);

        DropSeparatorResponse { selected, response }
    }
}

use super::cast_vec;

/// Up right, normalized. This is intuitive for me, though I imagine left-handed
/// users would disagree. FIXME: make this a preference
const ZOOM_IN_DIRECTION_NORMALIZED: egui::Vec2 = egui::Vec2 {
    x: std::f32::consts::FRAC_1_SQRT_2,
    y: -std::f32::consts::FRAC_1_SQRT_2,
};
const ZOOM_IN_DIRECTION_CURSOR: egui::CursorIcon = egui::CursorIcon::ResizeNeSw;
/// (Approx) the 175th root of two, such that dragging 175 logical pixels
/// results in a doubling or halving of the size.
const ZOOM_SPEED_RATIO_PER_PX: f32 = 1.004;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Tool {
    Pan,
    Rotate,
    Zoom,
}
pub struct Scrub {
    tool: Tool,
    /// Arbitrary if not dragged.
    drag_start: egui::Pos2,
}
impl Scrub {
    pub fn new(tool: Tool) -> Self {
        Self {
            tool,
            drag_start: egui::Pos2::ZERO,
        }
    }
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        view_transform: &mut fuzzpaint_types::similarity::Similarity,
    ) {
        let will_discard = ui.ctx().will_discard();
        let rect = ui.max_rect();
        let center = rect.center();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let hover_or_zero = response.hover_pos().unwrap_or(egui::Pos2::ZERO);

        let cursor = match self.tool {
            Tool::Pan => {
                if response.dragged() {
                    egui::CursorIcon::Grabbing
                } else {
                    egui::CursorIcon::Grab
                }
            }
            Tool::Rotate => super::rotate_cursor_from_angle((hover_or_zero - center).angle()),
            Tool::Zoom => {
                if response.dragged() {
                    ZOOM_IN_DIRECTION_CURSOR
                } else {
                    egui::CursorIcon::ZoomIn
                }
            }
        };
        let response = response.on_hover_and_drag_cursor(cursor);

        // Provide a context menu to flip the canvas
        if response.clicked_by(egui::PointerButton::Secondary) {
            self.drag_start = response.interact_pointer_pos().unwrap();
        }
        response.context_menu(|ui| {
            if ui.button("Flip Horizontally").clicked() {
                view_transform.flip_h_around(self.drag_start.x);
            }
            if ui.button("Flip Vertically").clicked() {
                view_transform.flip_v_around(self.drag_start.y);
            }
        });

        // Draw the center of rotation.
        if response.hovered() && matches!(self.tool, Tool::Rotate) && !will_discard {
            ui.painter()
                .circle_stroke(center, 5.0, egui::Stroke::new(1.0, egui::Color32::BLACK));
        }

        // dragged(), but without the latency.
        if !response.is_pointer_button_down_on() {
            // If you remove this bail it'll panic in the rotation logic :3
            return;
        }
        if response.drag_started() {
            self.drag_start = response.interact_pointer_pos().unwrap();
        }
        match self.tool {
            Tool::Pan => {
                view_transform.translate_by(cast_vec(response.drag_delta()));
            }
            Tool::Rotate => {
                if !will_discard {
                    let painter = ui.painter();
                    painter.line_segment(
                        [response.interact_pointer_pos().unwrap(), center],
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                    );
                    painter.circle_stroke(
                        center,
                        10.0,
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                    );
                }
                view_transform.rotate_around(
                    // Panics if not dragged. Checked above.
                    super::signed_delta_rotation_around(&response, center),
                    super::cast_vec(center),
                );
            }
            Tool::Zoom => {
                // + if zooming in, - if out.
                let zoom_px = response.drag_delta().dot(ZOOM_IN_DIRECTION_NORMALIZED);
                let zoom_power = ZOOM_SPEED_RATIO_PER_PX.powf(zoom_px);
                view_transform.scale_around(zoom_power, cast_vec(self.drag_start));
            }
        }
    }
}

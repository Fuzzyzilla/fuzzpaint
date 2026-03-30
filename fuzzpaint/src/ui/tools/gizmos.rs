use super::cast_vec;

pub struct Transform {
    // This is transient state for this interaction, not actually part of the
    // transform as written to the document.
    /// Where rotations and scales should be centered around. When None, it's
    /// implicit by the type of handle being dragged (Corners -> Opposite
    /// corner, edges -> opposite edge center, unless holding Alt then it's
    /// around the center of the rect)
    pub center: Option<egui::Pos2>,
    pub animating_from: Option<fuzzpaint_types::similarity::Similarity>,
    pub started_animating: bool,
    pub transform: fuzzpaint_types::similarity::Similarity,

    // The rect to be transformed.
    pub rect: egui::Rect,
}
impl Transform {
    pub fn show(&mut self, ui: &mut egui::Ui) {
        const SCALE_HANDLE_SIZE: f32 = 10.0;
        /// Must be larger than the scale handle size to function properly.
        const ROTATE_HANDLE_SIZE: f32 = 15.0;
        const CENTER_POINT_SIZE: f32 = 10.0;
        let handle_stroke = egui::Stroke::new(2.0, egui::Color32::BLACK);
        let light_stroke = egui::Stroke::new(1.0, egui::Color32::BLACK);

        let matrix = if self.animating_from.is_some() {
            // FIXME: If the rect isn't centered around 0,0, this doesn't work.
            /*
            let id = ui.id().with("lerp");
            if !self.started_animating {
                // Instantly jump to 0.0
                ui.animate_value_with_time(id, 0.0, 0.0);
            }
            self.started_animating = true;
            // Animate up to 1.0
            let t = ui
                .animate_value_with_time(id, 1.0, ui.style().animation_time);

            animating_from.translate_by(cast_vec(-self.rect.center().to_vec2()));
            let mut animating_to = self.transform;
            animating_to.translate_by(cast_vec(-self.rect.center().to_vec2()));

            let mut lerped = animating_from.visual_lerp_to(t, &animating_to);
            lerped.translate_by(cast_vec(self.rect.center()));
            if t > 0.999 {
                self.animating_from = None;
                self.started_animating = false;
            }
            println!("{t} => {lerped:#?}");
            */
            self.animating_from = None;
            self.started_animating = false;
            self.transform.into_mat3()
        } else {
            self.transform.into_mat3()
        };
        let rect_center = cast_vec(matrix.transform_point2(cast_vec(self.rect.center())));
        let holding_alt = ui.input(|i| i.modifiers.alt);

        // Locks a mutex, so cache it instead.
        let do_paint = !ui.will_discard();

        if do_paint {
            // Egui does not support rotated shapes.
            let start = cast_vec(matrix.transform_point2(cast_vec(self.rect.left_top())));
            ui.painter().line(
                [
                    start,
                    cast_vec(matrix.transform_point2(cast_vec(self.rect.right_top()))),
                    cast_vec(matrix.transform_point2(cast_vec(self.rect.right_bottom()))),
                    cast_vec(matrix.transform_point2(cast_vec(self.rect.left_bottom()))),
                    start,
                ]
                .into(),
                light_stroke,
            );
        }

        // Draggable area behind everything else to move and to give context
        // menu:
        {
            let mut min = egui::Pos2::new(f32::INFINITY, f32::INFINITY);
            let mut max = egui::Pos2::new(-f32::INFINITY, -f32::INFINITY);
            for point in [
                self.rect.left_top(),
                self.rect.right_top(),
                self.rect.right_bottom(),
                self.rect.left_bottom(),
            ] {
                let xformed = matrix.transform_point2(cast_vec(point));
                min.x = min.x.min(xformed.x);
                min.y = min.y.min(xformed.y);
                max.x = max.x.max(xformed.x);
                max.y = max.y.max(xformed.y);
            }
            let aabb = egui::Rect::from_min_max(min, max);
            let response = ui
                .allocate_rect(aabb, egui::Sense::click_and_drag())
                .on_hover_and_drag_cursor(egui::CursorIcon::Move);
            self.transform.translate_by(cast_vec(response.drag_delta()));

            response.context_menu(|ui| {
                if ui.button("Flip Horizontally").clicked() {
                    self.animating_from = Some(self.transform);
                    self.transform
                        .flip_h_around(self.center.unwrap_or(rect_center).x);
                    ui.close();
                }
                if ui.button("Flip Vertically").clicked() {
                    self.animating_from = Some(self.transform);
                    self.transform
                        .flip_v_around(self.center.unwrap_or(rect_center).y);
                    ui.close();
                }
                ui.separator();
                if ui.button("Reset Center").clicked() {
                    self.center = None;
                    ui.close();
                }
                if ui.button("Undo Changes").clicked() {
                    log::error!("unimplemented");
                    ui.close();
                }
                if ui.button("Reset to Identity").clicked() {
                    self.animating_from = Some(self.transform);
                    self.transform = fuzzpaint_types::similarity::Similarity::IDENTITY;
                    ui.close();
                }
            });
        }

        // Draggable circle to change transformation center
        {
            let center = self.center.unwrap_or(rect_center);

            let response = ui
                .allocate_rect(
                    egui::Rect::from_center_size(center, egui::Vec2::splat(CENTER_POINT_SIZE)),
                    egui::Sense::click_and_drag(),
                )
                .on_hover_cursor(egui::CursorIcon::Move);
            if response.dragged() {
                self.center = Some(center + response.drag_delta());
            } else if response.secondary_clicked()
                || response.clicked_by(egui::PointerButton::Middle)
            {
                self.center = None;
            }
            if do_paint {
                ui.painter().circle_stroke(
                    center,
                    if self.center.is_some() {
                        CENTER_POINT_SIZE
                    } else {
                        CENTER_POINT_SIZE * 0.5
                    },
                    if self.center.is_some() && !holding_alt {
                        handle_stroke
                    } else {
                        light_stroke
                    },
                );
            }
        }

        // A single draggable handle to rotate. Rendered at `point`, rotaing
        // around `center` (both in egui space).
        let mut rotate_handle = |point: egui::Pos2, center: egui::Pos2| {
            let handle_rect =
                egui::Rect::from_center_size(point, egui::Vec2::splat(ROTATE_HANDLE_SIZE));

            let response = ui
                .allocate_rect(handle_rect, egui::Sense::drag())
                // Rotate the cursor 90 deg so it's tangent to the direction of
                // scaling. Since there is no standard Rotation cursor, this is
                // the best i can do for now.
                .on_hover_and_drag_cursor(super::rotate_cursor_from_angle(
                    (center - point).angle(),
                ));
            if response.dragged() {
                self.transform.rotate_around(
                    // Angle is unsigned, figure out if cw or ccw.
                    super::signed_delta_rotation_around(&response, center),
                    cast_vec(center),
                );
            }

            // Only draw if hovered. better helps communicate the difference
            // between the scale handle and the rotation handle, since the
            // rotation cursors aren't great.
            if response.hovered() && do_paint {
                let painter = ui.painter();
                if response.dragged() {
                    painter.line_segment([point, center], light_stroke);
                }
                painter.circle_stroke(point, ROTATE_HANDLE_SIZE, handle_stroke);
            }
        };

        for pair in [
            [self.rect.right_bottom(), self.rect.left_top()],
            [self.rect.left_center(), self.rect.right_center()],
            [self.rect.right_top(), self.rect.left_bottom()],
            [self.rect.center_top(), self.rect.center_bottom()],
        ] {
            let pair = pair.map(|x| cast_vec(matrix.transform_point2(cast_vec(x))));
            for [from, to] in [[pair[0], pair[1]], [pair[1], pair[0]]] {
                rotate_handle(
                    from,
                    if holding_alt {
                        to
                    } else {
                        self.center.unwrap_or(rect_center)
                    },
                );
            }
        }

        // A single draggable handle to change the scale. Rendered at `point`,
        // scaling towards `center` (both in egui space).
        let mut scale_handle = |point: egui::Pos2, center: egui::Pos2| {
            let handle_rect =
                egui::Rect::from_center_size(point, egui::Vec2::splat(SCALE_HANDLE_SIZE));

            let response = ui
                .allocate_rect(handle_rect, egui::Sense::drag())
                .on_hover_and_drag_cursor(super::resize_cursor_from_angle(
                    (center - point).angle(),
                ));
            if response.dragged() {
                let handle_to_center = center - point;
                let distance_to_center_px = handle_to_center.length();
                let delta = response.drag_delta();

                let px_towards_center = delta.dot(handle_to_center.normalized());

                self.transform.scale_around(
                    (distance_to_center_px - px_towards_center) / distance_to_center_px,
                    cast_vec(center),
                );
            }

            if do_paint {
                let painter = ui.painter();
                if response.dragged() {
                    painter.line_segment([point, center], light_stroke);
                }
                painter.rect_stroke(handle_rect, 0, handle_stroke, egui::StrokeKind::Outside);
            }
        };

        // If holding alt, use the center of the rect. Otherwise, use the
        // defined center. If Neither, use the implicit center.
        let scale_center = holding_alt.then_some(rect_center).or(self.center);

        for pair in [
            [self.rect.right_bottom(), self.rect.left_top()],
            [self.rect.left_center(), self.rect.right_center()],
            [self.rect.right_top(), self.rect.left_bottom()],
            [self.rect.center_top(), self.rect.center_bottom()],
        ] {
            let pair = pair.map(|x| cast_vec(matrix.transform_point2(cast_vec(x))));
            for [from, to] in [[pair[0], pair[1]], [pair[1], pair[0]]] {
                scale_handle(from, scale_center.unwrap_or(to));
            }
        }
    }
}

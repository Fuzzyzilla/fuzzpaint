mod gizmos;
mod rulers;
mod snapper;
mod typography;
mod viewport_scrub;

/// Cursed generic :3 Makes egui Vec or Pos from/into ultraviolet vecs.
fn cast_vec<T: From<[f32; 2]>>(vec: impl Into<[f32; 2]>) -> T {
    From::from(vec.into())
}

/// Get a bidirectional resize cursor from the given angle in radians, clockwise
/// from right.
fn resize_cursor_from_angle(angle: f32) -> egui::CursorIcon {
    use std::f32::consts::{FRAC_PI_8, PI};
    const FRAC_1_PI_16: f32 = const { FRAC_PI_8 / 2.0 };
    const FRAC_7_PI_16: f32 = const { FRAC_1_PI_16 * 7.0 };
    const FRAC_9_PI_16: f32 = const { FRAC_1_PI_16 * 9.0 };
    const FRAC_15_PI_16: f32 = const { FRAC_1_PI_16 * 15.0 };

    // 0..PI
    let angle = angle.rem_euclid(PI);
    // Divide the semicircle into sixteenths. The Vertical and horizontal arrows
    // get a relatively smaller slice of the pi. *buh dum tss*
    match angle {
        FRAC_1_PI_16..FRAC_7_PI_16 => egui::CursorIcon::ResizeNwSe,
        FRAC_7_PI_16..FRAC_9_PI_16 => egui::CursorIcon::ResizeVertical,
        FRAC_9_PI_16..FRAC_15_PI_16 => egui::CursorIcon::ResizeNeSw,
        // _ should be unreachable, but float math:tm:
        0.0..FRAC_1_PI_16 | FRAC_15_PI_16..PI | _ => egui::CursorIcon::ResizeHorizontal,
    }
}
/// Get a bidirectional rotation cursor from the given angle from the center of
/// rotation to the mouse cursor, in radians, clockwise from right.
fn rotate_cursor_from_angle(angle: f32) -> egui::CursorIcon {
    // A quarter rotation offset from the resize cursors. That way, the bidi
    // arrow is perpendicular to the center of rotation.
    resize_cursor_from_angle(angle + std::f32::consts::FRAC_PI_2)
}

#[derive(Default)]
enum InnerState {
    #[default]
    Brush,
    Transform(gizmos::Transform),
    Scrub {
        tool: viewport_scrub::Tool,
        state: viewport_scrub::Scrub,
    },
    Typography(typography::Typography),
}

#[derive(Default)]
pub struct ToolState {
    tool: Tool,
    state: InnerState,
}
impl ToolState {
    pub fn gizmo_layer() -> egui::LayerId {
        egui::LayerId::background()
    }
    /// Draw background graphics. Only call if there is as document viewport open.
    pub fn gizmos(
        &mut self,
        ctx: &egui::Context,
        viewport: egui::Rect,
        interface: &mut super::interface::Interface<'_, '_>,
    ) {
        // Draw mouse cursors:
        let layer = Self::gizmo_layer();
        let scale_factor = ctx.zoom_factor();

        let painter = ctx.layer_painter(layer);
        let draw_pointer = |hover: crate::window::stylus_events::Hover| {
            let position = egui::Pos2::from(*hover.position().as_array()) / scale_factor;

            // Fade out with distance.
            let color = egui::Color32::BLACK.gamma_multiply(1.0 - hover.distance().unwrap_or(0.0));

            // Greater than some epsilon, in case of pointers that always return
            // a distance of 0.
            if let Some(distance) = hover.distance()
                && distance > 0.1
            {
                painter.circle_stroke(
                    position,
                    // Grow with distance.
                    distance * 20.0,
                    egui::Stroke::new(2.0, color),
                );
            }
            painter.circle_filled(position, 5.0, color);
        };

        // Doesn't detect CentralPanels :/
        if !ctx.is_using_pointer() && ctx.rect_contains_pointer(layer, viewport) {
            ctx.set_cursor_icon(egui::CursorIcon::None);
            if let Some(hover) = interface.pointers().primary_hover() {
                draw_pointer(hover);
            }
        }
        for hover in interface.pointers().auxiliary_hovers() {
            draw_pointer(hover);
        }

        {
            let mut ui = egui::Ui::new(
                ctx.clone(),
                egui::Id::new("gizmos"),
                egui::UiBuilder::new().max_rect(viewport).layer_id(layer),
            );
            self.do_tool(&mut ui);
        }
    }
    fn do_tool(&mut self, ui: &mut egui::Ui) {
        match self.tool {
            Tool::Brush | Tool::Eraser => self.state = InnerState::Brush,
            Tool::Transform => {
                if !matches!(&self.state, InnerState::Transform(_)) {
                    self.state = InnerState::Transform(gizmos::Transform {
                        center: None,
                        animating_from: None,
                        started_animating: false,
                        transform: fuzzpaint_types::similarity::Similarity::IDENTITY,
                        rect: egui::Rect::from_min_size(
                            egui::Pos2 { x: 300.0, y: 200.0 },
                            egui::Vec2::new(200.0, 100.0),
                        ),
                    });
                }
                let InnerState::Transform(state) = &mut self.state else {
                    unreachable!()
                };
                state.show(ui);
            }
            Tool::ViewPan | Tool::ViewScrubZoom | Tool::ViewRotate => {
                let tool = match self.tool {
                    Tool::ViewPan => viewport_scrub::Tool::Pan,
                    Tool::ViewScrubZoom => viewport_scrub::Tool::Zoom,
                    Tool::ViewRotate => viewport_scrub::Tool::Rotate,
                    _ => unreachable!(),
                };
                if !matches!(&self.state, InnerState::Scrub{tool: prev_tool, ..} if *prev_tool == tool)
                {
                    self.state = InnerState::Scrub {
                        tool,
                        state: viewport_scrub::Scrub::new(tool),
                    };
                }
                let InnerState::Scrub { state, .. } = &mut self.state else {
                    unreachable!()
                };
                state.show(
                    ui,
                    &mut fuzzpaint_types::similarity::Similarity::IDENTITY.clone(),
                );
            }
            Tool::Typography => {
                if !matches!(&self.state, InnerState::Typography(_)) {
                    self.state = InnerState::Typography(typography::Typography::new());
                }
                let InnerState::Typography(state) = &mut self.state else {
                    unreachable!()
                };
                state.show(ui);
            }
            _ => unimplemented!(),
        }
    }
    pub fn tool(&self) -> Tool {
        self.tool
    }
    pub fn show_toolbox_column(&mut self, ctx: &egui::Context, side: egui::panel::Side) {
        // Min size, expanding.
        const TOOLBOX_BUTTON_SIZE: f32 = 30.0;
        const ICON_SIZE_RATIO: f32 = 0.9;
        egui::SidePanel::new(side, "toolbox")
            .resizable(true)
            .default_width(TOOLBOX_BUTTON_SIZE)
            // Order is important, default_width overrides range.
            .width_range(TOOLBOX_BUTTON_SIZE..)
            .frame(egui::Frame {
                inner_margin: egui::Margin::ZERO,
                ..egui::Frame::side_top_panel(&ctx.style())
            })
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    // Very narrow bar, the scrollbar is quite large relative to it lol.
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| {
                        ui.style_mut().spacing.item_spacing = egui::Vec2::ZERO;

                        // Avoid rounding errors which cause incorrect wrapping
                        // when there should be space.
                        let button_size = TOOLBOX_BUTTON_SIZE - 0.1;

                        let spacing = ui.spacing_mut();
                        spacing.interact_size = egui::Vec2::splat(button_size);
                        spacing.button_padding = egui::Vec2::ZERO;

                        let font = egui::FontId::new(
                            button_size * ICON_SIZE_RATIO,
                            super::GOOGLE_MATERIAL_ICONS_FAMILY.clone(),
                        );

                        let mut is_first = true;
                        for &tool_group in Tool::GROUPS {
                            if !is_first {
                                ui.separator();
                            }
                            is_first = false;

                            ui.horizontal_wrapped(|ui| {
                                for &tool in tool_group {
                                    let mut button = egui::Button::new(
                                        egui::RichText::new(tool.icon().to_string())
                                            .font(font.clone()),
                                    )
                                    .min_size(egui::Vec2::splat(button_size))
                                    .frame_when_inactive(false);
                                    if tool == self.tool {
                                        button = button
                                            .selected(true)
                                            // Add a bit more visual distinction. Selected alone
                                            // is just the vaguest cyan tinge, lol.
                                            .frame_when_inactive(true);
                                    }
                                    let mut response =
                                        ui.add_enabled(self.tool_available(tool), button);
                                    response = response.on_disabled_hover_text("(unimplemented)");

                                    if let Some(name) = strum::EnumMessage::get_message(&tool) {
                                        response = response.on_hover_text(name);
                                    }
                                    if let Some(doc) = strum::EnumMessage::get_documentation(&tool)
                                    {
                                        response = response.on_hover_text(doc);
                                    }
                                    if response.clicked() {
                                        self.tool = tool;
                                    }
                                }
                            });
                        }
                    });
            });
    }
    /// Given the state of the document, should this tool be accessible in the toolbox?
    pub fn tool_available(&mut self, tool: Tool) -> bool {
        matches!(
            tool,
            Tool::Brush
                | Tool::Eraser
                | Tool::ViewPan
                | Tool::ViewScrubZoom
                | Tool::ViewRotate
                | Tool::Transform
                | Tool::Typography
        )
    }
}

#[derive(strum::EnumMessage, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tool {
    /// Freehand drawing and erasing.
    #[strum(message = "Brush")]
    #[default]
    Brush,
    /// Freehand erasing.
    // Is just `Brush` but always set to erase. Included so that there can be a
    // sidepanel button for an "eraser" tool instead of just the erase hotkey.
    #[strum(message = "Eraser")]
    Eraser,

    /// Fill contiguous regions of color.
    #[strum(message = "Flood Fill")]
    FloodFill,
    /// Edit color gradients
    #[strum(message = "Gradient")]
    Gradient,

    /// Sample traits from the document (colors, brushes, etc.)
    #[strum(message = "Sample")]
    Sample,

    /// Modify visual guide lines and markups.
    #[strum(message = "Guide")]
    Guide,
    /// Draw pre-defined shapes (Circles, rectangles, lines, catenaries, etc.)
    #[strum(message = "Trace")]
    Trace,
    /// Leave meta notes.
    #[strum(message = "Annotate")]
    Annotate,
    /// Measure distances and angles on the document.
    #[strum(message = "Ruler")]
    Ruler,

    /// Simple translation
    #[strum(message = "Move")]
    Move,
    /// Transforms stroke Vector scale, rotate, translation, and flip.
    // Silly but this has some branding behind it lol. This is a major feature
    // of fuzzpaint, it's important to trick the user into finding it by calling
    // it just "Transform" - not anything weird.
    #[strum(message = "Transform")]
    Transform,
    /// Arbitrary transform, including perspective and shear.
    #[strum(message = "Stretch")]
    Stretch,

    /// Text and text constraint editor.
    // Im calling this "typography" instead of "text" because im a typography
    // nerd and want to manifest this into a rlly good typography toolkit
    // instead of a basic textbox. :3
    #[strum(message = "Typography")]
    Typography,

    /// Modify the canvas size.
    #[strum(message = "Crop")]
    Crop,

    /// Freeform selection.
    #[strum(message = "Lasso")]
    Lasso,
    /// Rectangular selection.
    #[strum(message = "Marquee")]
    Marquee,

    /// Drag the canvas to pan the view.
    #[strum(message = "Pan View")]
    ViewPan,
    /// Drag the canvas to zoom the view in and out.
    #[strum(message = "Scrub Zoom View")]
    ViewScrubZoom,
    /// Drag the canvas to rotate the view.
    #[strum(message = "Rotate View")]
    ViewRotate,
}
impl Tool {
    pub const GROUPS: &[&[Self]] = &[
        &[Self::Brush, Self::Eraser],
        &[Self::FloodFill, Self::Gradient],
        &[Self::Sample],
        &[Self::Guide, Self::Trace, Self::Annotate, Self::Ruler],
        &[Self::Move, Self::Transform, Self::Stretch],
        &[Self::Typography],
        &[Self::Crop],
        &[Self::Lasso, Self::Marquee],
        &[Self::ViewPan, Self::ViewScrubZoom, Self::ViewRotate],
    ];
    pub fn icon(self) -> material_icons::Icon {
        use material_icons::Icon;
        match self {
            Tool::Brush => Icon::Brush,
            // Sux :V
            Tool::Eraser => Icon::EditOff,
            Tool::FloodFill => Icon::FormatColorFill,
            Tool::Gradient => Icon::Gradient,
            Tool::Sample => Icon::Colorize,
            Tool::Guide => Icon::BorderInner,
            Tool::Trace => Icon::ShapeLine,
            Tool::Annotate => Icon::StickyNote2,
            Tool::Ruler => Icon::SquareFoot,
            Tool::Move => Icon::ControlCamera,
            Tool::Transform => Icon::Transform,
            Tool::Stretch => Icon::AspectRatio,
            Tool::Typography => Icon::TextFields,
            Tool::Crop => Icon::Crop,
            // Sux :V
            Tool::Lasso => Icon::Gesture,
            Tool::Marquee => Icon::HighlightAlt,
            Tool::ViewPan => Icon::PanToolAlt,
            Tool::ViewScrubZoom => Icon::ZoomIn,
            Tool::ViewRotate => Icon::RotateRight,
        }
    }
}

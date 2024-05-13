mod brush_ui;
mod color_palette;
mod drag;
mod modal;
pub mod requests;

use modal::Modal;

use egui::{RichText, Ui};
use fuzzpaint_core::{
    blend::{Blend, BlendMode},
    brush,
    color::{self as fcolor, PaletteIndex},
    io,
    queue::{self, state_reader::CommandQueueStateReader},
    state,
    util::FiniteF32,
};

const STROKE_LAYER_ICON: &str = "✏";
const TEXT_LAYER_ICON: &str = "🗛";
const NOTE_LAYER_ICON: &str = "🖹";
const FILL_LAYER_ICON: &str = "⬛";
const GROUP_ICON: &str = "🗀";
const SCISSOR_ICON: &str = "✂";
const PLUS_ICON: char = '➕';
const PALETTE_ICON: char = '🎨';
const HISTORY_ICON: char = '🕒';
const HOME_ICON: char = '🏠';
const PIN_ICON: char = '📌';
const ALPHA_ICON: &str = "α";

/// Justify `(available_size, size, margin)` -> `(size', margin')`, such that `count` elements
/// will fill available space completely.
///
/// There are four ways to do this:
/// increase or decrease size or margin.
/// Current implementation increases size.
fn justify(available_size: f32, base_size: f32, base_margin: f32) -> (f32, f32) {
    // The math was originally a bit more sophisticated to avoid margin fenceposting,
    // but it seems egui eagerly adds margin after even the last element so it's weird!

    // get integer number of elements, at least one.
    let num_buttons = (available_size / (base_size + base_margin))
        .floor()
        .max(1.0);
    assert!(num_buttons.is_finite());

    let just_size = (available_size - (num_buttons * base_margin)) / num_buttons;

    (just_size, base_margin)
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
enum JustifyAxis {
    Horizontal,
    Vertical,
}
/// Adjusts UI style to be justified according to [`justify`].
///
/// Returns the size at which items should be drawn, margin is applied in-place
/// to the Ui via `Style::spacing::item_spacing`
fn justify_mut(ui: &mut Ui, axis: JustifyAxis, base_size: f32, base_margin: f32) -> f32 {
    let available = match axis {
        JustifyAxis::Horizontal => ui.available_width(),
        JustifyAxis::Vertical => ui.available_height(),
    };
    let (size, margin) = justify(available, base_size, base_margin);

    ui.style_mut().spacing.item_spacing = egui::Vec2::splat(margin);

    size
}
trait ResponseExt {
    /// Emulate a primary click whenever this action is triggered.
    fn or_action_clicked(
        self,
        frame: &crate::actions::ActionFrame,
        action: crate::actions::Action,
    ) -> Self;
    fn clicked_or_escape(self) -> bool;
}
impl ResponseExt for egui::Response {
    fn or_action_clicked(
        self,
        frame: &crate::actions::ActionFrame,
        action: crate::actions::Action,
    ) -> Self {
        if !self.enabled || !self.sense.click {
            return self;
        }
        let triggered = frame.action_trigger_count(action) > 0;
        let held = frame.is_action_held(action);

        Self {
            clicked: [
                self.clicked[0] || triggered,
                self.clicked[1],
                self.clicked[2],
                self.clicked[3],
                self.clicked[4],
            ],
            is_pointer_button_down_on: self.is_pointer_button_down_on || held,

            ..if held { self.highlight() } else { self }
        }
    }
    /// Returns true if [`egui::Response::clicked`] or `Escape` key is pressed, useful for cancel buttons.
    /// This does not take into account focus.
    fn clicked_or_escape(self) -> bool {
        self.clicked() || self.ctx.input(|input| input.key_pressed(egui::Key::Escape))
    }
}

enum CurrentModal {
    BrushCreation(brush_ui::CreationModal),
}

enum CloseState {
    None,
    Modal,
    Confirmed,
}

#[derive(Clone)]
struct PerDocumentData {
    id: state::DocumentID,
    graph_selection: Option<state::graph::AnyID>,
    graph_focused_subtree: Option<state::graph::NodeID>,
    name: String,
}
pub struct MainUI {
    close_state: CloseState,
    documents: Vec<PerDocumentData>,
    cur_document: Option<state::DocumentID>,

    modal: Option<CurrentModal>,
    picker_color: egui::ecolor::HsvaGamma,
    picker_in_flux: bool,
    picker_changed: bool,

    requests_send: crossbeam::channel::Sender<requests::UiRequest>,
    requests_recv: crossbeam::channel::Receiver<requests::UiRequest>,
    action_listener: crate::actions::ActionListener,
}
impl MainUI {
    #[must_use]
    pub fn new(action_listener: crate::actions::ActionListener) -> Self {
        let documents = crate::global::provider().document_iter();
        let documents: Vec<_> = documents
            .map(|id| PerDocumentData {
                id,
                graph_focused_subtree: None,
                graph_selection: None,
                name: "Unknown".into(),
            })
            .collect();
        let cur_document = documents.last().map(|doc| doc.id);

        let (requests_send, requests_recv) = crossbeam::channel::unbounded();
        Self {
            close_state: CloseState::None,
            documents,
            cur_document,

            modal: None,
            picker_color: egui::ecolor::HsvaGamma {
                h: 0.0,
                s: 0.0,
                v: 0.0,
                a: 1.0,
            },
            picker_in_flux: false,
            picker_changed: false,

            requests_send,
            requests_recv,
            action_listener,
        }
    }
    /// Marks that a close has been requested by the windower
    pub fn close_requested(&mut self) {
        // Close unconditionally if
        // * A close was requested again even though the modal is up.
        //   (Either we crashed and the modal isn't seen or the user *really* wants us to close lol)
        // * No open documents to save anyway.
        self.close_state =
            if matches!(self.close_state, CloseState::Modal) || self.documents.is_empty() {
                CloseState::Confirmed
            } else {
                // There are open docs, prompt
                CloseState::Modal
            }
    }
    /// Returns true if the app should close.
    #[must_use]
    pub fn should_close(&self) -> bool {
        matches!(self.close_state, CloseState::Confirmed)
    }
    /// Returns true if a top-level modal exists asking whether to close the app.
    #[must_use]
    fn has_close_modal(&self) -> bool {
        matches!(self.close_state, CloseState::Modal)
    }
    /// Returns true if any modal is open.
    #[must_use]
    fn has_any_modal(&self) -> bool {
        self.has_close_modal() || self.modal.is_some()
    }
    #[must_use]
    pub fn listen_requests(&self) -> crossbeam::channel::Receiver<requests::UiRequest> {
        self.requests_recv.clone()
    }
    /// Main UI and any modals, with the top bar, layers, brushes, color, etc. To be displayed in front of the document and it's gizmos.
    /// Returns the size of the document's viewport space - that is, the size of the rect not covered by any side/top/bottom panels.
    /// None if a full-screen menu is shown.
    pub fn ui(&mut self, ctx: &egui::Context) -> Option<(ultraviolet::Vec2, ultraviolet::Vec2)> {
        if self.has_close_modal() {
            self.do_close_modal(ctx);
        }
        // Display modals before main. Egui will place the windows without regard for free area.
        self.do_modal(ctx, !self.has_close_modal());

        // Show, but disable if modal exists.
        self.main_ui(ctx, !self.has_any_modal())
    }
    fn get_cur_interface(&mut self) -> Option<&mut PerDocumentData> {
        // Get the document's interface, or reset to none if not found.
        // Weird inspect_none
        if let Some(interface) = self.cur_document.and_then(|cur| {
            self.documents
                .iter_mut()
                .find(|interface| interface.id == cur)
        }) {
            Some(interface)
        } else {
            self.cur_document = None;
            *crate::AdHocGlobals::get().write() = None;
            None
        }
    }
    fn do_close_modal(&mut self, ctx: &egui::Context) {
        let clicked_elsewhere = egui::Window::new("Exit")
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label("There are unsaved documents. Do you really want to exit?");
                ui.horizontal(|ui| {
                    // On first run-thru it would be nice for this cancel button to auto-focus itself.
                    if ui.button("Cancel").clicked_or_escape() {
                        self.close_state = CloseState::None;
                    }
                    if ui.button("Exit").clicked() {
                        self.close_state = CloseState::Confirmed;
                    }
                });
            })
            .map_or(false, |resp| resp.response.clicked_elsewhere());

        // If the user clicks away from the window assume they cancelled.
        if clicked_elsewhere {
            self.close_state = CloseState::None;
        }
    }
    /// Execute the current modal's logic and window.
    fn do_modal(&mut self, ctx: &egui::Context, enabled: bool) {
        let Some(modal) = self.modal.as_mut() else {
            return;
        };

        let title = match modal {
            CurrentModal::BrushCreation(_) => brush_ui::CreationModal::NAME,
        };

        let mut is_open = true;

        let cancelled = egui::Window::new(title)
            .collapsible(false)
            .enabled(enabled)
            .open(&mut is_open)
            .show(ctx, |ui| match modal {
                CurrentModal::BrushCreation(b) => match b.do_ui(ui) {
                    modal::Response::Cancel(()) => true,
                    modal::Response::Continue => false,
                    _ => unimplemented!(),
                },
            })
            .and_then(|resp| resp.inner)
            .unwrap_or(false);

        // Closed :3
        if !is_open || cancelled {
            self.modal = None;
        }
    }
    fn new_document(&mut self) {
        // When making a new document, start out with a white bg and stroke layer.
        // (These additions are not included in the history, but that's Okay!)
        let mut graph = fuzzpaint_core::state::graph::BlendGraph::default();
        let _ = graph.add_leaf(
            state::graph::Location::IndexIntoRoot(0),
            "Background".to_owned(),
            state::graph::LeafType::SolidColor {
                blend: Blend::default(),
                source: fuzzpaint_core::color::ColorOrPalette::WHITE,
            },
        );

        let mut stroke_collection =
            fuzzpaint_core::state::stroke_collection::StrokeCollectionState::default();
        let new_collection = crate::FuzzID::default();
        stroke_collection.0.insert(
            new_collection,
            state::stroke_collection::StrokeCollection::default(),
        );

        // Insert the stroke layer we just allocated a collection for.
        let stroke_layer = graph
            .add_leaf(
                state::graph::Location::IndexIntoRoot(0),
                "Stroke Layer".to_owned(),
                state::graph::LeafType::StrokeLayer {
                    blend: Blend::default(),
                    collection: new_collection,
                },
            )
            .ok();
        if stroke_layer.is_none() {
            // Uh oh, failed to make that layer. Remove the collection to not leave it orphaned.
            stroke_collection.0.clear();
        }

        let name = "New Document".to_owned();

        // Give this state to a queue
        let new_doc = queue::DocumentCommandQueue::from_state(
            state::Document {
                path: None,
                name: name.clone(),
            },
            graph,
            stroke_collection,
            fuzzpaint_core::state::palette::Palette::default(),
        );

        let new_id = new_doc.id();
        // Can't fail, this is a newly allocated ID so it's unqieu
        let _ = crate::global::provider().insert(new_doc);
        let interface = PerDocumentData {
            id: new_id,
            graph_focused_subtree: None,
            graph_selection: stroke_layer.map(Into::into),
            name,
        };
        let _ = self.requests_send.send(requests::UiRequest::Document {
            target: new_id,
            request: requests::DocumentRequest::Opened,
        });
        self.cur_document = Some(new_id);
        self.documents.push(interface);
    }
    fn open_documents(&mut self) {
        // Synchronous and bad just for now.
        if let Some(files) = rfd::FileDialog::new().pick_files() {
            let point_repository = crate::global::points();
            let provider = crate::global::provider();

            // Keep track of the last successful loaded id
            let mut recent_success = None;
            for file in files {
                match io::read_path(file, point_repository) {
                    Ok(doc) => {
                        let id = doc.id();
                        if provider.insert(doc).is_ok() {
                            recent_success = Some(id);
                            self.documents.push(PerDocumentData {
                                id,
                                graph_focused_subtree: None,
                                graph_selection: None,
                                name: "Unknown".into(),
                            });
                        }
                    }
                    Err(e) => log::error!("Failed to load: {e:#}"),
                }
            }
            // Select last one, if any succeeded.
            if let Some(new_doc) = recent_success {
                self.cur_document = Some(new_doc);
            }
        }
    }
    /// Render just self. Modals and insets handled separately.
    fn main_ui(
        &mut self,
        ctx: &egui::Context,
        enabled: bool,
    ) -> Option<(ultraviolet::Vec2, ultraviolet::Vec2)> {
        let Ok(action_frame) = self.action_listener.frame() else {
            let viewport = ctx.available_rect();
            let pos = viewport.left_top();
            let size = viewport.size();
            return Some((
                ultraviolet::Vec2 { x: pos.x, y: pos.y },
                ultraviolet::Vec2 {
                    x: size.x,
                    y: size.y,
                },
            ));
        };
        let interface = self.get_cur_interface().cloned();

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            ui.set_enabled(enabled);
            self.menu_bar(ui);
        });

        // If there's a document, show relavent tool panels. Otherwise,
        // show a big splash screen.
        if self.cur_document.is_none() {
            // Don't show the bar if it has nothing to say!
            if !self.documents.is_empty() {
                egui::TopBottomPanel::top("document-bar").show(ctx, |ui| {
                    ui.set_enabled(enabled);
                    self.document_bar(ui);
                });
            }
            self.welcome_screen(ctx);

            // When the welcome screen is shown, there is no space for the document view.
            None
        } else {
            egui::TopBottomPanel::bottom("nav_bar").show(ctx, |ui| {
                ui.set_enabled(enabled);
                if let Some(interface) = interface {
                    Self::nav_bar(ui, interface.id, &self.requests_send, &action_frame);
                }
            });
            egui::SidePanel::right("layers").show(ctx, |ui| {
                ui.set_enabled(enabled);
                ui.label("Layers");
                ui.separator();
                if let Some(interface) = self.get_cur_interface() {
                    layers_panel(ui, interface);

                    // Update selections.
                    let mut globals = crate::AdHocGlobals::get().write();
                    let old_brush = globals.take().map(|globals| globals.brush);
                    *globals = Some(crate::AdHocGlobals {
                        document: interface.id,
                        brush: old_brush.unwrap_or(state::StrokeBrushSettings {
                            is_eraser: false,
                            brush: fuzzpaint_core::brush::UniqueID([0; 32]),
                            color_modulate: fcolor::ColorOrPalette::BLACK,
                            size_mul: FiniteF32::new(10.0).unwrap(),
                            spacing_px: FiniteF32::new(0.5).unwrap(),
                        }),
                        node: interface.graph_selection,
                    });
                }
            });

            egui::SidePanel::left("inspector")
                .resizable(true)
                .show(ctx, |ui| {
                    ui.set_enabled(enabled);
                    // Stats at bottom
                    egui::TopBottomPanel::bottom("stats-panel").show_inside(ui, stats_panel);
                    // Toolbox above that
                    egui::TopBottomPanel::bottom("tools-panel")
                        .show_inside(ui, |ui| tools_panel(ui, &action_frame, &self.requests_send));
                    // Brush panel takes the rest
                    self.colors_panel(ui, self.cur_document, &action_frame);
                });
            egui::TopBottomPanel::top("document-bar").show(ctx, |ui| {
                ui.set_enabled(enabled);
                self.document_bar(ui);
            });

            {
                let response = color_palette::picker_dock(ctx, &mut self.picker_color);
                self.picker_changed = response.response.changed();
                self.picker_in_flux = response.in_flux;
            }

            let viewport = ctx.available_rect();
            let pos = viewport.left_top();
            let size = viewport.size();
            Some((
                ultraviolet::Vec2 { x: pos.x, y: pos.y },
                ultraviolet::Vec2 {
                    x: size.x,
                    y: size.y,
                },
            ))
        }
    }
    /// File, Edit, ect
    fn menu_bar(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("🐑").font(egui::FontId::proportional(20.0)))
                .on_hover_text("Baa");
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    let add_button = |ui: &mut Ui, label, shortcut| -> egui::Response {
                        let mut button = egui::Button::new(label);
                        if let Some(shortcut) = shortcut {
                            button = button.shortcut_text(shortcut);
                        }
                        ui.add(button)
                    };
                    if add_button(ui, "New", Some("Ctrl+N")).clicked() {
                        self.new_document();
                    };
                    if add_button(ui, "Save", Some("Ctrl+S")).clicked() {
                        // Dirty testing implementation!
                        if let Some(current) = self.cur_document {
                            std::thread::spawn(move || {
                                if let Some(reader) = crate::global::provider()
                                    .inspect(current, queue::DocumentCommandQueue::peek_clone_state)
                                {
                                    let repo = crate::global::points();

                                    let try_block = || -> anyhow::Result<()> {
                                        let mut path = dirs::document_dir().unwrap();
                                        path.push("temp.fzp");
                                        let file = std::fs::File::create(path)?;

                                        let start = std::time::Instant::now();
                                        io::write_into(&reader, repo, &file)?;
                                        let duration = start.elapsed();

                                        file.sync_all()?;
                                        if let Some(size) =
                                            file.metadata().ok().map(|meta| meta.len())
                                        {
                                            let size = size as f64;
                                            let speed = size / duration.as_secs_f64();
                                            log::info!(
                                                "Wrote {} in {}us ({}/s)",
                                                human_bytes::human_bytes(size),
                                                duration.as_micros(),
                                                human_bytes::human_bytes(speed)
                                            );
                                        } else {
                                            log::info!("Wrote in {}us", duration.as_micros());
                                        }
                                        Ok(())
                                    };

                                    if let Err(e) = try_block() {
                                        log::error!("Failed to write document: {e:?}");
                                    }
                                }
                            });
                        }
                    }
                    // let _ = add_button(ui, "Save as", Some("Ctrl+Shift+S"));
                    if add_button(ui, "Open", Some("Ctrl+O")).clicked() {
                        self.open_documents();
                    }
                    //let _ = add_button(ui, "Open as new", None);
                    //let _ = add_button(ui, "Export", None);
                });
            });
        });
    }
    /// Show a center welcome/"home" panel when no document is selected.
    fn welcome_screen(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                const BIG_BUTTON_MAX_SIZE: f32 = 125.0;
                const BIG_BUTTON_MIN_SIZE: f32 = 75.0;
                const BIG_BUTTON_MARGIN: f32 = 10.0;

                // Use double-size heading.
                let header_height = ui
                    .style()
                    .text_styles
                    .get(&egui::TextStyle::Heading)
                    .unwrap()
                    .size
                    * 2.5;

                // Show big buttons for new and open.
                let available_for_buttons = ui.available_size() - egui::vec2(0.0, header_height);

                // Too wide to show side by side, do a vertical layout.
                let (buttons_vertical, button_size) =
                    if BIG_BUTTON_MIN_SIZE * 2.0 + BIG_BUTTON_MARGIN > available_for_buttons.x {
                        (
                            true,
                            available_for_buttons
                                .x
                                .min(BIG_BUTTON_MAX_SIZE - BIG_BUTTON_MARGIN),
                        )
                    } else {
                        let biggest_possible_size =
                            (available_for_buttons.x - BIG_BUTTON_MARGIN * 3.0) / 2.0;
                        let size = biggest_possible_size
                            .min(BIG_BUTTON_MAX_SIZE)
                            .min(available_for_buttons.y);

                        (false, size)
                    };

                // Vertical center by discarding half the region not used by header or buttons.
                let button_height = if buttons_vertical {
                    2.0 * button_size + BIG_BUTTON_MARGIN
                } else {
                    button_size + BIG_BUTTON_MARGIN
                };
                let dead_space = available_for_buttons.y
                    - button_height
                    - 2.0 * ui.style().spacing.interact_size.y;
                ui.advance_cursor_after_rect(egui::Rect::from_min_size(
                    ui.next_widget_position(),
                    egui::vec2(0.0, dead_space / 2.0),
                ));

                ui.label(RichText::new("🐑 Fuzzpaint").heading().size(header_height));

                ui.vertical_centered(|ui| {
                    let big_button =
                        |ui: &mut egui::Ui, at: egui::Rect, name: &str| -> egui::Response {
                            // Create a UI in the rect, "justify" (fill) in both axes
                            let mut ui = ui.child_ui(
                                at,
                                egui::Layout::left_to_right(egui::Align::Center)
                                    .with_main_justify(true)
                                    .with_cross_justify(true),
                            );
                            let button = egui::Button::new(name);

                            ui.add(button)
                        };

                    let top_center = ui.next_widget_position();

                    let (a, b) = if buttons_vertical {
                        let top = egui::Rect::from_min_size(
                            top_center + egui::vec2(-button_size, 0.0),
                            [button_size; 2].into(),
                        );
                        let bottom = egui::Rect::from_min_size(
                            top_center + egui::vec2(-button_size, button_size + BIG_BUTTON_MARGIN),
                            [button_size; 2].into(),
                        );

                        (top, bottom)
                    } else {
                        let left = egui::Rect::from_min_size(
                            top_center - egui::vec2(BIG_BUTTON_MARGIN + button_size, 0.0),
                            [button_size; 2].into(),
                        );
                        let right = egui::Rect::from_min_size(
                            top_center + egui::vec2(BIG_BUTTON_MARGIN, 0.0),
                            [button_size; 2].into(),
                        );

                        (left, right)
                    };

                    if big_button(ui, a, "➕ New").clicked() {
                        self.new_document();
                    }
                    if big_button(ui, b, "🗀 Open").clicked() {
                        self.open_documents();
                    }
                });
            });
        });
    }
    /// Lists open documents
    fn document_bar(&mut self, ui: &mut Ui) {
        egui::ScrollArea::horizontal().show(ui, |ui| {
            ui.horizontal(|ui| {
                // if there is a selected doc, show a home button to get back to
                // welcome screen. If there isn't, it shows anyway, so no need!
                if self.cur_document.is_some()
                    && ui
                        .add(egui::Button::new(HOME_ICON.to_string()).frame(false))
                        .on_hover_text("Return to welcome screen")
                        .clicked()
                {
                    self.cur_document = None;
                }
                // Then show, a clicakble header for each document.
                let mut deleted_ids = smallvec::SmallVec::<[state::DocumentID; 1]>::new();
                for PerDocumentData { id, name, .. } in &self.documents {
                    let id = *id;
                    egui::containers::Frame::group(ui.style())
                        .outer_margin(egui::Margin::symmetric(0.0, 0.0))
                        .inner_margin(egui::Margin::symmetric(0.0, 0.0))
                        .multiply_with_opacity(if self.cur_document == Some(id) {
                            1.0
                        } else {
                            0.5
                        })
                        .rounding(egui::Rounding {
                            ne: 2.0,
                            nw: 2.0,
                            ..0.0.into()
                        })
                        .show(ui, |ui| {
                            let middle_click_delete = ui
                                .selectable_value(&mut self.cur_document, Some(id), name)
                                .clicked_by(egui::PointerButton::Middle);
                            if ui
                                .add(egui::Button::new("✖").small().frame(false))
                                .clicked()
                                || middle_click_delete
                            {
                                deleted_ids.push(id);
                                //Disselect if deleted. This is poor behavior, it should have some sort of recent-ness stack!
                                if self.cur_document == Some(id) {
                                    self.cur_document = None;
                                }
                            }
                        })
                        .response
                        .on_hover_ui(|ui| {
                            ui.label(format!("{id}"));
                        });
                }
                self.documents
                    .retain(|interface| !deleted_ids.contains(&interface.id));
                // Finally, show an add button.
                if ui
                    .add(egui::Button::new(PLUS_ICON.to_string()).frame(false))
                    .clicked()
                {
                    self.new_document();
                }
            });
        });
    }
    /// Bottom trim showing view controls.
    fn nav_bar(
        ui: &mut Ui,
        document: state::DocumentID,
        requests: &crossbeam::channel::Sender<requests::UiRequest>,
        frame: &crate::actions::ActionFrame,
    ) {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Everything here is shown in reverse order!

            ui.label(":V");

            ui.add(egui::Separator::default().vertical());

            // Todo: Some actions (scale slider, rotation slider, etc) are impossible to implement as of now
            // as the rest of the world has no way to communicate into the UI (so, no reporting of current transform)

            //Zoom controls
            if ui.small_button("⟲").clicked() {
                let _ = requests.send(requests::UiRequest::Document {
                    target: document,
                    request: requests::DocumentRequest::View(requests::DocumentViewRequest::Fit),
                });
            }
            let mut zoom = None::<f32>;
            egui::ComboBox::new("Zoom", "Zoom")
                // We don't actually know the current zoom, mwehehehe so sneaky
                .selected_text("...")
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut zoom, Some(0.25), "25%");
                    ui.selectable_value(&mut zoom, Some(0.5), "50%");
                    ui.selectable_value(&mut zoom, Some(1.0), "100%");
                    ui.selectable_value(&mut zoom, Some(2.0), "200%");
                    ui.selectable_value(&mut zoom, Some(4.0), "400%");
                });
            // An option was chosen! Emit the command to scale
            if let Some(zoom) = zoom {
                let _ = requests.send(requests::UiRequest::Document {
                    target: document,
                    request: requests::DocumentRequest::View(
                        requests::DocumentViewRequest::RealSize(zoom),
                    ),
                });
            }
            // Handle Scroll wheel
            // future: configurable scroll direction and speed.
            // FIXME: respect cursor position.
            let scroll_zoom_cmds = frame.action_trigger_count(crate::actions::Action::ZoomIn)
                as f32
                - frame.action_trigger_count(crate::actions::Action::ZoomOut) as f32;
            let _ = requests.send(requests::UiRequest::Document {
                target: document,
                request: requests::DocumentRequest::View(requests::DocumentViewRequest::ZoomBy(
                    1.25f32.powf(scroll_zoom_cmds),
                )),
            });

            ui.add(egui::Separator::default().vertical());

            //Rotate controls
            if ui.small_button("⟲").clicked() {
                let _ = requests.send(requests::UiRequest::Document {
                    target: document,
                    request: requests::DocumentRequest::View(
                        requests::DocumentViewRequest::RotateTo(0.0),
                    ),
                });
            };
            latch::latch(ui, (document, "rotation"), 0.0, |ui, rotation: &mut f32| {
                let before = *rotation;
                let rotation_response = ui.add(
                    egui::DragValue::new(rotation)
                        .speed(0.5)
                        .fixed_decimals(0)
                        .suffix('°'),
                );
                if rotation_response.changed() {
                    // Use a delta angle request
                    let delta = *rotation - before;
                    let _ = requests.send(requests::UiRequest::Document {
                        target: document,
                        request: requests::DocumentRequest::View(
                            requests::DocumentViewRequest::RotateBy(delta.to_radians()),
                        ),
                    });
                }
                if rotation_response.dragged() {
                    latch::Latch::Continue
                } else {
                    latch::Latch::None
                }
            });
            ui.add(egui::Separator::default().vertical());

            // Undo/redo - only show if there is a currently selected layer.
            let undo = egui::Button::new("⮪");
            let redo = egui::Button::new("⮫");

            // Accept undo/redo actions
            let mut undos = frame.action_trigger_count(crate::actions::Action::Undo);
            let mut redos = frame.action_trigger_count(crate::actions::Action::Redo);

            // RTL - add in reverse :P
            if ui.add(redo).clicked() {
                redos += 1;
            };
            if ui.add(undo).clicked() {
                undos += 1;
            };
            // Submit undo/redos as requested.
            if redos != 0 {
                crate::global::provider().inspect(document, |document| document.redo_n(redos));
            }
            if undos != 0 {
                crate::global::provider().inspect(document, |document| document.undo_n(undos));
            }
        });
    }

    fn colors_panel(
        &mut self,
        ui: &mut Ui,
        current_doc: Option<state::DocumentID>,
        actions: &crate::actions::ActionFrame,
    ) {
        use az::SaturatingAs;

        let mut globals = crate::AdHocGlobals::get().write();
        if let Some(brush) = globals.as_mut().map(|globals| &mut globals.brush) {
            if let Some(current_doc) = current_doc {
                // Show palette
                // Between AdHocGlobals and this, two locks are held. Recipe for a deadlock.
                crate::global::provider().inspect(current_doc, |doc| {
                    // Unfortunately this is the second `write_with` this frame. I need a way for this to work better..
                    // A retained mode UI is probably the solution as well as just a good idea for the future.
                    doc.write_with(|w| {
                        let mut palette = w.palette();

                        if std::mem::take(&mut self.picker_changed) {
                            let picker_color = egui::Rgba::from(self.picker_color);
                            brush.color_modulate =
                                fcolor::Color::from_array_lossy(picker_color.to_array())
                                    .unwrap_or(fcolor::Color::BLACK)
                                    .into();
                        }

                        // Small buttons with color history, pins, and palettes.
                        let palette_response = color_palette::ColorPalette::new(
                            &mut brush.color_modulate,
                            &mut palette,
                        )
                        .scope(color_palette::HistoryScope::Local)
                        .in_flux(self.picker_in_flux)
                        .id_source(current_doc)
                        .swap(
                            // Swap top colors if requested.
                            actions.action_trigger_count(crate::actions::Action::ColorSwap) % 2
                                == 1,
                        )
                        .max_history(64)
                        .show(ui);

                        // set the picker if the brush was changed by the palette UI.
                        // There's a really weird lifetime issue though where seemingly both mutable borrows above
                        // are intertwined and continue indefinitely - thus the color can't be read...?
                        // So instead, it has it's own return type that uses those borrows to find
                        // the color for us. Weird werid werid.
                        // Do this unconditionally - so that the picker is always sync'd, even if
                        // ssome code runs that changes the brushes modulate without announcing it.
                        let [r, g, b, a] = palette_response.dereferenced_color.as_array();
                        let rgba = egui::Rgba::from_rgba_premultiplied(r, g, b, a);

                        self.picker_color = rgba.into();
                    });
                });
            }

            ui.horizontal(|ui| {
                ui.label("Brush");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(PLUS_ICON.to_string()).clicked() {
                        self.modal = Some(CurrentModal::BrushCreation(
                            brush_ui::CreationModal::default(),
                        ));
                    }
                })
            });
            ui.separator();
            ui.selectable_value(&mut brush.brush.0[0], 0, "Test brush A");
            ui.selectable_value(&mut brush.brush.0[0], 1, "Test brush B");

            let mut size_mul = brush.size_mul.get();
            let mut spacing_px = brush.spacing_px.get();
            // Apply size up/down actions
            // - for down, + for up
            'size_steps: {
                let size_steps = actions
                    .action_trigger_count(crate::actions::Action::BrushSizeUp)
                    .saturating_as::<i32>()
                    .saturating_sub(
                        actions
                            .action_trigger_count(crate::actions::Action::BrushSizeDown)
                            .saturating_as(),
                    );
                if size_steps == 0 {
                    break 'size_steps;
                }
                // Usually editors supply some kind of snapping here to snap to
                // common values instead. Todo!
                let factor = 2.0f32.powf(size_steps as f32 / 4.0);
                size_mul *= factor;
                spacing_px *= factor;
            }
            ui.add(
                egui::Slider::new(&mut spacing_px, 0.25..=10.0)
                    .text("Spacing")
                    .suffix("px")
                    .max_decimals(2)
                    .clamp_to_range(false),
            );
            // Prevent negative
            spacing_px = spacing_px.max(0.1);
            ui.add(
                egui::Slider::new(&mut size_mul, spacing_px..=50.0)
                    .text("Size")
                    .suffix("px")
                    .max_decimals(2)
                    .clamp_to_range(false),
            );
            // Prevent negative
            size_mul = size_mul.max(0.1);

            if let Ok(size_mul) = FiniteF32::new(size_mul) {
                brush.size_mul = size_mul;
            }
            if let Ok(spacing_px) = FiniteF32::new(spacing_px) {
                brush.spacing_px = spacing_px;
            }
        }
    }
}
/// For any tool, `(icon string, tooltip, opt_hotkey)`
fn tool_button_for(
    tool: crate::pen_tools::StateLayer,
) -> (&'static str, &'static str, Option<crate::actions::Action>) {
    use crate::{actions::Action, pen_tools::StateLayer};
    match tool {
        StateLayer::Brush => (STROKE_LAYER_ICON, "Brush", Some(Action::Brush)),
        StateLayer::Picker => ("✒", "Picker", Some(Action::Picker)),
        StateLayer::Gizmos => ("⌖", "Gizmos", Some(Action::Gizmo)),
        StateLayer::Lasso => ("?", "Lasso", Some(Action::Lasso)),
        // NO action for these! pen_tools takes care of it without latching.
        // TODO: that's a weird mixing of roles lol
        StateLayer::Eraser => ("?", "Eraser", None),
        StateLayer::ViewportPan => ("✋", "Pan View", None),
        StateLayer::ViewportRotate => ("🔃", "Rotate View", None),
        StateLayer::ViewportScrub => ("🔍", "Scrub View", None),
    }
}
fn tools_panel(
    ui: &mut Ui,
    action_frame: &crate::actions::ActionFrame,
    requests: &crossbeam::channel::Sender<requests::UiRequest>,
) {
    use crate::pen_tools::StateLayer;
    const TOOL_GROUPS: [&[StateLayer]; 3] = [
        &[StateLayer::Brush, StateLayer::Eraser, StateLayer::Picker],
        &[StateLayer::Lasso, StateLayer::Gizmos],
        &[
            StateLayer::ViewportPan,
            StateLayer::ViewportRotate,
            StateLayer::ViewportScrub,
        ],
    ];
    // size, grows to justify
    const BTN_BASE_SIZE: f32 = 20.0;
    const ICON_SIZE: f32 = 15.0;
    // Margin
    const BTN_BASE_MARGIN: f32 = 5.0;

    let button_size = justify_mut(ui, JustifyAxis::Horizontal, BTN_BASE_SIZE, BTN_BASE_MARGIN);

    let spacing = ui.spacing_mut();
    spacing.interact_size = egui::Vec2::splat(button_size);
    spacing.button_padding = egui::Vec2::ZERO;

    let font_height = ICON_SIZE / ui.ctx().pixels_per_point();
    let font = egui::FontId::monospace(font_height);

    for tool_group in TOOL_GROUPS {
        ui.horizontal_wrapped(|ui| {
            for &tool in tool_group {
                let (icon, tooltip, opt_action) = tool_button_for(tool);

                let button = egui::Button::new(egui::RichText::new(icon).font(font.clone()))
                    .min_size(egui::Vec2::splat(button_size));
                // Add button. Trigger if button clicked or action occured.
                let response = ui.add(button).on_hover_text(tooltip);
                let response = if let Some(action) = opt_action {
                    response.or_action_clicked(action_frame, action)
                } else {
                    response
                };
                if response.clicked() {
                    let _ = requests.send(requests::UiRequest::SetBaseTool { tool });
                }
            }
        });
    }
}
/// Edit a leaf layer's data. If modifications were made that should be pushed to the queue,
/// `true` is returned.
fn leaf_props_panel(
    ui: &mut Ui,
    leaf_id: state::graph::LeafID,
    leaf: &mut state::graph::LeafType,
    stroke_collections: &state::stroke_collection::StrokeCollectionState,
) -> bool {
    use state::graph::LeafType;
    ui.separator();
    let write = match leaf {
        // Nothing to show
        LeafType::Note => false,
        // Color picker
        LeafType::SolidColor { source, .. } => {
            let mut globals = crate::AdHocGlobals::get().write();
            let mut current_color = globals.as_mut().map(|g| &mut g.brush.color_modulate);

            ui.horizontal(|ui| {
                ui.set_enabled(current_color.is_some());
                // Add a preview button that also allows the user to select this color.
                if ui
                    .add_enabled(
                        current_color.is_some(),
                        color_palette::ColorSquare {
                            // Todo: Display paletted.
                            color: source.get().left_or(fcolor::Color::BLACK),
                            icon: source.is_palette().then_some(PALETTE_ICON),
                            ..Default::default()
                        },
                    )
                    .on_hover_text("Use this color")
                    .clicked()
                {
                    match &mut current_color {
                        Some(current) => **current = *source,
                        // UI disabled if None, unreachable :D
                        None => unreachable!(),
                    }
                }
                ui.label("Fill color");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("Replace")
                        .on_hover_text("Replace layer color with active color")
                        .clicked()
                    {
                        match current_color {
                            Some(current) => *source = *current,
                            // UI disabled if None, unreachable :D
                            None => unreachable!(),
                        }
                        true
                    } else {
                        false
                    }
                })
                .inner
            })
            .inner
        }
        LeafType::StrokeLayer { collection, .. } => {
            // Nothing interactible, but display some infos
            ui.label(
                egui::RichText::new(format!(
                    "{} stroke items from {}",
                    stroke_collections
                        .get(*collection)
                        .map_or(0, |collection| collection.strokes.len()),
                    *collection,
                ))
                .italics()
                .weak(),
            );
            false
        }
        LeafType::Text {
            text, px_per_em, ..
        } => {
            let mut changed =
                latch::latch(ui, (leaf_id, "pix-per-em"), *px_per_em, |ui, px_per_em| {
                    let response = ui.add(
                        egui::Slider::new(px_per_em, 20.0..=2000.0)
                            .clamp_to_range(true)
                            .logarithmic(true),
                    );

                    // There is, as far as I can tell, no way to do this right.
                    if response.has_focus() {
                        return latch::Latch::Continue;
                    }
                    if response.drag_released() || response.lost_focus() {
                        return latch::Latch::Finish;
                    }
                    match (response.changed(), response.dragged()) {
                        (_, true) => latch::Latch::Continue,
                        (true, false) => latch::Latch::Finish,
                        (false, false) => latch::Latch::None,
                    }
                })
                .on_finish(|new_px_per_em| *px_per_em = new_px_per_em)
                .is_some();
            // Clones on every frame. Buh. bad.
            changed |= latch::latch(ui, (leaf_id, "text"), text.clone(), |ui, new_text| {
                let response = ui.text_edit_multiline(new_text);
                // todo: Latch::None on esc.
                match (response.lost_focus(), response.has_focus()) {
                    (true, _) => latch::Latch::Finish,
                    (false, true) => latch::Latch::Continue,
                    (false, false) => latch::Latch::None,
                }
            })
            .on_finish(|new_text| *text = new_text)
            .is_some();

            changed
        }
    };

    write
}
fn layer_buttons(
    ui: &mut Ui,
    interface: &mut PerDocumentData,
    writer: &mut queue::writer::CommandQueueWriter,
) {
    ui.horizontal(|ui| {
        // Have a button for adding a layer of any type.
        // Doing it with an enum in this way makes the lifetimes work (too many borrows if each button
        // had the layer addition inline) and we obviously don't expect two of these to be triggered in one frame!
        enum NewLayerType {
            Stroke,
            Text,
            Fill,
            Note,
            Group,
        }
        let new_layer_button = egui::ComboBox::from_id_source("layer-add")
            .selected_text(PLUS_ICON.to_string())
            // Minimize size to fit around the icon.
            .width(0.0)
            .show_ui(ui, |ui| {
                let mut selection = None;
                if ui
                    // A bit of an abuse of shortcut text. Screenreaders hate her!
                    // Unsure of how to better achieve the effect...
                    .add(egui::Button::new("Stroke Layer").shortcut_text(STROKE_LAYER_ICON))
                    .clicked()
                {
                    selection = Some(NewLayerType::Stroke);
                }
                if ui
                    .add_enabled(
                        false,
                        egui::Button::new("Text Layer").shortcut_text(TEXT_LAYER_ICON),
                    )
                    .clicked()
                {
                    selection = Some(NewLayerType::Text);
                }
                if ui
                    .add(egui::Button::new("Fill Layer").shortcut_text(FILL_LAYER_ICON))
                    .clicked()
                {
                    selection = Some(NewLayerType::Fill);
                }
                if ui
                    .add(egui::Button::new("Note").shortcut_text(NOTE_LAYER_ICON))
                    .clicked()
                {
                    selection = Some(NewLayerType::Note);
                }
                ui.separator();
                if ui
                    .add(egui::Button::new("Blend Group").shortcut_text(GROUP_ICON))
                    .clicked()
                {
                    selection = Some(NewLayerType::Group);
                }

                selection
            });

        new_layer_button.response.on_hover_text("Add layer");

        // Option<Option<...>>, but we only care when Some(Some(...))
        let new_layer = new_layer_button.inner.flatten();

        // Add the layer if one was requested.
        if let Some(new_layer) = new_layer {
            let addition_location = match interface.graph_selection.as_ref() {
                // If a group is selected, we insert as the topmost child.
                Some(state::graph::AnyID::Node(id)) => state::graph::Location::IndexIntoNode(id, 0),
                // Otherwise, we insert as the sibling directly above selected.
                Some(any) => state::graph::Location::AboveSelection(any),
                // No selection, add into the root of the viewed subree
                None => match interface.graph_focused_subtree.as_ref() {
                    Some(root) => state::graph::Location::IndexIntoNode(root, 0),
                    None => state::graph::Location::IndexIntoRoot(0),
                },
            };
            interface.graph_selection = match new_layer {
                NewLayerType::Stroke => {
                    let new_stroke_collection = writer.stroke_collections().insert();
                    writer
                        .graph()
                        .add_leaf(
                            state::graph::LeafType::StrokeLayer {
                                blend: Blend::default(),
                                collection: new_stroke_collection,
                            },
                            addition_location,
                            "Stroke Layer".to_string(),
                        )
                        .ok()
                        .map(Into::into)
                }
                NewLayerType::Fill => writer
                    .graph()
                    .add_leaf(
                        state::graph::LeafType::SolidColor {
                            blend: Blend::default(),
                            source: fcolor::ColorOrPalette::WHITE,
                        },
                        addition_location,
                        "Fill".to_string(),
                    )
                    .ok()
                    .map(Into::into),
                NewLayerType::Text => writer
                    .graph()
                    .add_leaf(
                        state::graph::LeafType::Text {
                            blend: Blend::default(),
                            text: "Hello, world!".to_owned(),
                            px_per_em: 50.0,
                        },
                        addition_location,
                        "Text".to_string(),
                    )
                    .ok()
                    .map(Into::into),
                NewLayerType::Note => writer
                    .graph()
                    .add_leaf(
                        state::graph::LeafType::Note,
                        addition_location,
                        "Note".to_string(),
                    )
                    .ok()
                    .map(Into::into),
                NewLayerType::Group => writer
                    .graph()
                    .add_node(
                        state::graph::NodeType::GroupedBlend(Blend::default()),
                        addition_location,
                        "Group".to_string(),
                    )
                    .ok()
                    .map(Into::into),
            };
        };

        let mut graph = writer.graph();

        ui.add(egui::Separator::default().vertical());

        let merge_button = egui::Button::new("⤵");
        ui.add_enabled(false, merge_button)
            .on_hover_text("Merge down");

        if ui
            .add_enabled(interface.graph_selection.is_some(), egui::Button::new("✖"))
            .on_hover_text("Delete layer")
            .clicked()
        {
            // Explicitly ignore error.
            let _ = graph.delete(interface.graph_selection.unwrap());
            interface.graph_selection = None;
        };
    });
}
/// Side panel showing layer add buttons, layer tree, and layer options
fn layers_panel(ui: &mut Ui, interface: &mut PerDocumentData) {
    crate::global::provider().inspect(interface.id, |queue| {
        queue.write_with(|writer| {
            let graph = writer.graph();
            // Node properties editor panel, at the bottom. Shown only when a node is selected.
            // Must occur before the graph rendering to prevent ui overflow :V
            let node_props = interface
                .graph_selection
                // Ignore if there is a yanked node.
                .and_then(|node| graph.get(node))
                .cloned();

            egui::TopBottomPanel::bottom("LayerProperties").show_animated_inside(
                ui,
                node_props.is_some(),
                |ui| {
                    // Unwraps OK - guarded by show condition.
                    let node_props = node_props.unwrap();
                    let id = interface.graph_selection.unwrap();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(icon_of_node(&node_props)).monospace());
                        ui.label(format!("{} properties", node_props.name()));
                    });
                    match id {
                        state::graph::AnyID::Leaf(leaf_id) => {
                            let Ok(mut leaf) = node_props.into_leaf() else {
                                // bad interface :V
                                // match on ID says it must be a leaf.
                                unreachable!();
                            };
                            if leaf_props_panel(
                                ui,
                                leaf_id,
                                &mut leaf,
                                &writer.stroke_collections(),
                            ) {
                                let _ = writer.graph().set_leaf(leaf_id, leaf);
                            }
                        }
                        state::graph::AnyID::Node(_n) => {
                            // Todo!
                        }
                    }
                },
            );

            layer_buttons(ui, interface, writer);

            let mut graph = writer.graph();

            // Strange visual flicker when this button is clicked,
            // as the header remains for one frame after the graph switches back.
            // This could become more problematic if the client doesn't refresh the UI
            // one frame after, as the header would stay but there's no subtree selected!
            // Eh, todo. :P
            if interface.graph_focused_subtree.is_some() {
                ui.horizontal(|ui| {
                    if ui.small_button("⬅").clicked() {
                        interface.graph_focused_subtree = None;
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "Subtree of {}",
                            interface
                                .graph_focused_subtree
                                .as_ref()
                                .and_then(|subtree| graph.get(*subtree))
                                .map_or("Unknown", |data| data.name())
                        ))
                        .italics(),
                    );
                });
            }
            latch::latch(ui, "dnd-state", None, |ui, dnd_state| {
                egui::ScrollArea::new([false, true])
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        graph_edit_recurse(
                            ui,
                            &mut graph,
                            interface.graph_focused_subtree,
                            &mut interface.graph_selection,
                            &mut interface.graph_focused_subtree,
                            dnd_state,
                        );
                    });

                match dnd_state {
                    None => latch::Latch::None,
                    Some(DndState {
                        released: false, ..
                    }) => latch::Latch::Continue,
                    Some(_) => latch::Latch::Finish,
                }
            })
            .on_finish(|dnd| {
                // Drop just occured. Do move!
                let Some(dnd) = dnd else {
                    return;
                };

                let Some(drop_target) = dnd.drop_target else {
                    return;
                };

                // It is OK if this err - likely, in fact. If the user eg. places it in the same place it already was,
                // or tries to make a group a child of itself
                let _ = graph.reparent(
                    dnd.drag_target,
                    match &drop_target {
                        DroppedAt::Before(before) => state::graph::Location::AboveSelection(before),
                        DroppedAt::FirstChild(parent) => {
                            state::graph::Location::IndexIntoNode(parent, 0)
                        }
                        // After - index too large is clamped to the end, perfect!
                        DroppedAt::LastChild(None) => {
                            state::graph::Location::IndexIntoRoot(usize::MAX)
                        }
                        DroppedAt::LastChild(Some(parent)) => {
                            state::graph::Location::IndexIntoNode(parent, usize::MAX)
                        }
                    },
                );
            })
        });
    });
}
/// Panel showing debug stats
fn stats_panel(ui: &mut Ui) {
    ui.label("Memory Usage Stats");
    let point_resident_usage = crate::global::points().resident_usage();
    ui.label(format!(
        "Point repository: {}/{}",
        human_bytes::human_bytes(point_resident_usage.0 as f64),
        human_bytes::human_bytes(point_resident_usage.1 as f64),
    ));
}

fn icon_of_node(node: &state::graph::NodeData) -> &'static str {
    use state::graph::{LeafType, NodeType};
    const UNKNOWN: &str = "？";
    match (node.leaf(), node.node()) {
        // Leaves
        (Some(LeafType::SolidColor { .. }), None) => FILL_LAYER_ICON,
        (Some(LeafType::StrokeLayer { .. }), None) => STROKE_LAYER_ICON,
        (Some(LeafType::Text { .. }), None) => TEXT_LAYER_ICON,
        (Some(LeafType::Note), None) => NOTE_LAYER_ICON,

        // Groups
        (None, Some(NodeType::Passthrough | NodeType::GroupedBlend(..))) => GROUP_ICON,
        // Invalid states
        (Some(..), Some(..)) | (None, None) => UNKNOWN,
    }
}
mod latch {

    pub enum Latch {
        /// The interaction is finished. State will be returned and deleted from persistant memory.
        Finish,
        /// The interaction is ongoing. State will not be reported, but will be persisted.
        Continue,
        /// The interaction is cancelled or hasn't started. State will not be reported nor persisted.
        None,
    }
    pub struct Response<'ui, State> {
        // Needed for cancellation functionality
        ui: &'ui mut egui::Ui,
        persisted_id: egui::Id,
        output: Option<State>,
    }
    impl<State: 'static> Response<'_, State> {
        /// Stop the interaction, preventing it from reporting Finished in the future.
        #[allow(dead_code)]
        pub fn cancel(self) {
            // Delete egui's persisted state
            self.ui
                .data_mut(|data| data.remove::<State>(self.persisted_id));
        }
        /// Get the output. Returns Some only once when the operation has finished.
        pub fn result(self) -> Option<State> {
            self.output
        }
        /// Takes the output, if any, and calls the closure if the latch interaction just finished.
        pub fn on_finish<R>(mut self, f: impl FnOnce(State) -> R) -> Option<R> {
            self.output.take().map(f)
        }
    }
    /// Interactible UI component where changes in-progress should be ignored,
    /// only producing output when the interaction is fully finished.
    ///
    /// Takes a closure which inspects the mutable State, modifying it and reporting changes via
    /// the [Latch] enum. By default, when no interaction is occuring, it should report [`Latch::None`]
    pub fn latch<State, F>(
        ui: &mut egui::Ui,
        id_src: impl std::hash::Hash,
        state: State,
        f: F,
    ) -> Response<'_, State>
    where
        F: FnOnce(&mut egui::Ui, &mut State) -> Latch,
        // bounds implied by insert_temp
        State: 'static + Clone + std::any::Any + Send + Sync,
    {
        let persisted_id = egui::Id::new(id_src);
        let mut mutable_state = ui
            .data(|data| data.get_temp::<State>(persisted_id))
            .unwrap_or(state);
        let fn_response = f(ui, &mut mutable_state);

        match fn_response {
            // Intern the state.
            Latch::Continue => {
                ui.data_mut(|data| data.insert_temp::<State>(persisted_id, mutable_state));
                Response {
                    ui,
                    persisted_id,
                    output: None,
                }
            }
            // Return the state and clear it
            Latch::Finish => {
                ui.data_mut(|data| data.remove::<State>(persisted_id));
                Response {
                    ui,
                    persisted_id,
                    output: Some(mutable_state),
                }
            }
            // Nothing to do, clear it if it exists.
            Latch::None => {
                ui.data_mut(|data| data.remove::<State>(persisted_id));
                Response {
                    ui,
                    persisted_id,
                    output: None,
                }
            }
        }
    }
}

/// Inline UI component for changing a non-optional blend. Makes a copy of the blend internally,
/// only returning a new one on change (skipping partial changes like a dragging slider), returning None otherwise.
fn ui_layer_blend(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    blend: Blend,
    disable: bool,
) -> self::latch::Response<'_, Blend> {
    // Get the persisted blend, or use the caller's blend if none.
    latch::latch(ui, (&id, "blend-state"), blend, |ui, blend| {
        let mut finished = false;
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.set_enabled(!disable);
            finished |= ui
                .toggle_value(
                    &mut blend.alpha_clip,
                    egui::RichText::new(ALPHA_ICON).monospace().strong(),
                )
                .on_hover_text("Alpha clip")
                .clicked();

            // do NOT report "finished" mid-drag, only when it's complete!
            let response = ui.add(
                egui::DragValue::new(&mut blend.opacity)
                    .fixed_decimals(2)
                    .speed(0.01)
                    .clamp_range(0.0..=1.0),
            );
            changed |= response.dragged();
            // Bug: This reports a release on every frame when dragged and an egui
            // modal (eg, the combobox below) is open. wh y
            finished |= response.drag_released();

            egui::ComboBox::new(&id, "")
                .selected_text(blend.mode.as_ref())
                .show_ui(ui, |ui| {
                    for blend_mode in <BlendMode as strum::IntoEnumIterator>::iter() {
                        finished |= ui
                            .selectable_value(&mut blend.mode, blend_mode, blend_mode.as_ref())
                            .clicked();
                    }
                });
        });
        match (finished, changed) {
            (true, _) => latch::Latch::Finish,
            (false, true) => latch::Latch::Continue,
            (false, false) => latch::Latch::None,
        }
    })
}
/// Inline UI component for changing an optional (possibly passthrough) blend. Makes a copy of the blend internally,
/// only returning a new one on change (skipping partial changes like a dragging slider), returning None otherwise.
fn ui_passthrough_or_blend(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    blend: Option<Blend>,
    disable: bool,
) -> self::latch::Response<'_, Option<Blend>> {
    latch::latch(ui, (&id, "blend-state"), blend, |ui, blend| {
        let mut finished = false;
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.set_enabled(!disable);
            if let Some(blend) = blend.as_mut() {
                changed |= ui
                    .toggle_value(
                        &mut blend.alpha_clip,
                        egui::RichText::new(ALPHA_ICON).monospace().strong(),
                    )
                    .on_hover_text("Alpha clip")
                    .changed();
                finished |= changed;
                // do NOT report "finished" mid-drag, only when it's complete!
                let response = ui.add(
                    egui::DragValue::new(&mut blend.opacity)
                        .fixed_decimals(2)
                        .speed(0.01)
                        .clamp_range(0.0..=1.0),
                );
                changed |= response.dragged();
                // Bug: This reports a release on every frame when dragged and an egui
                // modal (eg, the combobox below) is open. wh y
                finished |= response.drag_released();
            };

            egui::ComboBox::new(&id, "")
                .selected_text(
                    blend
                        .map(|blend| blend.mode.as_ref().to_string())
                        .unwrap_or("Passthrough".to_string()),
                )
                .show_ui(ui, |ui| {
                    changed |= ui.selectable_value(blend, None, "Passthrough").clicked();
                    ui.separator();
                    for blend_mode in <BlendMode as strum::IntoEnumIterator>::iter() {
                        let select_value = Some(Blend {
                            mode: blend_mode,
                            // Set the blend to itself with new mode,
                            // or default fields if blend is None.
                            ..blend.unwrap_or_default()
                        });
                        changed |= ui
                            .selectable_value(blend, select_value, blend_mode.as_ref())
                            .clicked();
                    }
                    // All of these changes are considered finishing.
                    finished |= changed;
                });
        });

        match (finished, changed) {
            (true, _) => latch::Latch::Finish,
            (false, true) => latch::Latch::Continue,
            (false, false) => latch::Latch::None,
        }
    })
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum DroppedAt {
    // Last child of group, or None for last child of root.
    LastChild(Option<state::graph::NodeID>),
    /// Sibling above id
    Before(state::graph::AnyID),
    /// First child of id
    FirstChild(state::graph::NodeID),
}

#[derive(PartialEq, Eq, Clone, Copy)]
struct DndState {
    drag_target: state::graph::AnyID,
    drop_target: Option<DroppedAt>,
    released: bool,
}
impl DndState {
    fn drag_target(&self) -> state::graph::AnyID {
        self.drag_target
    }
    fn drop_target(&self) -> Option<DroppedAt> {
        self.drop_target
    }
}

fn graph_edit_recurse<
    // Well that's.... not great...
    W: queue::writer::CommandWrite<state::graph::commands::Command>,
>(
    ui: &mut Ui,
    graph: &mut state::graph::writer::GraphWriter<'_, W>,
    parent: Option<state::graph::NodeID>,
    selected_node: &mut Option<state::graph::AnyID>,
    focused_node: &mut Option<state::graph::NodeID>,
    dnd_state: &mut Option<DndState>,
) {
    let node_ids: Vec<_> = match parent {
        Some(root) => graph.iter_node(root).unwrap().map(|(id, _)| id).collect(),
        None => graph.iter_top_level().map(|(id, _)| id).collect(),
    };

    // Keep track of who the last child was.
    let mut is_empty = true;
    // Iterate!
    for id in node_ids {
        is_empty = false;

        let dnd_target = DroppedAt::Before(id);
        // Drag-n-drop target for inserting above this node
        let head_separator = drag::DropSeparator {
            active: dnd_state.is_some(),
            already_selected: dnd_state.as_ref().and_then(DndState::drop_target)
                == Some(dnd_target),
        };
        if head_separator.show(ui).selected {
            // Unwrap OK - isn't available for selection if None.
            dnd_state.as_mut().unwrap().drop_target = Some(dnd_target);
        };

        // Name and selection
        let header_response = ui.horizontal(|ui| {
            let data = graph.get(id).unwrap();

            // Disable everything if dragging a layer around.
            ui.set_enabled(dnd_state.is_none());

            // Drag-n-drop handle
            let dragged = ui
                .add(drag::Handle)
                .on_hover_text("Drag to reorder")
                .dragged();
            if !dragged && dnd_state.is_some_and(|dnd| dnd.drag_target == id) {
                // No longer dragged and we were the target, end drag.
                dnd_state.as_mut().unwrap().released = true;
            } else if dragged && dnd_state.is_none() {
                // Start drag!
                *dnd_state = Some(DndState {
                    drag_target: id,
                    drop_target: None,
                    released: false,
                });
            }

            // Choose an icon based on the type of the node
            let icon = icon_of_node(data);
            // Selection radio button + toggle function.
            let is_selected = *selected_node == Some(id);
            if ui
                .selectable_label(is_selected, egui::RichText::new(icon).monospace())
                .clicked()
            {
                if is_selected {
                    *selected_node = None;
                } else {
                    *selected_node = Some(id);
                }
            }

            let name = graph.name_mut(id).unwrap();

            // Fetch from last frame - are we hovered?
            let name_hovered_key = egui::Id::new((id, "name-hovered"));
            let hovered: Option<bool> = ui.data(|data| data.get_temp(name_hovered_key));
            let edit = egui::TextEdit::singleline(name).frame(hovered.unwrap_or(false));
            let name_response = ui.add(edit);

            // Send data to next frame, to tell that we're hovered or not.
            let interacted = name_response.has_focus() || name_response.hovered();
            ui.data_mut(|data| data.insert_temp(name_hovered_key, interacted));

            // Forward the response of the header items for right clicks, as it takes up all the click area!
            name_response
        });
        let data = graph.get(id).unwrap();
        // Type-specific UI elements
        match (data.leaf(), data.node()) {
            (Some(_), None) => {
                // Blend, if any.
                if let Some(old_blend) = data.blend() {
                    // Reports new blend when interaction is finished, disabled in yank mode.
                    ui_layer_blend(ui, (&id, "blend"), old_blend, dnd_state.is_some())
                        .on_finish(|new_blend| graph.change_blend(id, new_blend).unwrap());
                }
            }
            (None, Some(n)) => {
                // Unwrap nodeID:
                let state::graph::AnyID::Node(node_id) = id else {
                    panic!("Node data and ID mismatch!")
                };
                // Option to focus this subtree:
                header_response.inner.context_menu(|ui| {
                    if ui.button("Focus Subtree").clicked() {
                        *focused_node = Some(node_id);
                    }
                });
                // Display node type - passthrough or grouped blend
                let old_blend = n.blend();
                // Reports new blend when interaction finished, disabled in yank mode.
                ui_passthrough_or_blend(ui, (&id, "blend"), old_blend, dnd_state.is_some())
                    .on_finish(|new_blend| match (old_blend, new_blend) {
                        (Some(from), Some(to)) if from != to => {
                            // Simple blend change
                            graph.change_blend(id, to).unwrap();
                        }
                        (None, Some(to)) => {
                            // Type change - passthrough to grouped.
                            graph
                                .set_node(node_id, state::graph::NodeType::GroupedBlend(to))
                                .unwrap();
                        }
                        (Some(_), None) => {
                            // Type change - grouped to passthrough
                            graph
                                .set_node(node_id, state::graph::NodeType::Passthrough)
                                .unwrap();
                        }
                        _ => {
                            // No change
                        }
                    });

                // display children!
                egui::CollapsingHeader::new(egui::RichText::new("Children").italics().weak())
                    .id_source(id)
                    .default_open(true)
                    .show(ui, |ui| {
                        graph_edit_recurse(
                            ui,
                            graph,
                            Some(node_id),
                            selected_node,
                            focused_node,
                            dnd_state,
                        );
                    });
            }
            (None, None) => (),
            (Some(_), Some(_)) => panic!("Node is both a leaf and node???"),
        }
    }

    if is_empty {
        // Show a hint to add stuff.
        // this also occurs when an empty group is unrolled, show a drop target for first-child of this group.
        if let Some(group) = parent {
            let target = DroppedAt::FirstChild(group);
            let first_child_separator = drag::DropSeparator {
                active: dnd_state.is_some(),
                already_selected: dnd_state.as_ref().and_then(DndState::drop_target)
                    == Some(target),
            };
            if first_child_separator.show(ui).selected {
                // Unwrap OK - isn't available for selection if None.
                dnd_state.as_mut().unwrap().drop_target = Some(target);
            };
        }

        ui.label(
            egui::RichText::new("Nothing here... Add some layers!")
                .italics()
                .weak(),
        );
    } else {
        let target = DroppedAt::LastChild(parent);
        // Show a drag-n-drop target at the footer too, below the last target.
        let tail_separator = drag::DropSeparator {
            active: dnd_state.is_some(),
            already_selected: dnd_state.as_ref().and_then(DndState::drop_target) == Some(target),
        };
        if tail_separator.show(ui).selected {
            // Unwrap OK - isn't available for selection if None.
            dnd_state.as_mut().unwrap().drop_target = Some(target);
        };
    }
}

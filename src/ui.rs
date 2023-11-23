use crate::commands::queue::state_reader::CommandQueueStateReader;

pub mod requests;

const STROKE_LAYER_ICON: &'static str = "✏";
const NOTE_LAYER_ICON: &'static str = "🖹";
const FILL_LAYER_ICON: &'static str = "⬛";
const GROUP_ICON: &'static str = "🗀";
const SCISSOR_ICON: &'static str = "✂";

trait UILayer {
    /// Perform UI operations. If `is_background`, a layer is open above this layer - display
    /// as greyed out and ignore input!
    /// Return true to dismiss.
    fn do_ui(
        &mut self,
        ctx: &egui::Context,
        requests_channel: &mut requests::RequestSender,
        is_background: bool,
    ) -> bool;
}
struct PerDocumentData {
    id: crate::state::DocumentID,
    graph_selection: Option<crate::state::graph::AnyID>,
    graph_focused_subtree: Option<crate::state::graph::NodeID>,
    /// Yanked node during a reparent operation
    yanked_node: Option<crate::state::graph::AnyID>,
    name: String,
}
pub struct MainUI {
    // Could totally be static dispatch, but for simplicity:
    /// Displayed on top of everything, disabling all behind it in a stack-like manner
    modals: Vec<Box<dyn UILayer>>,
    /// Displayed as windows on the document viewport.
    inlays: Vec<Box<dyn UILayer>>,

    documents: Vec<PerDocumentData>,
    cur_document: Option<crate::state::DocumentID>,

    requests_send: requests::RequestSender,
    action_listener: crate::actions::ActionListener,
}
impl MainUI {
    pub fn new(
        requests_send: requests::RequestSender,
        action_listener: crate::actions::ActionListener,
    ) -> Self {
        let documents = crate::default_provider().document_iter();
        let documents: Vec<_> = documents
            .map(|id| PerDocumentData {
                id,
                graph_focused_subtree: None,
                graph_selection: None,
                yanked_node: None,
                name: "Unknown".into(),
            })
            .collect();
        let cur_document = documents.last().map(|doc| doc.id);
        Self {
            modals: vec![],
            inlays: vec![],
            documents,
            cur_document,
            requests_send,
            action_listener,
        }
    }
    /// Main UI and any modals, with the top bar, layers, brushes, color, etc. To be displayed in front of the document and it's gizmos.
    /// Returns the size of the document's viewport space - that is, the size of the rect not covered by any side/top/bottom panels.
    pub fn ui(&mut self, ctx: &egui::Context) -> (ultraviolet::Vec2, ultraviolet::Vec2) {
        // Display modals before main. Egui will place the windows without regard for free area.
        let mut is_background = false;
        for modal in self.modals.iter_mut().rev() {
            // todo: handle dismiss
            let _ = modal.do_ui(ctx, &mut self.requests_send, is_background);
            // Show all remaining elements as background.
            is_background = true;
        }
        // is_background set if at least one modal exists.
        let ret_value = self.main_ui(ctx, is_background);
        // Display windows after main. Egui will place the windows inside the free area
        for inlay in self.inlays.iter_mut() {
            // todo: handle dismiss
            let _ = inlay.do_ui(ctx, &mut self.requests_send, is_background);
        }

        ret_value
    }
    /// Render just self. Modals and insets handled separately.
    fn main_ui(
        &mut self,
        ctx: &egui::Context,
        is_background: bool,
    ) -> (ultraviolet::Vec2, ultraviolet::Vec2) {
        egui::TopBottomPanel::top("file").show(&ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("🐑").font(egui::FontId::proportional(20.0)))
                    .on_hover_text("Baa");
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        let add_button = |ui: &mut egui::Ui, label, shortcut| -> egui::Response {
                            let mut button = egui::Button::new(label);
                            if let Some(shortcut) = shortcut {
                                button = button.shortcut_text(shortcut);
                            }
                            ui.add(button)
                        };
                        if add_button(ui, "New", Some("Ctrl+N")).clicked() {
                            let new_doc = crate::commands::queue::DocumentCommandQueue::new();
                            let new_id = new_doc.id();
                            // Can't fail
                            let _ = crate::default_provider().insert(new_doc);
                            let interface = PerDocumentData {
                                id: new_id,
                                graph_focused_subtree: None,
                                graph_selection: None,
                                yanked_node: None,
                                name: "Unknown".into(),
                            };
                            let _ = self.requests_send.send(requests::UiRequest::Document {
                                target: new_id,
                                request: requests::DocumentRequest::Opened,
                            });
                            self.cur_document = Some(new_id);
                            self.documents.push(interface);
                        };
                        if add_button(ui, "Save", Some("Ctrl+S")).clicked() {
                            // Dirty testing implementation!
                            if let Some(current) = self.cur_document {
                                std::thread::spawn(move || -> () {
                                    if let Some(reader) = crate::default_provider()
                                        .inspect(current, |doc| doc.peek_clone_state())
                                    {
                                        let repo = crate::repositories::points::global();

                                        let try_block = || -> anyhow::Result<()> {
                                            let mut path = dirs::document_dir().unwrap();
                                            path.push("temp.fzp");
                                            let file = std::fs::File::create(path)?;

                                            let start = std::time::Instant::now();
                                            crate::io::write_into(reader, repo, &file)?;
                                            let duration = std::time::Instant::now() - start;

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
                        let _ = add_button(ui, "Save as", Some("Ctrl+Shift+S"));
                        if add_button(ui, "Open", Some("Ctrl+O")).clicked() {
                            // Synchronous and bad just for testing.
                            if let Some(files) = rfd::FileDialog::new().pick_files() {
                                let point_repository = crate::repositories::points::global();
                                let provider = crate::default_provider();

                                // Keep track of the last successful loaded id
                                let mut recent_success = None;
                                for file in files.into_iter() {
                                    match crate::io::read_path(file, point_repository) {
                                        Ok(doc) => {
                                            let id = doc.id();
                                            if provider.insert(doc).is_ok() {
                                                recent_success = Some(id);
                                            }
                                        }
                                        Err(e) => log::error!("Failed to load: {e:#}"),
                                    }
                                }
                                // Select last one, if any succeeded.
                                if let Some(new_doc) = recent_success {
                                    self.documents.push(PerDocumentData {
                                        id: new_doc,
                                        graph_focused_subtree: None,
                                        graph_selection: None,
                                        yanked_node: None,
                                        name: "Unknown".into(),
                                    });
                                    self.cur_document = Some(new_doc);
                                }
                            }
                        }
                        let _ = add_button(ui, "Open as new", None);
                        let _ = add_button(ui, "Export", None);
                    });
                    ui.menu_button("Edit", |_| ());
                    ui.menu_button("Image", |ui| {
                        if ui.button("Image Size").clicked() {
                            /*self.push_modal(|ui| {
                                ui.label("Hai :>");
                            });*/
                        };
                    });
                });
            });
        });
        egui::TopBottomPanel::bottom("Nav").show(&ctx, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Everything here is shown in reverse order!

                ui.label(":V");

                // if there is no current document, there is nothing for us to do here
                let Some(document) = self.cur_document else {
                    return;
                };

                // Find the document's interface
                let Some(interface) = self
                    .documents
                    .iter()
                    .find(|interface| interface.id == document)
                else {
                    // Not found! Reset cur_document.
                    self.cur_document = None;
                    return;
                };

                ui.add(egui::Separator::default().vertical());

                // Todo: Some actions (scale slider, rotation slider, etc) are impossible to implement as of now
                // as the rest of the world has no way to communicate into the UI (so, no reporting of current transform)

                //Zoom controls
                if ui.small_button("⟲").clicked() {
                    let _ = self.requests_send.send(requests::UiRequest::Document {
                        target: document,
                        request: requests::DocumentRequest::View(
                            requests::DocumentViewRequest::Fit,
                        ),
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
                    let _ = self.requests_send.send(requests::UiRequest::Document {
                        target: document,
                        request: requests::DocumentRequest::View(
                            requests::DocumentViewRequest::RealSize(zoom),
                        ),
                    });
                }

                ui.add(egui::Separator::default().vertical());

                //Rotate controls
                if ui.small_button("⟲").clicked() {
                    let _ = self.requests_send.send(requests::UiRequest::Document {
                        target: document,
                        request: requests::DocumentRequest::View(
                            requests::DocumentViewRequest::SetRotation(0.0),
                        ),
                    });
                };
                latch::latch(ui, (document, "rotation"), 0.0, |ui, rotation| {
                    let rotation_response = ui.drag_angle(rotation);
                    if rotation_response.changed() {
                        let _ = self.requests_send.send(requests::UiRequest::Document {
                            target: document,
                            request: requests::DocumentRequest::View(
                                requests::DocumentViewRequest::SetRotation(
                                    *rotation % std::f32::consts::TAU,
                                ),
                            ),
                        });
                    }
                    if !rotation_response.dragged() {
                        latch::Latch::None
                    } else {
                        latch::Latch::Continue
                    }
                });
                ui.add(egui::Separator::default().vertical());

                // Undo/redo - only show if there is a currently selected layer.
                let undo = egui::Button::new("⮪");
                let redo = egui::Button::new("⮫");

                if let Some(current_document) = self.cur_document {
                    // Accept undo/redo actions
                    let (mut undos, mut redos) = self
                        .action_listener
                        .frame()
                        .map(|frame| {
                            (
                                frame.action_trigger_count(crate::actions::Action::Undo),
                                frame.action_trigger_count(crate::actions::Action::Redo),
                            )
                        })
                        .unwrap_or((0, 0));
                    // RTL - add in reverse :P
                    if ui.add(redo).clicked() {
                        redos += 1
                    };
                    if ui.add(undo).clicked() {
                        undos += 1
                    };
                    // Submit undo/redos as requested.
                    if redos != 0 {
                        crate::default_provider()
                            .inspect(current_document, |document| document.redo_n(redos));
                    }
                    if undos != 0 {
                        crate::default_provider()
                            .inspect(current_document, |document| document.undo_n(undos));
                    }
                } else {
                    // RTL - add in reverse :P
                    ui.add_enabled(false, redo);
                    ui.add_enabled(false, undo);
                }
            });
        });

        egui::SidePanel::right("Layers").show(&ctx, |ui| {
            ui.label("Layers");
            ui.separator();
            // if there is no current document, there is nothing for us to do here
            let Some(document) = self.cur_document else {
                return;
            };

            // Find the document's interface
            let Some(interface) = self
                .documents
                .iter_mut()
                .find(|interface| interface.id == document)
            else {
                // Not found! Reset cur_document.
                self.cur_document = None;
                *crate::AdHocGlobals::get().write() = None;
                return;
            };

            crate::default_provider().inspect(interface.id, |queue| {
                queue.write_with(|writer| {
                    let graph = writer.graph();
                    // Node properties editor panel, at the bottom. Shown only when a node is selected.
                    // Must occur before the graph rendering to prevent ui overflow :V
                    let node_props = interface
                        .graph_selection
                        // Ignore if there is a yanked node.
                        .filter(|_| interface.yanked_node.is_none())
                        .and_then(|node| graph.get(node))
                        .cloned();

                    egui::TopBottomPanel::bottom("LayerProperties").show_animated_inside(
                        ui,
                        node_props.is_some(),
                        |ui| {
                            // Guarded by panel show condition
                            let node_props = node_props.unwrap();
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(icon_of_node(&node_props)).monospace(),
                                );
                                ui.label(format!("{} properties", node_props.name()));
                            });
                            ui.separator();
                            if let Ok(mut leaf) = node_props.into_leaf() {
                                use crate::state::graph::LeafType;
                                let write = match &mut leaf {
                                    // Nothing to show
                                    LeafType::Note => false,
                                    // Color picker
                                    LeafType::SolidColor { source, .. } => {
                                        let color_latch = ui
                                            .horizontal(|ui| {
                                                ui.label("Fill color:");
                                                // Latch onto fill color, to submit update only when selection is finished.
                                                latch::latch(
                                                    ui,
                                                    (&interface.graph_selection, "fill-color"),
                                                    *source,
                                                    |ui, [r, g, b, a]| {
                                                        let mut rgba =
                                                            egui::Rgba::from_rgba_premultiplied(
                                                                *r, *g, *b, *a,
                                                            );
                                                        egui::color_picker::color_edit_button_rgba(
                                                            ui,
                                                            &mut rgba,
                                                            // If the user wants Add they should use the blend modes mwuahaha
                                                            egui::color_picker::Alpha::OnlyBlend,
                                                        );
                                                        *r = rgba.r();
                                                        *b = rgba.b();
                                                        *g = rgba.g();
                                                        *a = rgba.a();

                                                        // None of the response fields for color pickers seem to indicate
                                                        // a finished interaction TwT
                                                        if ui.button("Apply").clicked() {
                                                            latch::Latch::Finish
                                                        } else {
                                                            latch::Latch::Continue
                                                        }
                                                    },
                                                )
                                                .result()
                                            })
                                            .inner;

                                        if let Some(color) = color_latch {
                                            *source = color;
                                            true
                                        } else {
                                            false
                                        }
                                    }
                                    LeafType::StrokeLayer { collection, .. } => {
                                        // Nothing interactible, but display some infos
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{} stroke items from {}",
                                                writer
                                                    .stroke_collections()
                                                    .get(*collection)
                                                    .map(|collection| collection.strokes.len())
                                                    .unwrap_or(0),
                                                *collection,
                                            ))
                                            .italics()
                                            .weak(),
                                        );
                                        false
                                    }
                                };

                                if write {
                                    if let Some(crate::state::graph::AnyID::Leaf(id)) =
                                        interface.graph_selection
                                    {
                                        let _ = writer.graph().set_leaf(id, leaf);
                                    }
                                }
                            }
                        },
                    );
                    // Buttons
                    ui.horizontal(|ui| {
                        // Copied logic since we can't borrow test_graph_selection throughout this whole
                        // ui section
                        macro_rules! add_location {
                            () => {
                                match interface.graph_selection.as_ref() {
                                    Some(crate::state::graph::AnyID::Node(id)) => {
                                        crate::state::graph::Location::IndexIntoNode(id, 0)
                                    }
                                    Some(any) => crate::state::graph::Location::AboveSelection(any),
                                    // No selection, add into the root of the viewed subree
                                    None => match interface.graph_focused_subtree.as_ref() {
                                        Some(root) => {
                                            crate::state::graph::Location::IndexIntoNode(root, 0)
                                        }
                                        None => crate::state::graph::Location::IndexIntoRoot(0),
                                    },
                                }
                            };
                        }

                        if ui
                            .button(STROKE_LAYER_ICON)
                            .on_hover_text("Add Stroke Layer")
                            .clicked()
                        {
                            let new_stroke_collection = writer.stroke_collections().insert();
                            interface.graph_selection = writer
                                .graph()
                                .add_leaf(
                                    crate::state::graph::LeafType::StrokeLayer {
                                        blend: Default::default(),
                                        collection: new_stroke_collection,
                                    },
                                    add_location!(),
                                    "Stroke Layer".to_string(),
                                )
                                .ok()
                                .map(Into::into);
                        }
                        // Borrow graph for the rest of the time.
                        let mut graph = writer.graph();
                        if ui
                            .button(NOTE_LAYER_ICON)
                            .on_hover_text("Add Note")
                            .clicked()
                        {
                            interface.graph_selection = graph
                                .add_leaf(
                                    crate::state::graph::LeafType::Note,
                                    add_location!(),
                                    "Note".to_string(),
                                )
                                .ok()
                                .map(Into::into);
                        }
                        if ui
                            .button(FILL_LAYER_ICON)
                            .on_hover_text("Add Fill Layer")
                            .clicked()
                        {
                            interface.graph_selection = graph
                                .add_leaf(
                                    crate::state::graph::LeafType::SolidColor {
                                        blend: Default::default(),
                                        source: [1.0; 4],
                                    },
                                    add_location!(),
                                    "Fill".to_string(),
                                )
                                .ok()
                                .map(Into::into);
                        }

                        ui.add(egui::Separator::default().vertical());

                        if ui.button(GROUP_ICON).on_hover_text("Add Group").clicked() {
                            interface.graph_selection = graph
                                .add_node(
                                    crate::state::graph::NodeType::Passthrough,
                                    add_location!(),
                                    "Group Layer".to_string(),
                                )
                                .ok()
                                .map(Into::into);
                        };

                        ui.add(egui::Separator::default().vertical());

                        let merge_button = egui::Button::new("⤵");
                        ui.add_enabled(false, merge_button);

                        let delete_button = egui::Button::new("✖");
                        if let Some(selection) = interface.graph_selection {
                            if ui.add_enabled(true, delete_button).clicked() {
                                // Explicitly ignore error.
                                let _ = graph.delete(selection);
                                interface.graph_selection = None;
                            }
                        } else {
                            ui.add_enabled(false, delete_button);
                        }
                        if let Some(yanked) = interface.yanked_node {
                            // Display an insert button
                            if ui
                                .button(egui::RichText::new("↪").monospace())
                                .on_hover_text("Insert yanked node here")
                                .clicked()
                            {
                                // Explicitly ignore error. There are many invalid options, in that case do nothing!
                                let _ = graph.reparent(yanked, add_location!());
                                interface.yanked_node = None;
                            }
                        } else {
                            // Display a cut button
                            let button =
                                egui::Button::new(egui::RichText::new(SCISSOR_ICON).monospace());
                            // Disable if no selection.
                            if ui
                                .add_enabled(interface.graph_selection.is_some(), button)
                                .on_hover_text("Reparent")
                                .clicked()
                            {
                                interface.yanked_node = interface.graph_selection;
                            }
                        }
                    });

                    ui.separator();
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
                                        .and_then(|subtree| graph.get(subtree.clone()))
                                        .map(|data| data.name())
                                        .unwrap_or("Unknown")
                                ))
                                .italics(),
                            );
                        });
                    }
                    egui::ScrollArea::new([false, true])
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            graph_edit_recurse(
                                ui,
                                &mut graph,
                                interface.graph_focused_subtree.clone(),
                                &mut interface.graph_selection,
                                &mut interface.graph_focused_subtree,
                                &mut interface.yanked_node,
                            )
                        });
                })
            });

            // Update selections.
            let mut globals = crate::AdHocGlobals::get().write();
            let old_brush = globals.take().map(|globals| globals.brush);
            *globals = Some(crate::AdHocGlobals {
                document,
                brush: old_brush.unwrap_or(crate::state::StrokeBrushSettings {
                    is_eraser: false,
                    brush: crate::brush::todo_brush().id(),
                    color_modulate: [0.0, 0.0, 0.0, 1.0],
                    size_mul: 10.0,
                    spacing_px: 0.5,
                }),
                node: interface.graph_selection,
            });
        });

        egui::SidePanel::left("Inspector")
            .resizable(false)
            .show(&ctx, |ui| {
                egui::TopBottomPanel::bottom("mem-usage")
                    .resizable(false)
                    .show_inside(ui, |ui| {
                        ui.label("Memory Usage Stats");
                        let point_resident_usage =
                            crate::repositories::points::global().resident_usage();
                        ui.label(format!(
                            "Point repository: {}/{}",
                            human_bytes::human_bytes(point_resident_usage.0 as f64),
                            human_bytes::human_bytes(point_resident_usage.1 as f64),
                        ));
                    });
                let mut globals = crate::AdHocGlobals::get().write();
                if let Some(brush) = globals.as_mut().map(|globals| &mut globals.brush) {
                    ui.label("Color");
                    ui.separator();
                    // Why..
                    let mut color = egui::Rgba::from_rgba_premultiplied(
                        brush.color_modulate[0],
                        brush.color_modulate[1],
                        brush.color_modulate[2],
                        brush.color_modulate[3],
                    );
                    if egui::color_picker::color_edit_button_rgba(
                        ui,
                        &mut color,
                        egui::color_picker::Alpha::OnlyBlend,
                    )
                    .changed()
                    {
                        brush.color_modulate = color.to_array();
                    };
                    ui.label("Brush");
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut brush.spacing_px, 0.25..=10.0)
                            .text("Spacing")
                            .suffix("px")
                            .max_decimals(2)
                            .clamp_to_range(false),
                    );
                    // Prevent negative
                    brush.spacing_px = brush.spacing_px.max(0.1);
                    ui.add(
                        egui::Slider::new(&mut brush.size_mul, brush.spacing_px..=50.0)
                            .text("Size")
                            .suffix("px")
                            .max_decimals(2)
                            .clamp_to_range(false),
                    );
                    // Prevent negative
                    brush.size_mul = brush.size_mul.max(0.1);
                }
            });

        egui::TopBottomPanel::top("documents").show(&ctx, |ui| {
            egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal(|ui| {
                    let mut deleted_ids =
                        smallvec::SmallVec::<[crate::state::DocumentID; 1]>::new();
                    for PerDocumentData { id, name, .. } in self.documents.iter() {
                        egui::containers::Frame::group(ui.style())
                            .outer_margin(egui::Margin::symmetric(0.0, 0.0))
                            .inner_margin(egui::Margin::symmetric(0.0, 0.0))
                            .multiply_with_opacity(if self.cur_document == Some(*id) {
                                1.0
                            } else {
                                0.0
                            })
                            .rounding(egui::Rounding {
                                ne: 2.0,
                                nw: 2.0,
                                ..0.0.into()
                            })
                            .show(ui, |ui| {
                                ui.selectable_value(&mut self.cur_document, Some(*id), name);
                                if ui.small_button("✖").clicked() {
                                    deleted_ids.push(*id);
                                    //Disselect if deleted.
                                    if self.cur_document == Some(*id) {
                                        self.cur_document = None;
                                    }
                                }
                            })
                            .response
                            .on_hover_ui(|ui| {
                                ui.label(format!("{}", id));
                            });
                    }
                    self.documents
                        .retain(|interface| !deleted_ids.contains(&interface.id));
                });
            });
        });

        let viewport = ctx.available_rect();
        let pos = viewport.left_top();
        let size = viewport.size();
        (
            ultraviolet::Vec2 { x: pos.x, y: pos.y },
            ultraviolet::Vec2 {
                x: size.x,
                y: size.y,
            },
        )
    }
}

fn icon_of_node(node: &crate::state::graph::NodeData) -> &'static str {
    use crate::state::graph::{LeafType, NodeType};
    const UNKNOWN: &'static str = "？";
    match (node.leaf(), node.node()) {
        // Leaves
        (Some(LeafType::SolidColor { .. }), None) => FILL_LAYER_ICON,
        (Some(LeafType::StrokeLayer { .. }), None) => STROKE_LAYER_ICON,
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
    pub struct LatchResponse<'ui, State> {
        // Needed for cancellation functionality
        ui: &'ui mut egui::Ui,
        persisted_id: egui::Id,
        output: Option<State>,
    }
    impl<State: 'static> LatchResponse<'_, State> {
        /// Stop the interaction, preventing it from reporting Finished in the future.
        pub fn cancel(self) {
            // Delete egui's persisted state
            self.ui
                .data_mut(|data| data.remove::<State>(self.persisted_id))
        }
        /// Get the output. Returns Some only once when the operation has finished.
        pub fn result(self) -> Option<State> {
            self.output
        }
    }
    /// Interactible UI component where changes in-progress should be ignored,
    /// only producing output when the interaction is fully finished.
    ///
    /// Takes a closure which inspects the mutable State, modifying it and reporting changes via
    /// the [Latch] enum. By default, when no interaction is occuring, it should report [Latch::None]
    pub fn latch<'ui, State, F>(
        ui: &'ui mut egui::Ui,
        id_src: impl std::hash::Hash,
        state: State,
        f: F,
    ) -> LatchResponse<'ui, State>
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
                LatchResponse {
                    ui,
                    persisted_id,
                    output: None,
                }
            }
            // Return the state and clear it
            Latch::Finish => {
                ui.data_mut(|data| data.remove::<State>(persisted_id));
                LatchResponse {
                    ui,
                    persisted_id,
                    output: Some(mutable_state),
                }
            }
            // Nothing to do, clear it if it exists.
            Latch::None => {
                ui.data_mut(|data| data.remove::<State>(persisted_id));
                LatchResponse {
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
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    blend: crate::blend::Blend,
    disable: bool,
) -> Option<crate::blend::Blend> {
    // Get the persisted blend, or use the caller's blend if none.
    latch::latch(ui, (&id, "blend-state"), blend, |ui, blend| {
        let mut finished = false;
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.set_enabled(!disable);
            finished |= ui
                .toggle_value(
                    &mut blend.alpha_clip,
                    egui::RichText::new("α").monospace().strong(),
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
                    for blend_mode in <crate::blend::BlendMode as strum::IntoEnumIterator>::iter() {
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
    .result()
}
/// Inline UI component for changing an optional (possibly passthrough) blend. Makes a copy of the blend internally,
/// only returning a new one on change (skipping partial changes like a dragging slider), returning None otherwise.
fn ui_passthrough_or_blend(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    blend: Option<crate::blend::Blend>,
    disable: bool,
) -> Option<Option<crate::blend::Blend>> {
    latch::latch(ui, (&id, "blend-state"), blend, |ui, blend| {
        let mut finished = false;
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.set_enabled(!disable);
            if let Some(blend) = blend.as_mut() {
                changed |= ui
                    .toggle_value(
                        &mut blend.alpha_clip,
                        egui::RichText::new("α").monospace().strong(),
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
                    for blend_mode in <crate::blend::BlendMode as strum::IntoEnumIterator>::iter() {
                        let select_value = Some(crate::blend::Blend {
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
    .result()
}
fn graph_edit_recurse<
    // Well that's.... not great...
    W: crate::commands::queue::writer::CommandWrite<crate::state::graph::commands::GraphCommand>,
>(
    ui: &mut egui::Ui,
    graph: &mut crate::state::graph::writer::GraphWriter<'_, W>,
    root: Option<crate::state::graph::NodeID>,
    selected_node: &mut Option<crate::state::graph::AnyID>,
    focused_node: &mut Option<crate::state::graph::NodeID>,
    yanked_node: &Option<crate::state::graph::AnyID>,
) {
    let node_ids: Vec<_> = match root {
        Some(root) => graph.iter_node(&root).unwrap().map(|(id, _)| id).collect(),
        None => graph.iter_top_level().map(|(id, _)| id).collect(),
    };

    let mut first = true;
    // Iterate!
    for id in node_ids.into_iter() {
        if !first {
            ui.separator();
        }
        // Name and selection
        let header_response = ui.horizontal(|ui| {
            let data = graph.get(id).unwrap();
            // Choose an icon based on the type of the node:
            // Yanked (if any) gets a scissor icon.
            let icon = if Some(id) == *yanked_node {
                SCISSOR_ICON
            } else {
                icon_of_node(data)
            };
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

            // Only show if not in reparent mode.
            ui.set_enabled(yanked_node.is_none());

            let name = graph.name_mut(id).unwrap();

            // Fetch from last frame - are we hovered?
            let name_hovered_key = egui::Id::new((id.clone(), "name-hovered"));
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
                    if let Some(new_blend) =
                        ui_layer_blend(ui, (&id, "blend"), old_blend, yanked_node.is_some())
                    {
                        // Automatically ignores if no change
                        graph.change_blend(id, new_blend).unwrap();
                    };
                }
            }
            (None, Some(n)) => {
                // Unwrap nodeID:
                let crate::state::graph::AnyID::Node(node_id) = id.clone() else {
                    panic!("Node data and ID mismatch!")
                };
                // Option to focus this subtree:
                header_response.inner.context_menu(|ui| {
                    if ui.button("Focus Subtree").clicked() {
                        *focused_node = Some(node_id.clone())
                    }
                });
                // Display node type - passthrough or grouped blend
                let old_blend = n.blend();
                // Reports new blend when interaction finished, disabled in yank mode.
                if let Some(new_blend) =
                    ui_passthrough_or_blend(ui, (&id, "blend"), old_blend, yanked_node.is_some())
                {
                    match (old_blend, new_blend) {
                        (Some(from), Some(to)) if from != to => {
                            // Simple blend change
                            graph.change_blend(id, to).unwrap();
                        }
                        (None, Some(to)) => {
                            // Type change - passthrough to grouped.
                            graph
                                .set_node(node_id, crate::state::graph::NodeType::GroupedBlend(to))
                                .unwrap()
                        }
                        (Some(_), None) => {
                            // Type change - grouped to passthrough
                            graph
                                .set_node(node_id, crate::state::graph::NodeType::Passthrough)
                                .unwrap()
                        }
                        _ => {
                            // No change
                        }
                    }
                };

                // display children!
                egui::CollapsingHeader::new(egui::RichText::new("Children").italics().weak())
                    .id_source(&id)
                    .default_open(true)
                    .show(ui, |ui| {
                        graph_edit_recurse(
                            ui,
                            graph,
                            Some(node_id),
                            selected_node,
                            focused_node,
                            yanked_node,
                        )
                    });
            }
            (None, None) => (),
            (Some(_), Some(_)) => panic!("Node is both a leaf and node???"),
        }
        first = false;
    }

    // (roundabout way to determine that) it's empty!
    if first {
        ui.label(egui::RichText::new("Nothing here...").italics().weak());
    }
}

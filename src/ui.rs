pub mod requests;

const STROKE_LAYER_ICON: &'static str = "✏";
const NOTE_LAYER_ICON: &'static str = "🖹";
const FILL_LAYER_ICON: &'static str = "⬛";
const GROUP_ICON: &'static str = "🗀";

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
    cur_layer: Option<crate::FuzzID<crate::StrokeLayer>>,
    // Still dirty... UI should be the ground truth for this! But at the same time, the document should own it's blend graph.
    // Hmmmm...
    layers: Vec<crate::FuzzID<crate::StrokeLayer>>,
}
pub struct MainUI {
    // Could totally be static dispatch, but for simplicity:
    /// Displayed on top of everything, disabling all behind it in a stack-like manner
    modals: Vec<Box<dyn UILayer>>,
    /// Displayed as windows on the document viewport.
    inlays: Vec<Box<dyn UILayer>>,

    documents: Vec<(crate::FuzzID<crate::Document>, PerDocumentData)>,
    cur_document: Option<crate::FuzzID<crate::Document>>,

    requests_send: requests::RequestSender,
    // Only some during a drag event on the view rotation slider. Not to be used aside from that! :P
    rotation_drag_value: Option<f32>,
    // Testinggg
    test_blend_graph: crate::state::graph::BlendGraph,
    test_graph_selection: Option<crate::state::graph::AnyID>,
    test_graph_focus: Option<crate::state::graph::NodeID>,
}
impl MainUI {
    pub fn new(requests_send: requests::RequestSender) -> Self {
        Self {
            modals: vec![],
            inlays: vec![],
            documents: vec![],
            cur_document: None,
            rotation_drag_value: None,
            requests_send,
            test_blend_graph: Default::default(),
            test_graph_selection: None,
            test_graph_focus: None,
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
                            let id = crate::FuzzID::<crate::Document>::default();
                            let document = crate::Document {
                                id,
                                layer_top_level: vec![],
                                name: "Uwu".into(),
                                path: None,
                            };
                            let interface = PerDocumentData {
                                cur_layer: None,
                                layers: vec![],
                            };
                            self.documents.push((id, interface));

                            let _ = self
                                .requests_send
                                .send(requests::UiRequest::NewDocument(id));
                        };
                        let _ = add_button(ui, "Save", Some("Ctrl+S"));
                        let _ = add_button(ui, "Save as", Some("Ctrl+Shift+S"));
                        let _ = add_button(ui, "Open", Some("Ctrl+O"));
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
                let Some((_, interface)) = self.documents.iter().find(|(doc, _)| *doc == document)
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
                let mut rotation = self.rotation_drag_value.get_or_insert(0.0);
                let rotation_response = ui.drag_angle(&mut rotation);
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
                // Discard rotation state if we're done interacting.
                if !rotation_response.dragged() {
                    self.rotation_drag_value = None;
                }

                ui.add(egui::Separator::default().vertical());

                // Undo/redo - only show if there is a currently selected layer.
                let undo = egui::Button::new("⮪");
                let redo = egui::Button::new("⮫");

                if let Some(current_layer) = interface.cur_layer {
                    // RTL - add in reverse :P
                    if ui.add(redo).clicked() {
                        let _ = self.requests_send.send(requests::UiRequest::Document {
                            target: document,
                            request: requests::DocumentRequest::Layer {
                                target: current_layer,
                                request: requests::LayerRequest::Redo,
                            },
                        });
                    }
                    if ui.add(undo).clicked() {
                        let _ = self.requests_send.send(requests::UiRequest::Document {
                            target: document,
                            request: requests::DocumentRequest::Layer {
                                target: current_layer,
                                request: requests::LayerRequest::Undo,
                            },
                        });
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
            let Some((_, interface)) = self.documents.iter_mut().find(|(doc, _)| *doc == document)
            else {
                // Not found! Reset cur_document.
                self.cur_document = None;
                return;
            };

            ui.horizontal(|ui| {
                // Copied logic since we can't borrow test_graph_selection throughout this whole
                // ui section
                macro_rules! add_location {
                    () => {
                        match self.test_graph_selection.as_ref() {
                            Some(crate::state::graph::AnyID::Node(id)) => {
                                crate::state::graph::Location::IndexIntoNode(id, 0)
                            }
                            Some(any) => crate::state::graph::Location::AboveSelection(any),
                            // No selection, add into the root of the viewed subree
                            None => match self.test_graph_focus.as_ref() {
                                Some(root) => crate::state::graph::Location::IndexIntoNode(root, 0),
                                None => crate::state::graph::Location::IndexIntoRoot(0),
                            },
                        }
                    };
                }

                /*
                if ui
                    .button(STROKE_LAYER_ICON)
                    .on_hover_text("Add Stroke Layer")
                    .clicked()
                {
                    self.test_graph_selection = self
                        .test_blend_graph
                        .add_leaf(
                            add_location!(),
                            "Stroke Layer".to_string(),
                            crate::state::graph::LeafType::StrokeLayer {
                                blend: Default::default(),
                                source: crate::FuzzID::default(),
                            },
                        )
                        .ok()
                        .map(Into::into);
                }
                if ui
                    .button(NOTE_LAYER_ICON)
                    .on_hover_text("Add Note")
                    .clicked()
                {
                    self.test_graph_selection = self
                        .test_blend_graph
                        .add_leaf(
                            add_location!(),
                            "Note".to_string(),
                            crate::state::graph::LeafType::Note,
                        )
                        .ok()
                        .map(Into::into);
                }
                if ui
                    .button(FILL_LAYER_ICON)
                    .on_hover_text("Add Fill Layer")
                    .clicked()
                {
                    self.test_graph_selection = self
                        .test_blend_graph
                        .add_leaf(
                            add_location!(),
                            "Fill".to_string(),
                            crate::state::graph::LeafType::SolidColor {
                                blend: Default::default(),
                                source: [Default::default(); 4],
                            },
                        )
                        .ok()
                        .map(Into::into);
                }

                ui.add(egui::Separator::default().vertical());

                if ui.button(GROUP_ICON).on_hover_text("Add Group").clicked() {
                    self.test_graph_selection = self
                        .test_blend_graph
                        .add_node(
                            add_location!(),
                            "Group Layer".to_string(),
                            crate::state::graph::NodeType::Passthrough,
                        )
                        .ok()
                        .map(Into::into);
                };*/

                ui.add(egui::Separator::default().vertical());

                let merge_button = egui::Button::new("⤵");
                ui.add_enabled(false, merge_button);

                if ui.button("✖").on_hover_text("Delete layer").clicked() {
                    /*
                    if let Some(id) = self.test_graph_selection.take() {
                        self.test_blend_graph.reparent(id, destination)
                    }*/
                };
            });

            ui.separator();

            // Strange visual flicker when this button is clicked,
            // as the header remains for one frame after the graph switches back.
            // This could become more problematic if the client doesn't refresh the UI
            // one frame after, as the header would stay but there's no subtree selected!
            // Eh, todo. :P
            if self.test_graph_focus.is_some() {
                ui.horizontal(|ui| {
                    if ui.small_button("⬅").clicked() {
                        self.test_graph_focus = None;
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "Subtree of {}",
                            self.test_graph_focus
                                .as_ref()
                                .and_then(|subtree| self.test_blend_graph.get(subtree.clone()))
                                .map(|data| data.name())
                                .unwrap_or("Unknown")
                        ))
                        .italics(),
                    );
                });
            }
            egui::ScrollArea::new([false, true])
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    graph_edit_recurse(
                        ui,
                        &self.test_blend_graph,
                        self.test_graph_focus.clone(),
                        &mut self.test_graph_selection,
                        &mut self.test_graph_focus,
                    )
                });
        });

        egui::SidePanel::left("Color picker").show(&ctx, |ui| {
            /*
            {
                let settings = &mut selections.brush_settings;
                ui.label("Color");
                ui.separator();
                // Why..
                let mut color = egui::Rgba::from_rgba_premultiplied(
                    settings.color_modulate[0],
                    settings.color_modulate[1],
                    settings.color_modulate[2],
                    settings.color_modulate[3],
                );
                if egui::color_picker::color_edit_button_rgba(
                    ui,
                    &mut color,
                    egui::color_picker::Alpha::OnlyBlend,
                )
                .changed()
                {
                    settings.color_modulate = color.to_array();
                };
            }

            ui.separator();
            ui.label("Brushes");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("➕").clicked() {
                    let brush = brush::Brush::default();
                    self.brushes.push(brush);
                }
                if ui.button("✖").on_hover_text("Delete brush").clicked() {
                    if let Some(id) = selections.cur_brush.take() {
                        self.brushes.retain(|brush| brush.id() != id);
                    }
                };
                ui.toggle_value(&mut selections.brush_settings.is_eraser, "Erase");
            });
            ui.separator();

            for brush in self.brushes.iter_mut() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut selections.cur_brush, Some(brush.id().weak()), "");
                        ui.text_edit_singleline(brush.name_mut());
                    })
                    .response
                    .on_hover_ui(|ui| {
                        ui.label(format!("{}", brush.id()));

                        //Smol optimization to avoid formatters
                        let mut buf = uuid::Uuid::encode_buffer();
                        let uuid = brush.universal_id().as_hyphenated().encode_upper(&mut buf);
                        ui.label(&uuid[..]);
                    });
                    egui::CollapsingHeader::new("Settings")
                        .id_source(brush.id())
                        .default_open(true)
                        .show(ui, |ui| {
                            let mut brush_kind = brush.style().brush_kind();

                            egui::ComboBox::new(brush.id(), "")
                                .selected_text(brush_kind.as_ref())
                                .show_ui(ui, |ui| {
                                    for kind in
                                        <brush::BrushKind as strum::IntoEnumIterator>::iter()
                                    {
                                        ui.selectable_value(&mut brush_kind, kind, kind.as_ref());
                                    }
                                });

                            //Changed by user, switch to defaults for the new kind
                            if brush_kind != brush.style().brush_kind() {
                                *brush.style_mut() = brush::BrushStyle::default_for(brush_kind);
                            }

                            match brush.style_mut() {
                                brush::BrushStyle::Stamped { .. } => {
                                    let slider = egui::widgets::Slider::new(
                                        &mut selections.brush_settings.size_mul,
                                        2.0..=50.0,
                                    )
                                    .clamp_to_range(true)
                                    .logarithmic(true)
                                    .suffix("px");

                                    ui.add(slider);

                                    let slider2 = egui::widgets::Slider::new(
                                        &mut selections.brush_settings.spacing_px,
                                        0.1..=10.0,
                                    )
                                    .clamp_to_range(true)
                                    .logarithmic(true)
                                    .suffix("px");

                                    ui.add(slider2);
                                }
                                brush::BrushStyle::Rolled => {}
                            }
                        })
                });
            }
            ui.separator();
            */
            ui.label("Memory Usage Stats");
            let point_resident_usage = crate::repositories::points::global().resident_usage();
            ui.label(format!(
                "Point repository: {}/{}",
                human_bytes::human_bytes(point_resident_usage.0 as f64),
                human_bytes::human_bytes(point_resident_usage.1 as f64),
            ));
        });

        egui::TopBottomPanel::top("documents").show(&ctx, |ui| {
            egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal(|ui| {
                    let mut deleted_ids =
                        smallvec::SmallVec::<[crate::FuzzID<crate::Document>; 1]>::new();
                    for (document_id, _) in self.documents.iter() {
                        egui::containers::Frame::group(ui.style())
                            .outer_margin(egui::Margin::symmetric(0.0, 0.0))
                            .inner_margin(egui::Margin::symmetric(0.0, 0.0))
                            .multiply_with_opacity(if self.cur_document == Some(*document_id) {
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
                                ui.selectable_value(
                                    &mut self.cur_document,
                                    Some(*document_id),
                                    "UwU", //&document.name,
                                );
                                if ui.small_button("✖").clicked() {
                                    deleted_ids.push(*document_id);
                                    //Disselect if deleted.
                                    if self.cur_document == Some(*document_id) {
                                        self.cur_document = None;
                                    }
                                }
                            })
                            .response
                            .on_hover_ui(|ui| {
                                ui.label(format!("{}", *document_id));
                            });
                    }
                    self.documents
                        .retain(|(document_id, _)| !deleted_ids.contains(document_id));
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

fn ui_layer_blend(ui: &mut egui::Ui, id: impl std::hash::Hash, blend: &mut crate::blend::Blend) {
    ui.horizontal(|ui| {
        ui.toggle_value(
            &mut blend.alpha_clip,
            egui::RichText::new("α").monospace().strong(),
        )
        .on_hover_text("Alpha clip");

        ui.add(
            egui::DragValue::new(&mut blend.opacity)
                .fixed_decimals(2)
                .speed(0.01)
                .clamp_range(0.0..=1.0),
        );
        egui::ComboBox::new(id, "")
            .selected_text(blend.mode.as_ref())
            .show_ui(ui, |ui| {
                for blend_mode in <crate::blend::BlendMode as strum::IntoEnumIterator>::iter() {
                    ui.selectable_value(&mut blend.mode, blend_mode, blend_mode.as_ref());
                }
            });
    });
}
fn ui_passthrough_or_blend(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    blend: &mut Option<crate::blend::Blend>,
) {
    ui.horizontal(|ui| {
        if let Some(blend) = blend.as_mut() {
            ui.toggle_value(
                &mut blend.alpha_clip,
                egui::RichText::new("α").monospace().strong(),
            )
            .on_hover_text("Alpha clip");
            ui.add(
                egui::DragValue::new(&mut blend.opacity)
                    .fixed_decimals(2)
                    .speed(0.01)
                    .clamp_range(0.0..=1.0),
            );
        };

        egui::ComboBox::new(id, "")
            .selected_text(
                blend
                    .map(|blend| blend.mode.as_ref().to_string())
                    .unwrap_or("Passthrough".to_string()),
            )
            .show_ui(ui, |ui| {
                ui.selectable_value(blend, None, "Passthrough");
                ui.separator();
                for blend_mode in <crate::blend::BlendMode as strum::IntoEnumIterator>::iter() {
                    ui.selectable_value(
                        blend,
                        Some(crate::blend::Blend {
                            mode: blend_mode,
                            ..Default::default()
                        }),
                        blend_mode.as_ref(),
                    );
                }
            });
    });
}
fn graph_edit_recurse(
    ui: &mut egui::Ui,
    graph: &crate::state::graph::BlendGraph,
    root: Option<crate::state::graph::NodeID>,
    selected_node: &mut Option<crate::state::graph::AnyID>,
    focused_node: &mut Option<crate::state::graph::NodeID>,
) {
    // Weird type for static iterator dispatch for the anonymous iterators exposed by graph.
    // Certainly there's an API change I should make to remove this weird necessity :P
    enum EitherIter<I, R, N>
    where
        R: Iterator<Item = I>,
        N: Iterator<Item = I>,
    {
        TopLevel(R),
        Node(N),
    }
    impl<I, R, N> Iterator for EitherIter<I, R, N>
    where
        R: Iterator<Item = I>,
        N: Iterator<Item = I>,
    {
        type Item = I;
        fn next(&mut self) -> Option<Self::Item> {
            match self {
                EitherIter::Node(n) => n.next(),
                EitherIter::TopLevel(r) => r.next(),
            }
        }
    }
    let mut iter = match root {
        Some(root) => EitherIter::Node(graph.iter_node(&root).unwrap()),
        None => EitherIter::TopLevel(graph.iter_top_level()),
    };
    let mut first = true;
    // Iterate!
    for (id, data) in iter {
        if !first {
            ui.separator();
        }
        // Name and selection
        let header_response = ui.horizontal(|ui| {
            // Choose an icon based on the type of the node:
            let icon = if let Some(leaf) = data.leaf() {
                match leaf {
                    crate::state::graph::LeafType::Note => NOTE_LAYER_ICON,
                    crate::state::graph::LeafType::StrokeLayer { .. } => STROKE_LAYER_ICON,
                    crate::state::graph::LeafType::SolidColor { .. } => FILL_LAYER_ICON,
                }
            } else {
                GROUP_ICON
            };
            ui.selectable_value(selected_node, Some(id.clone()), icon);

            let mut name = data.name().to_string();

            // Fetch from last frame - are we hovered?
            let name_hovered_key = egui::Id::new((id.clone(), "name-hovered"));
            let hovered: Option<bool> = ui.data(|data| data.get_temp(name_hovered_key));
            let edit = egui::TextEdit::singleline(&mut name).frame(hovered.unwrap_or(false));
            let name_response = ui.add(edit);

            // Send data to next frame, to tell that we're hovered or not.
            let interacted = name_response.has_focus() || name_response.hovered();
            ui.data_mut(|data| data.insert_temp(name_hovered_key, interacted));

            // Forward the response of the header items for right clicks, as it takes up all the click area!
            name_response
        });
        // Type-specific UI elements
        match (data.leaf(), data.node()) {
            (Some(_), None) => {}
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
                let mut blend = n.blend();
                ui_passthrough_or_blend(ui, id.clone(), &mut blend);

                // display children!
                egui::CollapsingHeader::new(egui::RichText::new("Children").italics().weak())
                    .default_open(true)
                    .show(ui, |ui| {
                        graph_edit_recurse(ui, graph, Some(node_id), selected_node, focused_node)
                    });
            }
            (None, None) => (),
            (Some(_), Some(_)) => panic!("Node is both a leaf and node???"),
        }
        // Blend, if any.
        if let Some(mut blend) = data.blend() {
            ui_layer_blend(ui, id, &mut blend)
        }
        first = false;
    }

    // (round about way to determine that) it's empty!
    if first {
        ui.label(egui::RichText::new("Nothing here...").italics().weak());
    }
}

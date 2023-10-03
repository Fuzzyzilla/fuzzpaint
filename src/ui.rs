pub mod requests;

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
    cur_layer: Option<crate::WeakID<crate::StrokeLayer>>,
    // Still dirty... UI should be the ground truth for this! But at the same time, the document should own it's blend graph.
    // Hmmmm...
    layers: Vec<crate::WeakID<crate::StrokeLayer>>,
}
pub struct MainUI {
    // Could totally be static dispatch, but for simplicity:
    /// Displayed on top of everything, disabling all behind it in a stack-like manner
    modals: Vec<Box<dyn UILayer>>,
    /// Displayed as windows on the document viewport.
    inlays: Vec<Box<dyn UILayer>>,

    documents: Vec<(crate::WeakID<crate::Document>, PerDocumentData)>,
    cur_document: Option<crate::WeakID<crate::Document>>,

    requests_send: requests::RequestSender,
    // Only some during a drag event on the view rotation slider. Not to be used aside from that! :P
    rotation_drag_value: Option<f32>,
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
                            let weak = id.weak();
                            let document = crate::Document {
                                id: weak,
                                layer_top_level: vec![],
                                name: "Uwu".into(),
                                path: None,
                            };
                            let interface = PerDocumentData {
                                cur_layer: None,
                                layers: vec![],
                            };
                            self.documents.push((weak, interface));

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
                if ui.button("➕").clicked() {
                    let layer_id = crate::FuzzID::<crate::StrokeLayer>::default();
                    let layer = crate::StrokeLayer {
                        id: layer_id.weak(),
                        name: format!("{layer_id}"),
                        blend: Default::default(),
                    };
                    if let Some(selected) = interface.cur_layer {
                        let selected_index = interface
                            .layers
                            .iter()
                            .enumerate()
                            .find(|(_, id)| **id == selected)
                            .map(|(idx, _)| idx);
                        if let Some(index) = selected_index {
                            // Insert atop selected index
                            interface.layers.insert(index, layer_id.weak());
                        } else {
                            // Selected layer not found! Just insert at top
                            interface.layers.push(layer_id.weak());
                        }
                    } else {
                        interface.layers.push(layer_id.weak());
                    }
                    // Select the new layer
                    interface.cur_layer = Some(layer_id.weak());

                    let _ = self.requests_send.send(requests::UiRequest::Document {
                        target: document,
                        request: requests::DocumentRequest::NewLayer(layer_id),
                    });
                }

                let folder_button = egui::Button::new("🗀");
                ui.add_enabled(false, folder_button);

                let merge_button = egui::Button::new("⤵");
                ui.add_enabled(false, merge_button);

                if ui.button("✖").on_hover_text("Delete layer").clicked() {
                    if let Some(selected) = interface.cur_layer.take() {
                        // Find the index of the selected layer
                        let selected_index = interface
                            .layers
                            .iter()
                            .enumerate()
                            .find(|(_, id)| **id == selected)
                            .map(|(idx, _)| idx);
                        // Remove, if found
                        if let Some(idx) = selected_index {
                            interface.layers.remove(idx);
                            let _ = self.requests_send.send(requests::UiRequest::Document {
                                target: document,
                                request: requests::DocumentRequest::Layer {
                                    target: selected,
                                    request: requests::LayerRequest::Deleted,
                                },
                            });
                        }
                    }
                };
            });

            ui.separator();
            /*
            egui::ScrollArea::vertical().show(ui, |ui| {
                let layers = document.layers_mut();
                for layer in layers.iter_mut().rev() {
                    Self::layer_edit(ui, &mut document_selections.cur_layer, layer);
                }
            });*/
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
                        smallvec::SmallVec::<[crate::WeakID<crate::Document>; 1]>::new();
                    for (document_id, interface) in self.documents.iter() {
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
        //alpha symbol for clipping (allocates every frame - why??)
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
fn layer_edit(
    ui: &mut egui::Ui,
    cur_layer: &mut Option<crate::WeakID<crate::StrokeLayer>>,
    layer: &mut crate::StrokeLayer,
) {
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.selectable_value(cur_layer, Some(layer.id), "✏");
            ui.text_edit_singleline(&mut layer.name);
        })
        .response
        .on_hover_ui(|ui| {
            ui.label(format!("{}", layer.id));
        });

        ui_layer_blend(ui, &layer.id, &mut layer.blend);
    });
}

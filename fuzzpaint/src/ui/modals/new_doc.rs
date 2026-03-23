use fuzzpaint_types::dpi;
// array of categories, each category containing an array of (name, size) tuples.
const SIZE_PRESETS: &[&[(&str, dpi::Length<ultraviolet::Vec2>)]] = &[
    &[(
        "US Letter",
        dpi::Length::physical(ultraviolet::Vec2::new(8.5, 11.0), dpi::PhysicalUnit::Inch),
    )],
    // Weirdly, the ISO standard has rational dimensions, despite the whole
    // *thing* of A-size paper being an irrational aspect ratio lol
    &[(
        "A4",
        dpi::Length::physical(
            ultraviolet::Vec2::new(21.0, 29.7),
            dpi::PhysicalUnit::Centimeter,
        ),
    )],
];
const LOGICAL_PX_DESCRIPTION: &str = "\
Multiplied by the scale factor on export to determine the final size of the \
canvas in physical pixels (ppx). This makes it possible to render the drawing \
at a higher or lower resolution when you go to export it.";
const PHYSICAL_PX_DESCRIPTION: &str = "\
The direct size of the final image, ignoring export scale factor. Generally, \
you should prefer Logical Pixels (px) with a scale factor of one. Otherwise, \
exporting the image at a different size will result in the logical content \
getting larger while the rendered region remains the same (proportionally \
smaller.)";

pub struct Modal {
    size: dpi::Length<ultraviolet::Vec2>,
    dots_per: (f32, dpi::PhysicalUnit),
    /// Which remote to create on?
    target_connection: Option<crate::connections::ConnectionID>,
}
impl Default for Modal {
    fn default() -> Self {
        Self {
            size: dpi::Length::logical_px(ultraviolet::Vec2::broadcast(1024.0)),
            dots_per: (300.0, dpi::PhysicalUnit::Inch),
            // Filled in on first UI.
            target_connection: None,
        }
    }
}
impl super::Modal for Modal {
    fn do_ui(
        &mut self,
        _id: egui::Id,
        ui: &mut egui::Ui,
        _state: &mut crate::ui::MainUI,
        interface: &mut crate::ui::interface::Interface,
    ) -> super::Response {
        super::title(ui, "New Document");
        // Ask the user which remote to create the document on.
        {
            // Collect the ids and names of the connections:
            let connections = interface
                .iter_connections()
                .map(|(id, conn)| (id, conn.name().to_owned()))
                .collect::<Vec<_>>();

            // No remotes, bail!
            let Some(first_conn) = connections.first() else {
                ui.label(
                    "You must connect to a local or remote server before you can create a document.",
                );
                return super::Response::Retain;
            };

            // If the server list no longer contains the selected server, clear the
            // selection.
            if connections
                .iter()
                .find(|(id, _)| self.target_connection.as_ref() == Some(id))
                .is_none()
            {
                self.target_connection = None;
            }
            // Default to the arbitrary "first" connection.
            self.target_connection.get_or_insert(first_conn.0.clone());
            // Find the name of the selected connection for the combo. The above
            // two statements make sure this is never None.s
            let selected_connection_name = connections
                .iter()
                .find_map(|(id, name)| {
                    (self.target_connection.as_ref() == Some(id)).then_some(name)
                })
                .unwrap();

            let target_connection = self.target_connection.as_mut().unwrap();

            egui::ComboBox::from_label("Create on")
                .selected_text(selected_connection_name)
                .show_ui(ui, |ui| {
                    for (id, name) in connections {
                        ui.selectable_value(target_connection, id, name);
                    }
                });
        }
        ui.separator();

        egui::ComboBox::from_id_salt(ui.id().with("presets"))
            .selected_text("Presets...")
            .show_ui(ui, |ui| {
                let mut is_first = true;
                for &category in SIZE_PRESETS {
                    if !is_first {
                        ui.separator();
                        is_first = false;
                    }
                    for &(name, size) in category {
                        if ui.selectable_label(false, name).clicked() {
                            self.size = size;
                        }
                    }
                }
            });

        ui.horizontal(|ui| {
            let disallow_decimals = matches!(self.size.unit(), dpi::Unit::PhysicalPx);
            let size = self.size.value_mut();
            ui.add(
                egui::DragValue::new(&mut size.x)
                    .range(0.0..=4096.0)
                    .max_decimals_opt(disallow_decimals.then_some(0))
                    .clamp_existing_to_range(true),
            );
            ui.label("×");
            ui.add(
                egui::DragValue::new(&mut size.y)
                    .range(0.0..=4096.0)
                    .max_decimals_opt(disallow_decimals.then_some(0))
                    .clamp_existing_to_range(true),
            );

            egui::ComboBox::from_id_salt(ui.id().with("size-unit"))
                .selected_text(format!("{}", self.size.unit()))
                .width(ui.style().spacing.combo_width / 4.0)
                .show_ui(ui, |ui| {
                    let mut new_unit = self.size.unit();
                    ui.selectable_value(&mut new_unit, dpi::Unit::LogicalPx, "px")
                        .on_hover_text("Logical Pixels (default)")
                        .on_hover_text(LOGICAL_PX_DESCRIPTION);
                    ui.selectable_value(&mut new_unit, dpi::Unit::PhysicalPx, "ppx")
                        .on_hover_text("Physical Pixels")
                        .on_hover_text(PHYSICAL_PX_DESCRIPTION);

                    ui.selectable_value(
                        &mut new_unit,
                        dpi::Unit::Physical(dpi::PhysicalUnit::Inch),
                        "in",
                    )
                    .on_hover_text("Inches");
                    ui.selectable_value(
                        &mut new_unit,
                        dpi::Unit::Physical(dpi::PhysicalUnit::Centimeter),
                        "cm",
                    )
                    .on_hover_text("Centimetres");
                    ui.selectable_value(
                        &mut new_unit,
                        dpi::Unit::Physical(dpi::PhysicalUnit::Point),
                        "pt",
                    )
                    .on_hover_text("Typographic Points");

                    let dim = *self.size.value_mut();
                    self.size = dpi::Length::from_value_unit(dim, new_unit);
                });
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.dots_per.0)
                    .range(1.0..=12000.0)
                    .clamp_existing_to_range(true),
            );
            ui.label("dots per");
            egui::ComboBox::from_id_salt(ui.id().with("dpi-unit"))
                .width(ui.style().spacing.combo_width / 4.0)
                .selected_text(format!("{}", self.dots_per.1))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.dots_per.1, dpi::PhysicalUnit::Inch, "in")
                        .on_hover_text("Inche");
                    ui.selectable_value(&mut self.dots_per.1, dpi::PhysicalUnit::Centimeter, "cm")
                        .on_hover_text("Centimetre");
                    ui.selectable_value(&mut self.dots_per.1, dpi::PhysicalUnit::Point, "pt")
                        .on_hover_text("Typographic Point");
                });
        });
        ui.label(egui::RichText::new(format!("Effective resolution: {}×{}@{}dpi", 0, 0, 0)).weak());
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Create").clicked() {
                log::warn!("unimplimented");
            }
        });
        super::Response::Retain
    }
}

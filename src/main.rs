#![feature(portable_simd)]
#![feature(once_cell_try)]
use std::sync::Arc;
mod egui_impl;
pub mod gpu_err;
pub mod vulkano_prelude;
pub mod window;
use brush::BrushStyle;
use cgmath::{Matrix4, SquareMatrix};
use vulkano::command_buffer;
use vulkano_prelude::*;
pub mod actions;
pub mod blend;
pub mod brush;
pub mod document_viewport_proxy;
pub mod gpu_tess;
pub mod id;
pub mod render_device;
pub mod stylus_events;
pub mod tess;
pub mod view_transform;
use blend::{Blend, BlendMode};

pub use id::{FuzzID, WeakID};
pub use tess::StrokeTessellator;

#[cfg(feature = "dhat_heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Obviously will be user specified on a per-document basis, but for now...
const DOCUMENT_DIMENSION: u32 = 32;
/// Premultiplied RGBA16F for interesting effects (negative + overbright colors and alpha) with
/// more than 11bit per channel precision in the \[0,1\] range.
/// Will it be user specified in the future?
const DOCUMENT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

use anyhow::Result as AnyResult;

pub fn preferences_dir() -> Option<std::path::PathBuf> {
    let mut base_dir = dirs::preference_dir()?;
    base_dir.push(env!("CARGO_PKG_NAME"));
    Some(base_dir)
}

pub struct GlobalHotkeys {
    failed_to_load: bool,
    actions_to_keys: actions::hotkeys::ActionsToKeys,
    keys_to_actions: actions::hotkeys::KeysToActions,
}
impl GlobalHotkeys {
    const FILENAME: &'static str = "hotkeys.ron";
    /// Shared global hotkeys, saved and loaded from user preferences.
    /// (Or defaulted, if unavailable for some reason)
    pub fn get() -> &'static Self {
        static GLOBAL_HOTKEYS: std::sync::OnceLock<GlobalHotkeys> = std::sync::OnceLock::new();

        GLOBAL_HOTKEYS.get_or_init(|| {
            let mut dir = preferences_dir();
            match dir.as_mut() {
                None => Self::no_path(),
                Some(dir) => {
                    dir.push(Self::FILENAME);
                    Self::load_or_default(&dir)
                }
            }
        })
    }
    pub fn no_path() -> Self {
        log::warn!("Hotkeys weren't available, defaulting.");
        use actions::hotkeys::*;
        let default = ActionsToKeys::default();
        // Default action map is reversable - this is assured by the default impl when debugging.
        let reverse = (&default).try_into().unwrap();

        Self {
            failed_to_load: true,
            keys_to_actions: reverse,
            actions_to_keys: default,
        }
    }
    fn load_or_default(path: &std::path::Path) -> Self {
        use actions::hotkeys::*;
        let mappings: anyhow::Result<(ActionsToKeys, KeysToActions)> = try_block::try_block! {
            let string = std::fs::read_to_string(path)?;
            let actions_to_keys : ActionsToKeys = ron::from_str(&string)?;
            let keys_to_actions : KeysToActions = (&actions_to_keys).try_into()?;

            Ok((actions_to_keys,keys_to_actions))
        };

        match mappings {
            Ok((actions_to_keys, keys_to_actions)) => Self {
                failed_to_load: false,
                actions_to_keys,
                keys_to_actions,
            },
            Err(_) => Self::no_path(),
        }
    }
    /// Return true if loading user's settings failed. This can be useful for
    /// displaying a warning.
    pub fn did_fail_to_load(&self) -> bool {
        self.failed_to_load
    }
    pub fn save(&self) -> anyhow::Result<()> {
        let mut preferences =
            preferences_dir().ok_or_else(|| anyhow::anyhow!("No preferences dir found"))?;
        // Explicity do *not* create recursively. If not found, the user probably has a good reason.
        // Ignore errors (could already exist). Any real errors will be emitted by file access below.
        let _ = std::fs::DirBuilder::new().create(&preferences);

        preferences.push(Self::FILENAME);
        let writer = std::io::BufWriter::new(std::fs::File::create(&preferences)?);
        Ok(ron::ser::to_writer_pretty(
            writer,
            &self.actions_to_keys,
            Default::default(),
        )?)
    }
}

pub struct StrokeLayer {
    id: FuzzID<Self>,
    name: String,
    blend: blend::Blend,
}
impl Default for StrokeLayer {
    fn default() -> Self {
        let id = Default::default();
        Self {
            name: format!("Layer {}", id),
            id,
            blend: Default::default(),
        }
    }
}
pub struct Document {
    /// The path from which the file was loaded, or None if opened as new.
    path: Option<std::path::PathBuf>,
    /// Name of the document, from its path or generated.
    name: String,

    // In structure, a document is rather similar to a GroupLayer :O
    /// Layers that make up this document
    layer_top_level: Vec<StrokeLayer>,

    /// ID that is unique within this execution of the program
    id: FuzzID<Document>,
}
impl Default for Document {
    fn default() -> Self {
        let id = FuzzID::default();
        Self {
            path: None,
            layer_top_level: Vec::new(),
            name: format!("New Document {}", id.id()),
            id,
        }
    }
}
impl Document {
    // Internal structure in public interface???
    pub fn layers_mut(&mut self) -> &mut Vec<StrokeLayer> {
        &mut self.layer_top_level
    }
}

struct PerDocumentInterface {
    zoom: f32,
    rotate: f32,
}
impl Default for PerDocumentInterface {
    fn default() -> Self {
        Self {
            zoom: 100.0,
            rotate: 0.0,
        }
    }
}
pub struct DocumentUserInterface {
    color: egui::Color32,

    // modal_stack: Vec<Box<dyn FnMut(&mut egui::Ui) -> ()>>,
    brushes: Vec<brush::Brush>,

    document_interfaces: std::collections::HashMap<WeakID<Document>, PerDocumentInterface>,

    viewport: egui::Rect,
}
impl Default for DocumentUserInterface {
    fn default() -> Self {
        let new_brush = brush::Brush::default();
        Self {
            color: egui::Color32::BLUE,
            brushes: vec![new_brush],
            document_interfaces: Default::default(),

            viewport: egui::Rect {
                min: egui::Pos2::ZERO,
                max: egui::Pos2::ZERO,
            },
        }
    }
}

impl DocumentUserInterface {
    /// Get the available area for document rendering, in logical pixels.
    /// None if there is no space for a viewport.
    pub fn get_document_viewport(&self) -> Option<egui::Rect> {
        // Avoid giving a zero or negative size viewport.
        let size = self.viewport.size();

        if size.x > 0.0 && size.y > 0.0 {
            Some(self.viewport)
        } else {
            None
        }
    }
    fn ui_layer_blend(ui: &mut egui::Ui, id: impl std::hash::Hash, blend: &mut Blend) {
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
                    for blend_mode in <BlendMode as strum::IntoEnumIterator>::iter() {
                        ui.selectable_value(&mut blend.mode, blend_mode, blend_mode.as_ref());
                    }
                });
        });
    }
    fn layer_edit(
        ui: &mut egui::Ui,
        cur_layer: &mut Option<WeakID<StrokeLayer>>,
        layer: &mut StrokeLayer,
    ) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(cur_layer, Some(layer.id.weak()), "✏");
                ui.text_edit_singleline(&mut layer.name);
            })
            .response
            .on_hover_ui(|ui| {
                ui.label(format!("{}", layer.id));
            });

            Self::ui_layer_blend(ui, &layer.id, &mut layer.blend);
        });
    }
    /*
    fn target_layer(&self) -> Option<LayerID> {
        let id = self.cur_document?;
        // Selected layer of the currently focused document, if any
        self.document_interfaces.get(&id)?.cur_layer
    }
    fn ui_layer_iter(
        ui: &mut egui::Ui,
        document_interface: &mut PerDocumentInterface,
        layers: &mut [LayerNode],
    ) {
        if layers.is_empty() {
            ui.label(
                egui::RichText::new("Empty, for now...")
                    .italics()
                    .color(egui::Color32::DARK_GRAY),
            );
        }
        for layer in layers.iter_mut() {
            ui.group(|ui| {
                match layer {
                    LayerNode::Group {
                        layer,
                        children,
                        id,
                    } => {
                        ui.horizontal(|ui| {
                            ui.selectable_value(
                                &mut document_interface.cur_layer,
                                Some(id.weak()),
                                "🗀",
                            );
                            ui.text_edit_singleline(&mut layer.name);
                        })
                        .response
                        .on_hover_ui(|ui| {
                            ui.label(format!("{}", id));
                            ui.label(format!("{}", layer.id));
                        });

                        let mut passthrough = layer.blend.is_none();
                        ui.checkbox(&mut passthrough, "Passthrough");
                        if !passthrough {
                            //Get or insert default is unstable? :V
                            let blend = layer.blend.get_or_insert_with(Default::default);

                            Self::ui_layer_blend(ui, &id, blend);
                        } else {
                            layer.blend = None;
                        }

                        let children = children.get_mut();
                        egui::CollapsingHeader::new("Children")
                            .id_source(&id)
                            .default_open(true)
                            .enabled(!children.is_empty())
                            .show_unindented(ui, |ui| {
                                Self::ui_layer_slice(ui, document_interface, children);
                            });
                    }
                    LayerNode::StrokeLayer { layer, id } => {
                        ui.horizontal(|ui| {
                            ui.selectable_value(
                                &mut document_interface.cur_layer,
                                Some(id.weak()),
                                "✏",
                            );
                            ui.text_edit_singleline(&mut layer.name);
                        })
                        .response
                        .on_hover_ui(|ui| {
                            ui.label(format!("{}", id));
                            ui.label(format!("{}", layer.id));
                        });

                        Self::ui_layer_blend(ui, id, &mut layer.blend);
                    }
                }
            })
            .response
            .context_menu(|ui| {
                if ui.button("Focus Subtree...").clicked() {
                    document_interface.focused_subtree = Some(layer.id().weak());
                }
            });
        }
    }*/
    pub fn ui(&mut self, ctx: &egui::Context) {
        let globals = GLOBALS.get_or_init(Globals::new);
        let mut selections = globals.selections().blocking_write();
        let mut documents = globals.documents().blocking_write();
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
                            let document = Document::default();
                            self.document_interfaces
                                .insert(document.id.weak(), Default::default());
                            documents.push(document);
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
                //Everything here is BACKWARDS!!

                ui.label(":V");

                ui.add(egui::Separator::default().vertical());

                // if there is no current document, there is nothing for us to do here
                let Some(document) = selections.cur_document else {
                    return;
                };

                // Get (or create) the document interface
                let interface = self.document_interfaces.entry(document).or_default();

                //Zoom controls
                if ui.small_button("⟲").clicked() {
                    interface.zoom = 100.0;
                }
                if ui.small_button("➕").clicked() {
                    //Next power of two
                    interface.zoom =
                        (2.0f32.powf(((interface.zoom / 100.0).log2() + 0.001).ceil()) * 100.0)
                            .max(12.5);
                }

                ui.add(
                    egui::Slider::new(&mut interface.zoom, 12.5..=12800.0)
                        .logarithmic(true)
                        .clamp_to_range(true)
                        .suffix("%")
                        .trailing_fill(true),
                );

                if ui.small_button("➖").clicked() {
                    //Previous power of two
                    interface.zoom =
                        (2.0f32.powf(((interface.zoom / 100.0).log2() - 0.001).floor()) * 100.0)
                            .max(12.5);
                }

                ui.add(egui::Separator::default().vertical());

                //Rotate controls
                if ui.small_button("⟲").clicked() {
                    interface.rotate = 0.0;
                };
                if ui.drag_angle(&mut interface.rotate).changed() {
                    interface.rotate = interface.rotate % std::f32::consts::TAU;
                }

                ui.add(egui::Separator::default().vertical());

                if ui.button("⮪").clicked() {
                    selections
                        .undos
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                };
            });
        });

        egui::SidePanel::right("Layers").show(&ctx, |ui| {
            ui.label("Layers");
            ui.separator();

            // if there is no current document, there is nothing for us to do here
            let Some(document_id) = selections.cur_document else {
                return;
            };

            // Find the document, otherwise clear selection
            let Some(document) = documents.iter_mut().find(|doc| &doc.id == document_id) else {
                selections.cur_document = None;
                return;
            };

            let document_selections = selections
                .document_selections
                .entry(document_id)
                .or_default();

            ui.horizontal(|ui| {
                if ui.button("➕").clicked() {
                    let layer = StrokeLayer::default();
                    let new_weak_id = layer.id.weak();
                    if let Some(selected) = document_selections.cur_layer {
                        let layers = document.layers_mut();
                        let selected_index = layers
                            .iter()
                            .enumerate()
                            .find(|(_, layer)| layer.id.weak() == selected)
                            .map(|(idx, _)| idx);
                        if let Some(index) = selected_index {
                            // Insert atop selected index
                            document.layers_mut().insert(index + 1, layer)
                        } else {
                            // Selected layer not found! Just insert at top
                            document.layers_mut().push(layer)
                        }
                    } else {
                        document.layers_mut().push(layer)
                    }
                    // Select the new layer
                    document_selections.cur_layer = Some(new_weak_id);
                }

                let folder_button = egui::Button::new("🗀");
                ui.add_enabled(false, folder_button);

                let _ = ui.button("⤵").on_hover_text("Merge down");
                if ui.button("✖").on_hover_text("Delete layer").clicked() {
                    if let Some(selected) = document_selections.cur_layer.take() {
                        let layers = document.layers_mut();
                        // Find the index of the selected layer
                        let selected_index = layers
                            .iter()
                            .enumerate()
                            .find(|(_, layer)| layer.id.weak() == selected)
                            .map(|(idx, _)| idx);
                        // Remove, if found
                        if let Some(idx) = selected_index {
                            layers.remove(idx);
                        }
                    }
                };
            });
            /*
            if let Some(subtree) = document_interface.focused_subtree {
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("⬅").clicked() {
                        document_interface.focused_subtree = None;
                    }
                    ui.label(format!("Viewing subtree of {subtree}"));
                });
            }*/

            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                let layers = document.layers_mut();
                for layer in layers.iter_mut().rev() {
                    Self::layer_edit(ui, &mut document_selections.cur_layer, layer);
                }
                /*
                match document_interface
                    .focused_subtree
                    .and_then(|tree| document.layers.find_mut_children_of(tree))
                {
                    Some(subtree) => {
                        Self::ui_layer_slice(ui, document_interface, subtree);
                    }
                    None => {
                        document_interface.focused_subtree = None;
                        Self::ui_layer_slice(
                            ui,
                            document_interface,
                            &mut document.layers.top_level,
                        );
                    }
                }*/
            });
        });

        egui::SidePanel::left("Color picker").show(&ctx, |ui| {
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

            /* Old Image picker, useful later
                'image_fail: {
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("images", &["png"])
                    .set_directory(".")
                    .pick_file() else {break 'image_fail};

                let Ok(image) = image::open(path) else {break 'image_fail};
                let image = image.to_rgba8();
                let image_size = image.dimensions();
                let image_size = [image_size.0 as usize, image_size.1 as usize];
                let image_data = image.as_bytes();

                let image = egui::ColorImage::from_rgba_unmultiplied(image_size, image_data);

                let handle = ctx.load_texture("Brush texture", image, egui::TextureOptions::LINEAR);

                brush.image = Some(handle);
                brush.image_uv = egui::Rect{min: egui::Pos2::ZERO, max: egui::Pos2 {x: 1.0, y: 1.0}};
            }*/
        });

        egui::TopBottomPanel::top("documents").show(&ctx, |ui| {
            egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal(|ui| {
                    let mut deleted_ids = smallvec::SmallVec::<[_; 1]>::new();
                    for document in documents.iter() {
                        egui::containers::Frame::group(ui.style())
                            .outer_margin(egui::Margin::symmetric(0.0, 0.0))
                            .inner_margin(egui::Margin::symmetric(0.0, 0.0))
                            .multiply_with_opacity(
                                if selections.cur_document == Some(document.id.weak()) {
                                    1.0
                                } else {
                                    0.0
                                },
                            )
                            .rounding(egui::Rounding {
                                ne: 2.0,
                                nw: 2.0,
                                ..0.0.into()
                            })
                            .show(ui, |ui| {
                                ui.selectable_value(
                                    &mut selections.cur_document,
                                    Some(document.id.weak()),
                                    &document.name,
                                );
                                if ui.small_button("✖").clicked() {
                                    deleted_ids.push(document.id.weak());
                                    //Disselect if deleted.
                                    if selections.cur_document == Some(document.id.weak()) {
                                        selections.cur_document = None;
                                    }
                                }
                            })
                            .response
                            .on_hover_ui(|ui| {
                                ui.label(format!("{}", document.id));
                            });
                    }
                    documents.retain(|document| !deleted_ids.contains(&document.id.weak()));
                    for id in deleted_ids.into_iter() {
                        self.document_interfaces.remove(&id);
                    }
                });
            });
        });

        self.viewport = ctx.available_rect();
    }
}

mod stroke_renderer {
    /// The data managed by the renderer.
    /// For now, in persuit of actually getting a working product one day,
    /// this is a very coarse caching sceme. In the future, perhaps a bit more granular
    /// control can occur, should performance become an issue:
    ///  * Caching images of incrementally older states, reducing work to get to any given state (performant undo)
    ///  * Caching tesselation output
    pub struct RenderData {
        image: Arc<vk::StorageImage>,
        pub view: Arc<vk::ImageView<vk::StorageImage>>,
    }

    use crate::vk;
    use anyhow::Result as AnyResult;
    use std::sync::Arc;
    use vulkano::{pipeline::graphics::vertex_input::Vertex, pipeline::Pipeline, sync::GpuFuture};
    mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/stamp.vert",
        }
    }

    pub struct StrokeLayerRenderer {
        context: Arc<crate::render_device::RenderContext>,
        texture_descriptor: Arc<vk::PersistentDescriptorSet>,
        gpu_tess: super::gpu_tess::GpuStampTess,
        pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl StrokeLayerRenderer {
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> AnyResult<Self> {
            let image = image::open("brushes/splotch.png")
                .unwrap()
                .into_luma_alpha8();

            //Iter over transparencies.
            let image_grey = image.iter().skip(1).step_by(2).cloned();

            let mut cb = vk::AutoCommandBufferBuilder::primary(
                context.allocators().command_buffer(),
                context.queues().transfer().idx(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
            )?;
            let (image, sampler) = {
                let image = vk::ImmutableImage::from_iter(
                    context.allocators().memory(),
                    image_grey,
                    vk::ImageDimensions::Dim2d {
                        width: image.width(),
                        height: image.height(),
                        array_layers: 1,
                    },
                    vulkano::image::MipmapsCount::One,
                    vk::Format::R8_UNORM,
                    &mut cb,
                )?;
                context
                    .now()
                    .then_execute(context.queues().transfer().queue().clone(), cb.build()?)?
                    .then_signal_fence_and_flush()?
                    .wait(None)?;

                let view = vk::ImageView::new(
                    image.clone(),
                    vk::ImageViewCreateInfo {
                        component_mapping: vk::ComponentMapping {
                            //Red is coverage of white, with premul.
                            a: vk::ComponentSwizzle::Red,
                            r: vk::ComponentSwizzle::Red,
                            b: vk::ComponentSwizzle::Red,
                            g: vk::ComponentSwizzle::Red,
                        },
                        ..vk::ImageViewCreateInfo::from_image(&image)
                    },
                )?;

                let sampler = vk::Sampler::new(
                    context.device().clone(),
                    vk::SamplerCreateInfo {
                        min_filter: vk::Filter::Linear,
                        mag_filter: vk::Filter::Linear,
                        ..Default::default()
                    },
                )?;

                (view, sampler)
            };

            // Safety: dual_src_blend currently broken in vulkano - pull request submitted, for now
            // no erase U_U
            let frag = unsafe {
                vulkano::shader::ShaderModule::from_bytes(
                    context.device().clone(),
                    include_bytes!("shaders/stamp.frag.spv"),
                )
            }?;
            let vert = vert::load(context.device().clone())?;
            // Unwraps ok here, using GLSL where "main" is the only allowed entry point.
            let frag = frag.entry_point("main").unwrap();
            let vert = vert.entry_point("main").unwrap();

            // DualSrcBlend (~75% coverage) is used to control whether to erase or draw on a per-fragment basis
            // [1.0; 4] = draw, [0.0; 4] = erase.
            let mut premul_dyn_constants = vk::ColorBlendState::new(1);
            premul_dyn_constants.blend_constants = vk::StateMode::Fixed([1.0; 4]);
            premul_dyn_constants.attachments[0].blend = Some(vk::AttachmentBlend {
                alpha_source: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
                color_source: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
                alpha_destination:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                color_destination:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                alpha_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
                color_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
            });

            let pipeline = vk::GraphicsPipeline::start()
                .fragment_shader(frag, ())
                .vertex_shader(vert, ())
                .vertex_input_state(super::gpu_tess::interface::OutputStrokeVertex::per_vertex())
                .input_assembly_state(vk::InputAssemblyState::new()) //Triangle list, no prim restart
                .color_blend_state(premul_dyn_constants)
                .rasterization_state(vk::RasterizationState::new()) // No cull
                .viewport_state(vk::ViewportState::viewport_fixed_scissor_irrelevant([
                    vk::Viewport {
                        depth_range: 0.0..1.0,
                        dimensions: [super::DOCUMENT_DIMENSION as f32; 2],
                        origin: [0.0; 2],
                    },
                ]))
                .render_pass(
                    vulkano::pipeline::graphics::render_pass::PipelineRenderPassType::BeginRendering(
                        vulkano::pipeline::graphics::render_pass::PipelineRenderingCreateInfo {
                            view_mask: 0,
                            color_attachment_formats: vec![Some(super::DOCUMENT_FORMAT)],
                            depth_attachment_format: None,
                            stencil_attachment_format: None,
                            ..Default::default()
                        }
                    )
                )
                .build(context.device().clone())?;

            let descriptor_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0, image, sampler,
                )],
            )?;

            let tess = super::gpu_tess::GpuStampTess::new(context.clone())?;

            Ok(Self {
                context,
                pipeline,
                gpu_tess: tess,
                texture_descriptor: descriptor_set,
            })
        }
        /// Allocate a new RenderData object. Initial contents are undefined!
        pub fn uninit_render_data(&self) -> anyhow::Result<RenderData> {
            let image = vk::StorageImage::with_usage(
                self.context.allocators().memory(),
                vulkano::image::ImageDimensions::Dim2d {
                    width: super::DOCUMENT_DIMENSION,
                    height: super::DOCUMENT_DIMENSION,
                    array_layers: 1,
                },
                super::DOCUMENT_FORMAT,
                vk::ImageUsage::COLOR_ATTACHMENT | vk::ImageUsage::STORAGE,
                vk::ImageCreateFlags::empty(),
                [
                    // Todo: if these are the same queue, what happen?
                    self.context.queues().graphics().idx(),
                    self.context.queues().compute().idx(),
                ]
                .into_iter(),
            )?;
            let view = vk::ImageView::new_default(image.clone())?;

            use vulkano::VulkanObject;
            log::info!("Made render data at id{:?}", view.handle());

            Ok(RenderData { image, view })
        }
        pub fn draw(
            &self,
            strokes: &[super::WeakStroke],
            renderbuf: &RenderData,
            clear: bool,
        ) -> AnyResult<vk::sync::future::SemaphoreSignalFuture<impl vk::sync::GpuFuture>> {
            let (future, vertices, indirects) = self.gpu_tess.tess(strokes)?;
            let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
            )?;

            let mut matrix = cgmath::Matrix4::from_scale(2.0 / super::DOCUMENT_DIMENSION as f32);
            matrix.y *= -1.0;
            matrix.w.x -= 1.0;
            matrix.w.y += 1.0;

            command_buffer
                .begin_rendering(vulkano::command_buffer::RenderingInfo {
                    color_attachments: vec![Some(
                        vulkano::command_buffer::RenderingAttachmentInfo {
                            clear_value: if clear {
                                Some([0.0, 0.0, 0.0, 0.0].into())
                            } else {
                                None
                            },
                            load_op: if clear {
                                vulkano::render_pass::LoadOp::Clear
                            } else {
                                vulkano::render_pass::LoadOp::Load
                            },
                            store_op: vulkano::render_pass::StoreOp::Store,
                            ..vulkano::command_buffer::RenderingAttachmentInfo::image_view(
                                renderbuf.view.clone(),
                            )
                        },
                    )],
                    contents: vulkano::command_buffer::SubpassContents::Inline,
                    depth_attachment: None,
                    ..Default::default()
                })?
                .bind_pipeline_graphics(self.pipeline.clone())
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    Into::<[[f32; 4]; 4]>::into(matrix),
                )
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    self.texture_descriptor.clone(),
                )
                .bind_vertex_buffers(0, vertices)
                .draw_indirect(indirects)?
                .end_rendering()?;

            let command_buffer = command_buffer.build()?;

            // After tessellation finishes, render.
            Ok(future
                .then_execute(
                    self.context.queues().graphics().queue().clone(),
                    command_buffer,
                )?
                .then_signal_semaphore_and_flush()?)
        }
    }
}

#[derive(Clone)]
pub struct StrokeBrushSettings {
    brush: brush::WeakBrushID,
    /// `a` is flow, NOT opacity, since the stroke is blended continuously not blended as a group.
    color_modulate: [f32; 4],
    spacing_px: f32,
    size_mul: f32,
    /// If true, the blend constants must be set to generate an erasing effect.
    is_eraser: bool,
}
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct StrokePoint {
    pos: [f32; 2],
    pressure: f32,
    /// Arc length of stroke from beginning to this point
    dist: f32,
}

impl StrokePoint {
    pub fn lerp(&self, other: &Self, factor: f32) -> Self {
        let inv_factor = 1.0 - factor;

        let s = std::simd::f32x4::from_array([self.pos[0], self.pos[1], self.pressure, self.dist]);
        let o =
            std::simd::f32x4::from_array([other.pos[0], other.pos[1], other.pressure, other.dist]);
        // FMA is planned but unimplemented ;w;
        let n = s * std::simd::f32x4::splat(inv_factor) + (o * std::simd::f32x4::splat(factor));
        Self {
            pos: [n[0], n[1]],
            pressure: n[2],
            dist: n[3],
        }
    }
}
pub struct Stroke {
    /// Unique id during this execution of the program.
    /// We have 64 bits, this won't get exhausted anytime soon! :P
    id: FuzzID<Stroke>,
    brush: StrokeBrushSettings,
    points: Vec<StrokePoint>,
}
/// Decoupled data from header, stored in separate manager. Header managed by UI.
/// Stores the strokes generated from pen input, with optional render data inserted by renderer.
pub struct StrokeLayerData {
    strokes: Vec<ImmutableStroke>,
    undo_cursor_position: Option<usize>,
}
/// Collection of layer data (stroke contents and render data) mapped from ID
pub struct StrokeLayerManager {
    layers: std::collections::HashMap<WeakID<StrokeLayer>, StrokeLayerData>,
}
impl Default for StrokeLayerManager {
    fn default() -> Self {
        Self {
            layers: Default::default(),
        }
    }
}

pub struct ImmutableStroke {
    id: FuzzID<Stroke>,
    brush: StrokeBrushSettings,
    points: Arc<[StrokePoint]>,
}
impl From<Stroke> for ImmutableStroke {
    fn from(value: Stroke) -> Self {
        Self {
            id: value.id,
            brush: value.brush,
            points: value.points.into(),
        }
    }
}

#[derive(Clone)]
pub struct WeakStroke {
    id: WeakID<Stroke>,
    brush: StrokeBrushSettings,
    // Not a weak reference, despite the name. I hadn't thought of this til now x3
    // to me it makes sense that the existence of this weak reference should keep the
    // resource alive.
    points: Arc<[StrokePoint]>,
}
impl From<&ImmutableStroke> for WeakStroke {
    fn from(value: &ImmutableStroke) -> Self {
        Self {
            id: value.id.weak(),
            brush: value.brush.clone(),
            points: value.points.clone(),
        }
    }
}

struct DocumentSelections {
    pub cur_layer: Option<WeakID<StrokeLayer>>,
}
impl Default for DocumentSelections {
    fn default() -> Self {
        Self { cur_layer: None }
    }
}
struct Selections {
    pub cur_document: Option<WeakID<Document>>,
    pub document_selections: std::collections::HashMap<WeakID<Document>, DocumentSelections>,
    pub cur_brush: Option<brush::WeakBrushID>,
    pub brush_settings: StrokeBrushSettings,
    pub undos: std::sync::atomic::AtomicU32,
}
impl Default for Selections {
    fn default() -> Self {
        Self {
            cur_document: None,
            document_selections: Default::default(),
            cur_brush: None,
            brush_settings: StrokeBrushSettings {
                brush: brush::todo_brush().id().weak(),
                color_modulate: [0.0, 0.0, 0.0, 1.0],
                size_mul: 15.0,
                spacing_px: 0.75,
                is_eraser: false,
            },
            undos: 0.into(),
        }
    }
}

// Icky. with a planned client-server architecture, we won't have as many globals -w-;;
// (well, a server is still a global, but the interface will be much less hacked-)
struct Globals {
    stroke_layers: tokio::sync::RwLock<StrokeLayerManager>,
    documents: tokio::sync::RwLock<Vec<Document>>,
    selections: tokio::sync::RwLock<Selections>,
}
impl Globals {
    fn new() -> Self {
        Self {
            stroke_layers: tokio::sync::RwLock::new(Default::default()),
            documents: tokio::sync::RwLock::new(Vec::new()),
            selections: Default::default(),
        }
    }
    fn strokes(&'_ self) -> &'_ tokio::sync::RwLock<StrokeLayerManager> {
        &self.stroke_layers
    }
    fn documents(&'_ self) -> &'_ tokio::sync::RwLock<Vec<Document>> {
        &self.documents
    }
    fn selections(&'_ self) -> &'_ tokio::sync::RwLock<Selections> {
        &self.selections
    }
}
static GLOBALS: std::sync::OnceLock<Globals> = std::sync::OnceLock::new();
enum RenderMessage {
    SwitchDocument(WeakID<Document>),
    StrokeLayer {
        layer: WeakID<StrokeLayer>,
        kind: StrokeLayerRenderMessageKind,
    },
}
enum StrokeLayerRenderMessageKind {
    /// The blend settings (opacity, clip, mode) for the layer were modified.
    BlendChanged(Blend),
    /// A stroke was appended to the given layer.
    /// Could be sourced from redos or new data entirely.
    Append(WeakStroke),
    /// This number of strokes were trucated (undone or replaced)
    Truncate(usize),
}
async fn render_worker(
    renderer: Arc<render_device::RenderContext>,
    document_preview: Arc<document_viewport_proxy::DocumentViewportPreviewProxy>,
    mut render_recv: tokio::sync::mpsc::UnboundedReceiver<RenderMessage>,
) -> AnyResult<()> {
    let layer_render = stroke_renderer::StrokeLayerRenderer::new(renderer.clone())?;
    let blend_engine = blend::BlendEngine::new(renderer.device().clone())?;

    let globals = GLOBALS.get_or_init(Globals::new);

    let mut cached_renders =
        std::collections::HashMap::<WeakID<StrokeLayer>, stroke_renderer::RenderData>::new();
    // Keep track of layer states locally, as globals can be mutated out-of-step with render commands
    // We expect eventual consistency though!
    let mut weak_layer_strokes =
        std::collections::HashMap::<WeakID<StrokeLayer>, Vec<WeakStroke>>::new();

    // An event that was peeked and rejected during aggregation.
    // Take it the next time around.
    let mut peeked = None;

    let mut document_to_draw = None;

    // Take the peeked value, or try to receive new value.
    while let Some(message) = match peeked.take() {
        Some(v) => {
            // We skip awaiting the reciever, and have more work immediately.
            // yield so that we don't starve the runtime.
            tokio::task::yield_now().await;
            Some(v)
        }
        None => render_recv.recv().await,
    } {
        // Try to aggregate more events, to batch work effectively.
        let mut events = vec![message];
        let mut skip_blend = false;
        loop {
            match render_recv.try_recv() {
                Ok(v) => {
                    // Accept value, or put into `peeked` and break if it
                    // is incompatible with aggregated work

                    // unwrap ok - vec always has at least one element.
                    let can_aggregate = match (events.last().unwrap(), &v) {
                        // Multiple edits to the same layer can be aggregated
                        (
                            RenderMessage::StrokeLayer { layer: target, .. },
                            RenderMessage::StrokeLayer { layer: new, .. },
                        ) if new == target => true,
                        // Multiple document switches can be aggregated
                        // (all are ignored but the last one)
                        (RenderMessage::SwitchDocument(..), RenderMessage::SwitchDocument(..)) => {
                            true
                        }
                        // Defer switch to next pass, but the switching document means we can skip
                        // re-rendering the document after the stroke layer commands.
                        (RenderMessage::StrokeLayer { .. }, RenderMessage::SwitchDocument(..)) => {
                            skip_blend = true;
                            false
                        }
                        _ => {
                            // Incompatible events. Defer event to next pass, and break
                            false
                        }
                    };

                    if can_aggregate {
                        events.push(v);
                    } else {
                        // peeked is always None here, as we took it above. no events lost :3
                        peeked = Some(v);
                        break;
                    }
                }
                Err(_) => {
                    // Empty or closed, we don't care - break and work
                    // on what events we managed to collect (always at least one)
                    break;
                }
            }
        }
        match events.first().unwrap() {
            // All our commands are layer commands relating to this target:
            RenderMessage::StrokeLayer { layer: target, .. } => {
                // clear the layer image before rendering? true if undone past the cached version.
                let mut clear = false;
                // keep track of rendering work to do. May include old strokes too,
                // if the truncate commands put us back in time before the cached image.
                let mut strokes_to_render = Vec::<WeakStroke>::new();
                let layer_strokes = weak_layer_strokes.entry(target.clone()).or_default();

                // Keep track of the blend changes. Only the last will apply.
                // None at the end of iteration means it hasn't changed - read from
                // document data directly.
                let mut new_blend = None;
                for event in events.iter() {
                    match event {
                        RenderMessage::StrokeLayer { layer, kind } => {
                            assert!(layer == target);

                            match kind {
                                StrokeLayerRenderMessageKind::Append(s) => {
                                    strokes_to_render.push(s.clone());
                                    layer_strokes.push(s.clone());
                                }
                                StrokeLayerRenderMessageKind::Truncate(num) => {
                                    let num_to_render = strokes_to_render.len();

                                    // Remove strokes from render queue
                                    strokes_to_render.drain(num_to_render.saturating_sub(*num)..);

                                    let global_strokes_truncate = layer_strokes
                                        .len()
                                        .checked_sub(*num)
                                        .expect("Cant truncate past empty");

                                    layer_strokes.drain(global_strokes_truncate..);

                                    // we took as many as possible from new commands - how many left over to remove?
                                    // if more than zero, we have to fetch strokes from globals.
                                    let num = num.saturating_sub(num_to_render);
                                    if num > 0 {
                                        // We're no longer strictly adding new content, we're removing what's
                                        // already been drawn. Clear the image.
                                        clear = true;
                                        strokes_to_render.extend(layer_strokes.iter().cloned());
                                    }
                                }
                                StrokeLayerRenderMessageKind::BlendChanged(blend) => {
                                    new_blend = Some(blend);
                                }
                            }
                        }
                        _ => panic!("Incorrect render command aggregation!"),
                    }
                }

                if !clear && strokes_to_render.is_empty() && new_blend.is_none() {
                    // No work to do. Go listen for new events.
                    continue;
                }

                let mut draw_semaphore_future = None;

                if clear && strokes_to_render.len() == 0 {
                    // Clear without remaking - just delete the data.
                    cached_renders.remove(&target);
                } else {
                    // get or try insert with - create buffer if it doesn't exist, and get a ref to it.
                    let cache_entry = match cached_renders.entry(target.clone()) {
                        std::collections::hash_map::Entry::Occupied(occupied) => {
                            &*occupied.into_mut()
                        }
                        std::collections::hash_map::Entry::Vacant(vacant) => {
                            &*vacant.insert(layer_render.uninit_render_data()?)
                        }
                    };
                    match layer_render.draw(&strokes_to_render, cache_entry, clear) {
                        Ok(semaphore) => draw_semaphore_future = Some(semaphore),
                        Err(e) => {
                            log::warn!("{e:?}");
                        }
                    };
                }

                // implicit sync: if skip_blend or document_to_draw.is_none the draw_semaphore_future will not be awaited.
                // next draw to this layer may need to be synchronized. Vulkano handles this in the background, but
                // it's an important subtlety so... note for future self!

                if !skip_blend {
                    if let Some(document_to_draw) = document_to_draw {
                        // wawawawawaaaw
                        // let blend = new_blend.unwrap();
                        // collect blend info from realtime document.
                        // need to track this locally, this info could be from wayyy in the future.
                        let blend_info: Vec<_> = {
                            let read = globals.documents.read().await;
                            let Some(doc) =
                                read.iter().find(|doc| doc.id.weak() == document_to_draw)
                            else {
                                // Document not found, nothin to draw.
                                continue;
                            };

                            doc.layer_top_level
                                .iter()
                                .filter_map(|layer| {
                                    Some((
                                        layer.blend.clone(),
                                        cached_renders.get(&layer.id.weak())?.view.clone(),
                                    ))
                                })
                                .collect()
                        };

                        let proxy = document_preview.write().await;
                        let commands = blend_engine.blend(
                            &renderer,
                            proxy.clone(),
                            true,
                            &blend_info,
                            [0; 2],
                            [0; 2],
                        )?;

                        let fence = match draw_semaphore_future {
                            Some(semaphore) => semaphore
                                .then_execute(
                                    renderer.queues().compute().queue().clone(),
                                    commands,
                                )?
                                .boxed_send(),
                            None => renderer
                                .now()
                                .then_execute(
                                    renderer.queues().compute().queue().clone(),
                                    commands,
                                )?
                                .boxed_send(),
                        }
                        .then_signal_fence_and_flush()?;
                        proxy.submit_with_fence(fence);
                    }
                }
            }
            // All our commands are document switch commands:
            // (we only care about the final one)
            RenderMessage::SwitchDocument(..) => {
                let RenderMessage::SwitchDocument(new_document) = events.last().unwrap() else {
                    panic!("Incorrect render command aggregation!")
                };
                document_to_draw = Some(new_document.clone());

                // <rerender viewport>
                let blend_info: Vec<_> = {
                    let read = globals.documents.read().await;
                    let Some(doc) = read.iter().find(|doc| doc.id.weak() == *new_document) else {
                        // Document not found, nothin to draw.
                        continue;
                    };

                    doc.layer_top_level
                        .iter()
                        .filter_map(|layer| {
                            Some((
                                layer.blend.clone(),
                                cached_renders.get(&layer.id.weak())?.view.clone(),
                            ))
                        })
                        .collect()
                };

                let proxy = document_preview.write().await;
                let commands = blend_engine.blend(
                    &renderer,
                    proxy.clone(),
                    true,
                    &blend_info,
                    [0; 2],
                    [0; 2],
                )?;

                let fence = renderer
                    .now()
                    .then_execute(renderer.queues().compute().queue().clone(), commands)?
                    .boxed_send()
                    .then_signal_fence_and_flush()?;
                proxy.submit_with_fence(fence);
            }
        }
    }
    Ok(())
}
async fn stylus_event_collector(
    mut event_stream: tokio::sync::broadcast::Receiver<stylus_events::StylusEventFrame>,
    mut action_listener: actions::ActionListener,
    document_preview: Arc<document_viewport_proxy::DocumentViewportPreviewProxy>,
    render_send: tokio::sync::mpsc::UnboundedSender<RenderMessage>,
) -> AnyResult<()> {
    let globals = GLOBALS.get_or_init(Globals::new);
    // Create a document and a few layers and select them, to speed up testing iterations :P
    {
        let (default_document, default_layer) = {
            let mut document = Document::default();
            let document_id = document.id.weak();

            let layer = StrokeLayer::default();
            let layer_id = layer.id.weak();
            document.layer_top_level.push(layer);

            globals.documents.write().await.push(document);
            (document_id, layer_id)
        };

        let mut selections = globals.selections().write().await;
        selections.cur_document = Some(default_document);
        let document_selections = selections
            .document_selections
            .entry(default_document)
            .or_default();
        document_selections.cur_layer = Some(default_layer);
    }

    let mut last_document = None;

    let mut current_stroke = None::<Stroke>;

    loop {
        match event_stream.recv().await {
            Ok(event_frame) => {
                // We need a transform in order to do any of our work!
                let Some(transform) = document_preview.get_view_transform().await else {
                    continue;
                };

                // Collect all the data we need from selections global immediately:
                let (cur_document, cur_layer, cur_brush, ui_undos) = {
                    let read = globals.selections().read().await;
                    let Some(cur_document) = read.cur_document else {
                        continue;
                    };
                    let Some(cur_layer) = read
                        .document_selections
                        .get(&cur_document)
                        .and_then(|selections| selections.cur_layer)
                    else {
                        continue;
                    };
                    let brush = read.brush_settings.clone();

                    let undos = read.undos.swap(0, std::sync::atomic::Ordering::Relaxed) as usize;

                    (cur_document, cur_layer, brush, undos)
                };

                if Some(cur_document) != last_document {
                    last_document = Some(cur_document.clone());
                    // Notify renderer of the change
                    render_send.send(RenderMessage::SwitchDocument(cur_document.clone()))?;
                }

                let frame = action_listener.frame().ok();
                let (key_undos, key_redos, is_pan, is_scrub, is_mirror) = match frame {
                    None => (0, 0, false, false, false),
                    Some(f) => (
                        f.action_trigger_count(actions::Action::Undo),
                        f.action_trigger_count(actions::Action::Redo),
                        f.is_action_held(actions::Action::ViewportPan),
                        f.is_action_held(actions::Action::ViewportScrub),
                        // An odd number of mirror requests, we assume two mirror requests cancel out
                        f.action_trigger_count(actions::Action::ViewportFlipHorizontal) % 2 == 1,
                    ),
                };

                // Assume redos cancel out undos
                let net_undos = (key_undos + ui_undos) as isize - (key_redos as isize);

                if net_undos != 0 {
                    let mut stroke_manager = globals.strokes().write().await;

                    let layer_data =
                        stroke_manager
                            .layers
                            .entry(cur_layer)
                            .or_insert_with(|| StrokeLayerData {
                                strokes: vec![],
                                undo_cursor_position: None,
                            });

                    match (layer_data.undo_cursor_position, net_undos) {
                        // Redone when nothing undone - do nothin!
                        (None, ..=0) => (),
                        // New undos!
                        (None, undos @ 1..) => {
                            // `as` cast ok - we checked that it's positive.
                            // clamp undos to the size of the data.
                            let undos = layer_data.strokes.len().min(undos as usize);
                            // Subtract won't overflow
                            layer_data.undo_cursor_position =
                                Some(layer_data.strokes.len() - undos);
                            // broadcast the truncation request
                            render_send.send(RenderMessage::StrokeLayer {
                                layer: cur_layer,
                                kind: StrokeLayerRenderMessageKind::Truncate(undos),
                            })?;
                        }
                        // Redos when there were outstanding undos
                        (Some(old_cursor), negative_redos @ ..=0) => {
                            // weird conventions I invented here :V
                            // `as` cast ok - we checked that it's negative, and inverted it.
                            let redos = (-negative_redos) as usize;
                            let new_cursor = old_cursor.saturating_add(redos);

                            // This many redos will move cursor past the end
                            let bounds = if new_cursor >= layer_data.strokes.len() {
                                layer_data.undo_cursor_position = None;
                                // append all strokes from the cursor onward
                                old_cursor..layer_data.strokes.len()
                            } else {
                                layer_data.undo_cursor_position = Some(new_cursor);

                                // append all strokes from the old cursor until the new
                                old_cursor..new_cursor
                            };

                            // Tell the renderer to append these strokes once more
                            for stroke in &layer_data.strokes[bounds] {
                                render_send.send(RenderMessage::StrokeLayer {
                                    layer: cur_layer,
                                    kind: StrokeLayerRenderMessageKind::Append(stroke.into()),
                                })?;
                            }
                        }
                        // There were outstanding undos, and we're adding to them.
                        (Some(old_cursor), undos @ 1..) => {
                            // Prevent undos from taking index below 0
                            // `as` cast ok - we checked that it's positive
                            let undos = old_cursor.min(undos as usize);

                            // Won't overflow
                            layer_data.undo_cursor_position = Some(old_cursor - undos);

                            render_send.send(RenderMessage::StrokeLayer {
                                layer: cur_layer,
                                kind: StrokeLayerRenderMessageKind::Truncate(undos),
                            })?;
                        }
                        // All cases are handled.
                        _ => unreachable!(),
                    }
                }

                if is_pan || is_scrub {
                    // treat stylus events as viewport movement
                    todo!()
                } else {
                    // treat stylus as new strokes
                    for event in event_frame.iter() {
                        if event.pressed {
                            // Get stroke-in-progress or start anew.
                            let this_stroke = current_stroke.get_or_insert_with(|| Stroke {
                                brush: cur_brush.clone(),
                                id: Default::default(),
                                points: Vec::new(),
                            });
                            let Ok(pos) =
                                transform.unproject(cgmath::point2(event.pos.0, event.pos.1))
                            else {
                                // If transform is ill-formed, we can't do work.
                                continue;
                            };

                            // Calc cumulative distance from the start, or 0.0 if this is the first point.
                            let dist = this_stroke
                                .points
                                .last()
                                .map(|last| {
                                    let delta = [last.pos[0] - pos.x, last.pos[1] - pos.y];
                                    last.dist + (delta[0] * delta[0] + delta[1] * delta[1]).sqrt()
                                })
                                .unwrap_or(0.0);

                            this_stroke.points.push(StrokePoint {
                                pos: [pos.x, pos.y],
                                pressure: event.pressure.unwrap_or(1.0),
                                dist,
                            })
                        } else {
                            if let Some(stroke) = current_stroke.take() {
                                // Not pressed and a stroke exists - take it, freeze it, and put it on current layer!
                                let immutable: ImmutableStroke = stroke.into();

                                let mut stroke_manager = globals.strokes().write().await;

                                let layer_data = stroke_manager
                                    .layers
                                    .entry(cur_layer)
                                    .or_insert_with(|| StrokeLayerData {
                                        strokes: vec![],
                                        undo_cursor_position: None,
                                    });

                                // If there was an undo cursor, truncate everything after
                                // and replace with new data.
                                if let Some(cursor) = layer_data.undo_cursor_position.take() {
                                    layer_data.strokes.drain(cursor..);
                                }

                                let weak_ref = (&immutable).into();
                                layer_data.strokes.push(immutable);

                                render_send.send(RenderMessage::StrokeLayer {
                                    layer: cur_layer,
                                    kind: StrokeLayerRenderMessageKind::Append(weak_ref),
                                })?;
                            }
                        }
                    }
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(num)) => {
                log::warn!("Lost {num} stylus frames!");
            }
            // Stream closed, no more data to handle - we're done here!
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    #[cfg(feature = "dhat_heap")]
    let _profiler = {
        log::trace!("Installed dhat");
        dhat::Profiler::new_heap()
    };
    env_logger::builder()
        .filter_level(log::LevelFilter::max())
        .init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    if let Err(e) = GlobalHotkeys::get().save() {
        log::warn!("Failed to save hotkey config:\n{e:?}");
    };

    GLOBALS.get_or_init(Globals::new);

    // Test image generators.
    //let (image, future) = make_test_image(render_context.clone())?;
    //let (image, future) = load_document_image(render_context.clone(), &std::path::PathBuf::from("/home/aspen/Pictures/thesharc.png"))?;
    //future.wait(None)?;

    let document_view = Arc::new(document_viewport_proxy::DocumentViewportPreviewProxy::new(
        &render_surface,
    )?);
    let window_renderer = window_surface.with_render_surface(
        render_surface,
        render_context.clone(),
        document_view.clone(),
    )?;

    let event_stream = window_renderer.stylus_events();
    let action_listener = window_renderer.action_listener();

    std::thread::spawn(move || {
        #[cfg(feature = "dhat_heap")]
        // Keep alive. Winit takes ownership of main, and will never
        // drop this unless we steal it.
        let _profiler = _profiler;

        let result = {
            // We don't expect this channel to get very large, but it's important
            // that messages don't get lost under any circumstance, lest an expensive
            // document rebuild be needed :P
            let (render_sender, render_reciever) =
                tokio::sync::mpsc::unbounded_channel::<RenderMessage>();

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            // between current_thread runtime and try_join, these tasks are
            // not actually run in parallel, just interleaved. This is preferable
            // for now, just a note for future self UwU
            runtime.block_on(async {
                tokio::try_join!(
                    render_worker(render_context, document_view.clone(), render_reciever,),
                    stylus_event_collector(
                        event_stream,
                        action_listener,
                        document_view,
                        render_sender,
                    ),
                )
            })
        };
        if let Err(e) = result {
            log::error!("Helper task exited with err, runtime terminated:\n{e:?}")
        }
    });

    window_renderer.run();
}

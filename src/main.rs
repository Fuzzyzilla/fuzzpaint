#![feature(portable_simd)]
use std::sync::Arc;
mod egui_impl;
pub mod gpu_err;
pub mod vulkano_prelude;
pub mod window;
use brush::BrushStyle;
use cgmath::{Matrix4, SquareMatrix};
use vulkano::command_buffer;
use vulkano_prelude::*;
pub mod blend;
pub mod brush;
pub mod document_viewport_proxy;
pub mod id;
pub mod render_device;
pub mod stylus_events;
pub mod tess;
pub mod gpu_tess;
use blend::{Blend, BlendMode};

pub use id::{FuzzID, WeakID};
pub use tess::StrokeTessellator;

/// Obviously will be user specified on a per-document basis, but for now...
const DOCUMENT_DIMENSION: u32 = 1024;
/// Premultiplied RGBA16F for interesting effects (negative + overbright colors and alpha) with
/// more than 11bit per channel precision in the [0,1] range.
/// Will it be user specified in the future?
const DOCUMENT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

use anyhow::Result as AnyResult;

pub struct GroupLayer {
    name: String,

    /// Some - grouped rendering, None - Passthrough
    blend: Option<blend::Blend>,

    /// ID that is unique within this execution of the program
    id: FuzzID<GroupLayer>,
}
impl Default for GroupLayer {
    fn default() -> Self {
        let id = FuzzID::default();
        Self {
            name: format!("Group {}", id.id()),
            id,
            blend: None,
        }
    }
}
pub struct StrokeLayer {
    name: String,
    blend: blend::Blend,

    /// ID that is unique within this execution of the program
    id: FuzzID<StrokeLayer>,
}

impl Default for StrokeLayer {
    fn default() -> Self {
        let id = FuzzID::default();
        Self {
            name: format!("Layer {}", id.id().wrapping_add(1)),
            id,
            blend: Default::default(),
        }
    }
}

pub enum LayerNode {
    Group {
        layer: GroupLayer,
        // Make a tree in rust without unsafe challenge ((very hard))
        children: std::cell::UnsafeCell<Vec<LayerNode>>,
        id: FuzzID<LayerNode>,
    },
    StrokeLayer {
        layer: StrokeLayer,
        id: FuzzID<LayerNode>,
    },
}
impl LayerNode {
    pub fn id(&self) -> &FuzzID<LayerNode> {
        match self {
            Self::Group { id, .. } => id,
            Self::StrokeLayer { id, .. } => id,
        }
    }
}

impl From<GroupLayer> for LayerNode {
    fn from(layer: GroupLayer) -> Self {
        Self::Group {
            layer,
            children: Vec::new().into(),
            id: Default::default(),
        }
    }
}
impl From<StrokeLayer> for LayerNode {
    fn from(layer: StrokeLayer) -> Self {
        Self::StrokeLayer {
            layer,
            id: Default::default(),
        }
    }
}

pub struct LayerGraph {
    top_level: Vec<LayerNode>,
}
impl LayerGraph {
    // Maybe this is a silly way to do things. It ended up causing a domino effect that caused the need
    // for unsafe code, maybe I should rethink this. Regardless, it's an implementation detail, so it'll do for now.
    fn find_recurse<'a>(
        &'a self,
        traverse_stack: &mut Vec<&'a LayerNode>,
        at: WeakID<LayerNode>,
    ) -> bool {
        //Find search candidates
        let nodes_to_search = if traverse_stack.is_empty() {
            self.top_level.as_slice()
        } else {
            // Return false if last element is not a group (shouldn't occur)
            let LayerNode::Group { children, .. } = traverse_stack.last().clone().unwrap() else {
                return false;
            };

            //Safety - We hold an immutable reference to self, thus no mutable access to `children` can occur as well.
            unsafe { &*children.get() }.as_slice()
        };

        for node in nodes_to_search.iter() {
            //Found it!
            if node.id() == at {
                traverse_stack.push(node);
                return true;
            }

            //Traverse deeper...
            match node {
                LayerNode::Group { .. } => {
                    traverse_stack.push(node);
                    if self.find_recurse(traverse_stack, at) {
                        return true;
                    }
                    //Done traversing subtree and it wasn't found, remove subtree.
                    traverse_stack.pop();
                }
                _ => (),
            }
        }

        // Did not find it and did not early return, must not have been found.
        return false;
    }
    /// Find the given layer ID in the tree, returning the path to it, if any.
    /// If a path is returned, the final element will be the layer itself.
    fn find<'a>(&'a self, at: WeakID<LayerNode>) -> Option<Vec<&'a LayerNode>> {
        let mut traverse_stack = Vec::new();

        if self.find_recurse(&mut traverse_stack, at) {
            Some(traverse_stack)
        } else {
            None
        }
    }
    fn insert_node(&mut self, node: LayerNode) {
        self.top_level.push(node);
    }
    fn insert_node_at(&mut self, at: WeakID<LayerNode>, node: LayerNode) {
        match self.find(at) {
            None => self.insert_node(node),
            Some(path) => {
                match path.last().unwrap() {
                    LayerNode::Group { children, .. } => {
                        //`at` is a group - insert as highest child of `at`.
                        //Forget borrows
                        drop(path);
                        //reinterprit as mutable (uh oh)
                        //Safety - We hold exclusive access to self, thus no concurrent access to the tree can occur
                        //and no other references exist.
                        let children = unsafe { &mut *children.get() };

                        children.push(node);
                    }
                    _ => {
                        //`at` is something else - insert on the same level, immediately above `at`.'

                        //Parent is just top level
                        let siblings = if path.len() < 2 {
                            drop(path);
                            &mut self.top_level
                        } else {
                            //Find siblings
                            let Some(LayerNode::Group {
                                children: siblings, ..
                            }) = path.get(path.len() - 2)
                            else {
                                //(should be impossible) parent doesn't exist or isn't a group, add to top level instead.
                                self.insert_node(node);
                                return;
                            };

                            drop(path);

                            //reinterprit as mutable (uh oh)
                            //Safety - We hold exclusive access to self, thus no concurrent access to the tree can occur
                            //and no other references exist.
                            unsafe { &mut *siblings.get() }
                        };

                        //Find idx of `at`
                        let Some((idx, _)) = siblings
                            .iter()
                            .enumerate()
                            .find(|(_, node)| node.id() == at)
                        else {
                            //`at` isn't a child of `at`'s parent - should be impossible! add to top of siblings instead.
                            siblings.push(node);
                            return;
                        };

                        //Insert after idx.
                        siblings.insert(idx, node);
                    }
                }
            }
        }
    }
    /// Insert the group at the highest position of the top level
    pub fn insert_layer(&mut self, layer: impl Into<LayerNode>) -> WeakID<LayerNode> {
        let node: LayerNode = layer.into();
        let node_id = node.id().weak();
        self.insert_node(node);
        node_id
    }
    /// Insert the group at the given position
    /// If the position is a group, insert at the highest position in the group
    /// If the position is a layer, insert above it.
    /// If the position doesn't exist, behaves as `insert_group`.
    pub fn insert_layer_at(
        &mut self,
        at: WeakID<LayerNode>,
        layer: impl Into<LayerNode>,
    ) -> WeakID<LayerNode> {
        let node: LayerNode = layer.into();
        let node_id = node.id().weak();
        self.insert_node_at(at, node);
        node_id
    }
    pub fn find_mut_children_of<'a>(
        &'a mut self,
        parent: WeakID<LayerNode>,
    ) -> Option<&'a mut [LayerNode]> {
        let path = self.find(parent)?;

        // Get children, or return none if not found or not a group
        let Some(LayerNode::Group { children, .. }) = path.last() else {
            return None;
        };

        unsafe {
            // Safety - the return value continues to mutably borrow self,
            // so no other access can occur.
            Some((*children.get()).as_mut_slice())
        }
    }
    /// Remove and return the node of the given ID. None if not found.
    pub fn remove(&mut self, at: WeakID<LayerNode>) -> Option<LayerNode> {
        let path = self.find(at)?;

        //Parent is top-level
        if path.len() < 2 {
            let (idx, _) = self
                .top_level
                .iter()
                .enumerate()
                .find(|(_, node)| node.id() == at)?;
            Some(self.top_level.remove(idx))
        } else {
            let LayerNode::Group {
                children: siblings, ..
            } = path.get(path.len() - 2)?
            else {
                return None;
            };

            //Safety - has exclusive access to self, so the graph cannot be concurrently accessed
            unsafe {
                let siblings = &mut *siblings.get();

                let (idx, _) = siblings
                    .iter()
                    .enumerate()
                    .find(|(_, node)| node.id() == at)?;

                Some(siblings.remove(idx))
            }
        }
    }
}
impl Default for LayerGraph {
    fn default() -> Self {
        Self {
            top_level: Vec::new(),
        }
    }
}

pub struct Document {
    /// The path from which the file was loaded, or None if opened as new.
    path: Option<std::path::PathBuf>,
    /// Name of the document, from its path or generated.
    name: String,

    /// Layers that make up this document
    layers: LayerGraph,

    /// ID that is unique within this execution of the program
    id: FuzzID<Document>,
}
impl Default for Document {
    fn default() -> Self {
        let id = FuzzID::default();
        Self {
            path: None,
            layers: Default::default(),
            name: format!("New Document {}", id.id()),
            id,
        }
    }
}
struct PerDocumentInterface {
    zoom: f32,
    rotate: f32,

    focused_subtree: Option<WeakID<LayerNode>>,
    cur_layer: Option<WeakID<LayerNode>>,
}
impl Default for PerDocumentInterface {
    fn default() -> Self {
        Self {
            zoom: 100.0,
            rotate: 0.0,
            cur_layer: None,
            focused_subtree: None,
        }
    }
}
pub struct DocumentUserInterface {
    color: egui::Color32,

    // modal_stack: Vec<Box<dyn FnMut(&mut egui::Ui) -> ()>>,
    cur_brush: Option<brush::WeakBrushID>,
    brushes: Vec<brush::Brush>,

    cur_document: Option<WeakID<Document>>,
    documents: Vec<Document>,

    document_interfaces: std::collections::HashMap<WeakID<Document>, PerDocumentInterface>,

    viewport: egui::Rect,
}
impl Default for DocumentUserInterface {
    fn default() -> Self {
        Self {
            color: egui::Color32::BLUE,
            //modal_stack: Vec::new(),
            cur_brush: None,
            brushes: Vec::new(),

            cur_document: None,
            documents: Vec::new(),
            document_interfaces: Default::default(),

            viewport: egui::Rect {
                min: egui::Pos2::ZERO,
                max: egui::Pos2::ZERO,
            },
        }
    }
}

impl DocumentUserInterface {
    fn target_layer(&self) -> Option<WeakID<LayerNode>> {
        let id = self.cur_document?;
        // Selected layer of the currently focused document, if any
        self.document_interfaces.get(&id)?.cur_layer
    }

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
    fn ui_layer_slice(
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
    }
    /*
    fn do_modal(&mut self, ctx: &egui::Context, add_contents: &Box<dyn FnMut(&mut egui::Ui) -> ()>) {
        egui::Area::new("Modal")
            .order(egui::Order::TOP)
            .movable(true)
            .show(&ctx, |ui| {
                egui::Frame::window(ui.style())
                    .show(ui, *add_contents)
            });
    }
    fn push_modal(&mut self, add_contents: impl FnMut(&mut egui::Ui) -> ()) {
        self.modal_stack.push(Box::new(add_contents))
    }*/
    pub fn ui(&mut self, ctx: &egui::Context) {
        /*
        if !self.modal_stack.is_empty() {
            for modal in self.modal_stack.iter() {
                self.do_modal(&ctx, modal);
            }
        }*/

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
                            self.documents.push(document);
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
                let Some(document) = self.cur_document else {
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

                let _ = ui.button("⮪");
            });
        });

        egui::SidePanel::right("Layers").show(&ctx, |ui| {
            ui.label("Layers");
            ui.separator();

            // if there is no current document, there is nothing for us to do here
            let Some(document_id) = self.cur_document else {
                return;
            };

            // Find the document, otherwise clear selection
            let Some(document) = self.documents.iter_mut().find(|doc| &doc.id == document_id)
            else {
                self.cur_document = None;
                return;
            };

            let document_interface = self.document_interfaces.entry(document_id).or_default();

            ui.horizontal(|ui| {
                if ui.button("➕").clicked() {
                    let layer = StrokeLayer::default();
                    if let Some(selected) = document_interface.cur_layer {
                        document.layers.insert_layer_at(selected, layer);
                    } else {
                        document.layers.insert_layer(layer);
                    }
                }
                if ui.button("🗀").clicked() {
                    let group = GroupLayer::default();
                    if let Some(selected) = document_interface.cur_layer {
                        document.layers.insert_layer_at(selected, group);
                    } else {
                        document.layers.insert_layer(group);
                    }
                }
                let _ = ui.button("⤵").on_hover_text("Merge down");
                if ui.button("✖").on_hover_text("Delete layer").clicked() {
                    if let Some(layer_id) = document_interface.cur_layer.take() {
                        document.layers.remove(layer_id);
                    }
                };
            });

            if let Some(subtree) = document_interface.focused_subtree {
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("⬅").clicked() {
                        document_interface.focused_subtree = None;
                    }
                    ui.label(format!("Viewing subtree of {subtree}"));
                });
            }

            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
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
                }
            });
        });

        egui::SidePanel::left("Color picker").show(&ctx, |ui| {
            ui.label("Color");
            ui.separator();

            egui::color_picker::color_picker_color32(
                ui,
                &mut self.color,
                egui::color_picker::Alpha::OnlyBlend,
            );

            ui.separator();
            ui.label("Brushes");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("➕").clicked() {
                    let brush = brush::Brush::default();
                    self.brushes.push(brush);
                }
                if ui.button("✖").on_hover_text("Delete brush").clicked() {
                    if let Some(id) = self.cur_brush.take() {
                        self.brushes.retain(|brush| brush.id() != id);
                    }
                };
                let mut erase = false;
                ui.toggle_value(&mut erase, "Erase");
            });
            ui.separator();

            for brush in self.brushes.iter_mut() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.cur_brush, Some(brush.id().weak()), "");
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
                                brush::BrushStyle::Stamped { spacing } => {
                                    let slider = egui::widgets::Slider::new(spacing, 0.1..=200.0)
                                        .clamp_to_range(true)
                                        .logarithmic(true)
                                        .suffix("px");

                                    ui.add(slider);
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
                    for document in self.documents.iter() {
                        egui::containers::Frame::group(ui.style())
                            .outer_margin(egui::Margin::symmetric(0.0, 0.0))
                            .inner_margin(egui::Margin::symmetric(0.0, 0.0))
                            .multiply_with_opacity(
                                if self.cur_document == Some(document.id.weak()) {
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
                                    &mut self.cur_document,
                                    Some(document.id.weak()),
                                    &document.name,
                                );
                                if ui.small_button("✖").clicked() {
                                    deleted_ids.push(document.id.weak());
                                    //Disselect if deleted.
                                    if self.cur_document == Some(document.id.weak()) {
                                        self.cur_document = None;
                                    }
                                }
                            })
                            .response
                            .on_hover_ui(|ui| {
                                ui.label(format!("{}", document.id));
                            });
                    }
                    self.documents
                        .retain(|document| !deleted_ids.contains(&document.id.weak()));
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
    use crate::gpu_tess;
    /// The data managed by the renderer.
    /// For now, in persuit of actually getting a working product one day,
    /// this is a very coarse caching sceme. In the future, perhaps a bit more granular
    /// control can occur, should performance become an issue:
    ///  * Caching images of incrementally older states, reducing work to get to any given state (performant undo)
    ///  * Caching tesselation output
    pub struct RenderData {
        image: Arc<vk::StorageImage>,
        view: Arc<vk::ImageView<vk::StorageImage>>,
    }

    use crate::{
        tess::{StreamStrokeTessellator, TessellatedStrokeVertex},
        vk,
    };
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

        /// Three host-visible buffers of size TESS_BUFFER_SIZE for swapping work between
        /// Graphics pipeline and cpu tesselator
        tess_buffer: Arc<vk::Buffer>,
        /// For now, only single access. Future could have more granular access control.
        tess_subbuffer: std::sync::Mutex<vk::Subbuffer<[TessellatedStrokeVertex]>>,

        render_pass: Arc<vk::RenderPass>,
        pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl StrokeLayerRenderer {
        /// Size of one tess buffer. Must align to nonCoherentAtomSize.
        /// nonCoherentAtomSize is always a power of two and universally < 1kiB,
        /// so 4MiB will certainly always work.
        const TESS_BUFFER_VERTS: u64 = 1024 * 1024;
        const TESS_BUFFER_SIZE: u64 =
            Self::TESS_BUFFER_VERTS * std::mem::size_of::<TessellatedStrokeVertex>() as u64;
        const TESS_BUFFER_COUNT: u64 = 3;

        pub fn new(context: Arc<crate::render_device::RenderContext>) -> AnyResult<Self> {
            let render_pass = vulkano::single_pass_renderpass!(
                context.device().clone(),
                attachments: {
                    output_image: {
                        // TODO: An optimized render sequence could involve this being a load.
                        load: Clear,
                        store: Store,
                        format: crate::DOCUMENT_FORMAT,
                        samples: 1,
                    },
                },
                pass: {
                    color: [output_image],
                    depth_stencil: {},
                },
            )?;

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

            // Safety: this spirv binary is generated by `shaderc` so it should be fine. `from_bytes` does some cursory checks,
            // which ensures I don't accidentally use a asm file or somethin x3
            let frag = unsafe {
                vulkano::shader::ShaderModule::from_bytes(context.device().clone(), include_bytes!("shaders/stamp.frag.spv"))
            }?;
            let vert = vert::load(context.device().clone())?;
            // Unwraps ok here, using GLSL where "main" is the only allowed entry point.
            let frag = frag.entry_point("main").unwrap();
            let vert = vert.entry_point("main").unwrap();

            // DualSrcBlend (~75% coverage) is used to control whether to erase or draw on a per-fragment basis
            // [1.0; 4] = draw, [0.0; 4] = erase.
            let mut premul_dyn_constants = vk::ColorBlendState::new(1);
            premul_dyn_constants.blend_constants = vk::StateMode::Dynamic;
            premul_dyn_constants.attachments[0].blend = Some(vk::AttachmentBlend {
                alpha_source: vulkano::pipeline::graphics::color_blend::BlendFactor::Src1Color,
                color_source: vulkano::pipeline::graphics::color_blend::BlendFactor::Src1Alpha,
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
                .render_pass(render_pass.clone().first_subpass())
                .build(context.device().clone())?;

            let descriptor_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0, image, sampler,
                )],
            )?;

            // Allocate tesselation buffer
            // Align to a nonCoherentAtomSize (may be greater than 64, vulkano no likey that)
            let non_coherent_atom_size = context
                .device()
                .physical_device()
                .properties()
                .non_coherent_atom_size;
            if Self::TESS_BUFFER_SIZE % non_coherent_atom_size.as_devicesize() != 0 {
                anyhow::bail!("Tess buffer size is not aligned for non-coherently use")
            }
            // This is an unreasonable assumption - about 1/8 of devices will fail here. what do? todo!
            if non_coherent_atom_size.as_devicesize() > 64u64 {
                anyhow::bail!("Device nonCoherentAtomSize too strict to allocate")
            }
            let tess_buffer = vk::Buffer::new(
                context.allocators().memory(),
                vk::BufferCreateInfo {
                    usage: vk::BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    allocate_preference:
                        vulkano::memory::allocator::MemoryAllocatePreference::AlwaysAllocate,
                    usage: vulkano::memory::allocator::MemoryUsage::Upload,
                    ..Default::default()
                },
                vulkano::memory::allocator::DeviceLayout::new(
                    std::num::NonZeroU64::new(Self::TESS_BUFFER_SIZE * Self::TESS_BUFFER_COUNT)
                        .unwrap(),
                    // Panics if >64, checked earlier
                    non_coherent_atom_size,
                )
                .unwrap(),
            )?;
            // Subdivide the large buffer into TESS_BUFFER_COUNT smaller, [f32] buffers
            let whole_buffer = vk::Subbuffer::<[u8]>::new(tess_buffer.clone());
            let tess_subbuffer = whole_buffer
                .try_cast_slice::<TessellatedStrokeVertex>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;

            Ok(Self {
                context,
                pipeline,
                render_pass,
                tess_buffer,
                tess_subbuffer: tess_subbuffer.into(),
                texture_descriptor: descriptor_set,
            })
        }
        /// Allocate a new RenderData object. Initial contents are undefined!
        pub fn empty_render_data(&self) -> anyhow::Result<RenderData> {
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

            Ok(RenderData { image, view })
        }
    }
}

#[derive(Clone)]
pub struct StrokeBrushSettings {
    brush: brush::WeakBrushID,
    /// `a` is flow, NOT opacity, since the stroke is blended continuously not blended as a group.
    color_modulate: [f32; 4],
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
    render_data: Option<stroke_renderer::RenderData>,
}
/// Collection of layer data (stroke contents and render data) mapped from ID
pub struct StrokeLayerManager {
    layers: std::collections::HashMap<FuzzID<StrokeLayer>, StrokeLayerData>,
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
#[derive(Clone)]
pub struct WeakStroke {
    id: WeakID<Stroke>,
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
impl From<&ImmutableStroke> for WeakStroke {
    fn from(value: &ImmutableStroke) -> Self {
        Self {
            id: value.id.weak(),
            brush: value.brush.clone(),
            points: value.points.clone(),
        }
    }
}

// Icky. with a planned client-server architecture, we won't have as many globals -w-;;
// (well, a server is still a global, but the interface will be much less hacked-)
struct Globals {
    stroke_layers: tokio::sync::RwLock<StrokeLayerManager>,
    // Todo: LayerGraph (and thus Document) is currently !Sync due to UnsafeCell in the graph implementation.
    documents: tokio::sync::RwLock<Vec<()>>,
}
impl Globals {
    fn new() -> Self {
        Self {
            stroke_layers: tokio::sync::RwLock::new(Default::default()),
            documents: tokio::sync::RwLock::new(Vec::new()),
        }
    }
    fn strokes(&'_ self) -> &'_ tokio::sync::RwLock<StrokeLayerManager> {
        &self.stroke_layers
    }
    fn documents(&'_ self) -> &'_ tokio::sync::RwLock<Vec<()>> {
        &self.documents
    }
}
static GLOBALS: std::sync::OnceLock<Globals> = std::sync::OnceLock::new();

/// Proxy called into by the window renderer to perform the necessary synchronization and such to render the screen
/// behind the Egui content.
pub trait PreviewRenderProxy {
    /// Create the render commands for this frame. Assume used resources are borrowed until a matching "render_complete" for this
    /// frame idx is called.
    fn render(&mut self, swapchain_image_idx: u32) -> AnyResult<Arc<vk::PrimaryAutoCommandBuffer>>;

    /// When the future of a previous render has completed
    fn render_complete(&mut self, idx: u32);
    fn surface_changed(&mut self, render_surface: &render_device::RenderSurface);
}

fn listener(
    mut event_stream: tokio::sync::broadcast::Receiver<stylus_events::StylusEventFrame>,
    renderer: Arc<render_device::RenderContext>,
    document_preview: Arc<
        parking_lot::RwLock<document_viewport_proxy::DocumentViewportPreviewProxy>,
    >,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    runtime.block_on(async {
        let tesselator = tess::rayon::RayonTessellator;
        let mut current_stroke = None::<Stroke>;
        let brush = StrokeBrushSettings {
            brush: brush::todo_brush().id().weak(),
            color_modulate: [1.0; 4],
            size_mul: 5.0,
            is_eraser: false,
        };

        loop {
            match event_stream.recv().await {
                Ok(event_frame) => {
                    let matrix = { document_preview.read().get_matrix() }.invert().unwrap();
                    for event in event_frame.iter() {
                        if event.pressed {
                            // Get stroke-in-progress or start anew.
                            let this_stroke = current_stroke.get_or_insert_with(|| Stroke {
                                brush: brush.clone(),
                                id: Default::default(),
                                points: Vec::new(),
                            });
                            let pos = matrix * cgmath::vec4(event.pos.0, event.pos.1, 0.0, 1.0);
                            let pos = [
                                pos.x * DOCUMENT_DIMENSION as f32,
                                (1.0 - pos.y) * DOCUMENT_DIMENSION as f32,
                            ];

                            // Calc cumulative distance from the start, or 0.0 if this is the first point.
                            let dist = this_stroke
                                .points
                                .last()
                                .map(|last| {
                                    // I should really be using a linalg library lmao
                                    let delta = [last.pos[0] - pos[0], last.pos[1] - pos[1]];
                                    last.dist + (delta[0] * delta[0] + delta[1] * delta[1]).sqrt()
                                })
                                .unwrap_or(0.0);

                            this_stroke.points.push(StrokePoint {
                                pos,
                                pressure: event.pressure.unwrap_or(1.0),
                                dist,
                            })
                        } else {
                            if let Some(stroke) = current_stroke.take() {
                                // Not pressed and a stroke exists - take it, freeze it, and put it on current layer!
                                let immutable: ImmutableStroke = stroke.into();
                                todo!()
                            }
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(num)) => {
                    log::warn!("Lost {num} stylus frames!");
                }
                // Stream closed, no more data to handle - we're done here!
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
            }
        }
    })
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    env_logger::builder()
        .filter_level(log::LevelFilter::max())
        .init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    blend::BlendEngine::new(render_context.device().clone()).unwrap();

    GLOBALS.get_or_init(Globals::new);

    // Test image generators.
    //let (image, future) = make_test_image(render_context.clone())?;
    //let (image, future) = load_document_image(render_context.clone(), &std::path::PathBuf::from("/home/aspen/Pictures/thesharc.png"))?;
    //future.wait(None)?;

    let document_view = Arc::new(parking_lot::RwLock::new(
        document_viewport_proxy::DocumentViewportPreviewProxy::new(&render_surface)?,
    ));
    let window_renderer = window_surface.with_render_surface(
        render_surface,
        render_context.clone(),
        document_view.clone(),
    )?;

    let event_stream = window_renderer.stylus_events();

    std::thread::spawn(move || {
        listener(event_stream, render_context.clone(), document_view.clone())
    });

    window_renderer.run();
}

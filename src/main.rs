#![feature(portable_simd)]
use std::sync::Arc;
mod egui_impl;
pub mod gpu_err;
pub mod vulkano_prelude;
pub mod window;
use cgmath::{Matrix4, SquareMatrix};
use vulkano::command_buffer;
use vulkano_prelude::*;
pub mod brush;
pub mod document_viewport_proxy;
pub mod id;
pub mod render_device;
pub mod stylus_events;
pub mod tess;
pub mod blend;
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
    /// The data managed by the renderer.
    /// For now, in persuit of actually getting a working product one day,
    /// this is a very coarse caching sceme. In the future, perhaps a bit more granular
    /// control can occur, should performance become an issue:
    ///  * Caching images of incrementally older states, reducing work to get to any given state (performant undo)
    ///  * Caching tesselation output
    pub struct RenderData {
        image: Arc<vk::StorageImage>,
    }

    use crate::vk;
    use anyhow::Result as AnyResult;
    use std::sync::Arc;
    use vulkano::{pipeline::graphics::vertex_input::Vertex, pipeline::Pipeline, sync::GpuFuture};
    mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
            #version 460
    
            layout(push_constant) uniform Matrix {
                mat4 mvp;
            } push_matrix;
    
            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec2 uv;
            layout(location = 2) in vec4 color;
    
            layout(location = 0) out vec4 out_color;
            layout(location = 1) out vec2 out_uv;
    
            void main() {
                out_color = color;
                out_uv = uv;
                vec4 position_2d = push_matrix.mvp * vec4(pos, 0.0, 1.0);
                gl_Position = vec4(position_2d.xy, 0.0, 1.0);
            }"
        }
    }
    mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
            #version 460
            layout(set = 0, binding = 0) uniform sampler2D brush_tex;

            layout(location = 0) in vec4 color;
            layout(location = 1) in vec2 uv;
    
            layout(location = 0) out vec4 out_color;
    
            void main() {
                out_color = color * texture(brush_tex, uv);
            }"
        }
    }

    pub struct StrokeLayerRenderer {
        context: Arc<crate::render_device::RenderContext>,
        texture_descriptor: Arc<vk::PersistentDescriptorSet>,

        render_pass: Arc<vk::RenderPass>,
        pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl StrokeLayerRenderer {
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

            let frag = frag::load(context.device().clone())?;
            let vert = vert::load(context.device().clone())?;
            // Unwraps ok here, using GLSL where "main" is the only allowed entry point.
            let frag = frag.entry_point("main").unwrap();
            let vert = vert.entry_point("main").unwrap();

            // Premultiplied blending with blend constants specified at rendertime.
            // constant of [1.0; 4] is normal, constant of [0.0; 4] is eraser. [r,g,b,1.0] can be used to modulate color, but not alpha.
            let mut premul_dyn_constants = vk::ColorBlendState::new(1);
            premul_dyn_constants.blend_constants = vk::StateMode::Dynamic;
            premul_dyn_constants.attachments[0].blend = Some(vk::AttachmentBlend {
                alpha_source: vulkano::pipeline::graphics::color_blend::BlendFactor::ConstantColor,
                color_source: vulkano::pipeline::graphics::color_blend::BlendFactor::ConstantAlpha,
                alpha_destination:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                color_destination:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                alpha_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
                color_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
            });

            let pipeline = vk::GraphicsPipeline::start()
                .fragment_shader(frag, frag::SpecializationConstants::default())
                .vertex_shader(vert, vert::SpecializationConstants::default())
                .vertex_input_state(super::tess::TessellatedStrokeVertex::per_vertex())
                .input_assembly_state(vk::InputAssemblyState::new()) //Triangle list, no prim restart
                .color_blend_state(premul_dyn_constants)
                .rasterization_state(vk::RasterizationState::new()) // No cull
                .viewport_state(vk::ViewportState::viewport_dynamic_scissor_irrelevant())
                .render_pass(render_pass.clone().first_subpass())
                .build(context.device().clone())?;

            let descriptor_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0, image, sampler,
                )],
            )?;

            Ok(Self {
                context,
                pipeline,
                render_pass,
                texture_descriptor: descriptor_set,
            })
        }
        /// Render stroke from scratch - clear or create cached image, tesselate, and render all strokes.
        pub async fn render_all(
            &self,
            stroke_data: &mut super::StrokeLayerData,
        ) -> AnyResult<()> {
            todo!()
        }
        /// Render new strokes into and on top of current cached contents. If no content, initialize to empty then draw.
        pub async fn render_append(
            &self,
            render_data: &mut RenderData,
            strokes: &[super::Stroke],
        ) -> AnyResult<()> {
            todo!()
        }
        /// Render provided tessellated strokes from and into the provided buffer.
        /// Assumes the tessellated stroke infos match the contenst of the buffer.
        /// Some cursory checks are made to ensure this is the case, but don't rely on them.
        pub fn render(
            &self,
            //render_data: &mut super::LayerRenderData,
        ) -> AnyResult<vk::PrimaryAutoCommandBuffer> {
            // Skip if no tessellated verts
            let Some(verts) = render_data.tessellated_stroke_vertices.clone() else {
                anyhow::bail!("No vertices to render.");
            };

            // "Get or try insert with" - make the image if it doesn't already exist.
            let render_view = match render_data.image {
                Some(ref image) => image.clone(),
                None => {
                    let new_image = vk::StorageImage::with_usage(
                        self.context.allocators().memory(),
                        vulkano::image::ImageDimensions::Dim2d {
                            width: crate::DOCUMENT_DIMENSION,
                            height: crate::DOCUMENT_DIMENSION,
                            array_layers: 1,
                        },
                        crate::DOCUMENT_FORMAT,
                        vk::ImageUsage::COLOR_ATTACHMENT,
                        vk::ImageCreateFlags::empty(),
                        [
                            self.context.queues().graphics().idx(),
                            self.context.queues().compute().idx(),
                        ],
                    )?;
                    let view = vk::ImageView::new_default(new_image)?;
                    render_data.image = Some(view.clone());
                    view
                }
            };

            let x =smallvec::smallvec![7,,,,];

            let framebuffer = vk::Framebuffer::new(
                self.render_pass.clone(),
                vk::FramebufferCreateInfo {
                    attachments: vec![render_view],
                    ..Default::default()
                },
            )?;

            let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;

            let push_matrix = cgmath::ortho(
                0.0,
                crate::DOCUMENT_DIMENSION as f32,
                crate::DOCUMENT_DIMENSION as f32,
                0.0,
                -1.0,
                1.0,
            );

            let mut prev_blend_constants = [1.0; 4];

            command_buffer
                .begin_render_pass(
                    vk::RenderPassBeginInfo {
                        clear_values: vec![Some([0.0; 4].into())],
                        ..vk::RenderPassBeginInfo::framebuffer(framebuffer)
                    },
                    vk::SubpassContents::Inline,
                )?
                .bind_pipeline_graphics(self.pipeline.clone())
                .bind_vertex_buffers(0, [verts])
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    vec![self.texture_descriptor.clone()],
                )
                .set_viewport(
                    0,
                    [vk::Viewport {
                        depth_range: 0.0..1.0,
                        dimensions: [crate::DOCUMENT_DIMENSION as f32; 2],
                        origin: [0.0; 2],
                    }],
                )
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    vert::Matrix {
                        mvp: push_matrix.into(),
                    },
                )
                .set_blend_constants(prev_blend_constants);

            for info in render_data.tessellated_stroke_infos.iter() {
                // Draws can further be batched here. For now, just avoid resetting all the time.
                if info.blend_constants != prev_blend_constants {
                    prev_blend_constants = info.blend_constants;
                    command_buffer.set_blend_constants(info.blend_constants);
                }
                command_buffer.draw(info.vertices, 1, info.first_vertex, 0)?;
            }

            command_buffer.end_render_pass()?;
            Ok(command_buffer.build()?)
        }
    }
}

pub struct StrokeBrushSettings {
    brush: brush::WeakBrushID,
    /// `a` is flow, NOT opacity, since the stroke is blended continuously not blended as a group.
    color_modulate: [f32; 4],
    size_mul: f32,
    /// If true, the blend constants must be set to generate an erasing effect.
    is_eraser: bool,
}
#[derive(Clone, Copy)]
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
    strokes: Vec<Stroke>,
    render_data: Option<stroke_renderer::RenderData>,
}
/// Collection of layer data (stroke contents and render data) mapped from ID
pub struct StrokeLayerManager {
    layers: std::collections::HashMap<FuzzID<StrokeLayer>, StrokeLayerData>,
}

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

    let tesselator = tess::rayon::RayonTessellator;
    let stroke_renderer = stroke_renderer::StrokeLayerRenderer::new(renderer.clone()).unwrap();

    let mut was_pressed = false;
    loop {
        match runtime.block_on(event_stream.recv()) {
            Ok(event_frame) => {
                let mut changed = false;

                let matrix = document_preview.read().get_matrix().invert().unwrap();
                for event in event_frame.iter() {
                    if event.pressed {
                        let pos = matrix * cgmath::vec4(event.pos.0, event.pos.1, 0.0, 1.0);
                        let pos = [
                            pos.x * DOCUMENT_DIMENSION as f32,
                            (1.0 - pos.y) * DOCUMENT_DIMENSION as f32,
                        ];


                        changed = true;
                    }

                    was_pressed = event.pressed;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(num)) => {
                log::warn!("Lost {num} stylus frames!");
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
        }
    }
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

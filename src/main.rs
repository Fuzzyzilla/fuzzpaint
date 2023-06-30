#![feature(array_chunks)]

use std::sync::Arc;
mod egui_impl;
pub mod gpu_err;
pub mod vulkano_prelude;
pub mod window;
use cgmath::{Matrix4, SquareMatrix};
use vulkano::command_buffer;
use vulkano_prelude::*;
pub mod document_viewport_proxy;
pub mod render_device;
pub mod stylus_events;

use anyhow::Result as AnyResult;

#[derive(strum::AsRefStr, PartialEq, Eq, strum::EnumIter, Copy, Clone)]
pub enum BlendMode {
    Normal,
    Screen,
    Multiply,
}
impl Default for BlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

pub struct Blend {
    mode: BlendMode,
    opacity: f32,
    alpha_clip: bool,
}
impl Default for Blend {
    fn default() -> Self {
        Self {
            mode: Default::default(),
            opacity: 1.0,
            alpha_clip: false,
        }
    }
}

// Collection of pending IDs by type.
static ID_SERVER: std::sync::OnceLock<
    parking_lot::RwLock<std::collections::HashMap<std::any::TypeId, std::sync::atomic::AtomicU64>>,
> = std::sync::OnceLock::new();

/// ID that is unique within this execution of the program.
/// IDs with different types may share a value but should not be considered equal.
pub struct FuzzID<T: std::any::Any> {
    id: u64,
    // Namespace marker
    _phantom: std::marker::PhantomData<T>,
}

//Derivation of these traits fails, for some reason.
impl<T: std::any::Any> Clone for FuzzID<T> {
    fn clone(&self) -> Self {
        FuzzID {
            id: self.id,
            _phantom: Default::default(),
        }
    }
}
impl<T: std::any::Any> Copy for FuzzID<T> {}
impl<T: std::any::Any> std::cmp::PartialEq for FuzzID<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T: std::any::Any> std::cmp::Eq for FuzzID<T> {}
impl<T: std::any::Any> std::hash::Hash for FuzzID<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.id);
    }
}

impl<T: std::any::Any> FuzzID<T> {
    pub fn id(&self) -> u64 {
        self.id
    }
}
impl<T: std::any::Any> Default for FuzzID<T> {
    fn default() -> Self {
        let map = ID_SERVER.get_or_init(Default::default);
        let id = {
            let read = map.upgradable_read();
            let ty = std::any::TypeId::of::<T>();
            if let Some(atomic) = read.get(&ty) {
                //We don't really care about the order things happen in, it just needs
                //to be unique.
                atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            } else {
                // We need to insert into the map - transition to exclusive access
                let mut write = parking_lot::RwLockUpgradableReadGuard::upgrade(read);
                // Initialize at 1, return ID 0
                write.insert(ty, 1.into());
                0
            }
        };

        Self {
            id,
            _phantom: Default::default(),
        }
    }
}
impl<T: std::any::Any> std::fmt::Display for FuzzID<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //Unwrap here is safe - the rsplit will always return at least one element, even for empty strings.
        write!(
            f,
            "{}#{}",
            std::any::type_name::<T>().rsplit("::").next().unwrap(),
            self.id
        )
    }
}

pub struct GroupLayer {
    name: String,

    /// Some - grouped rendering, None - Passthrough
    blend: Option<Blend>,

    /// ID that is unique within this execution of the program
    id: FuzzID<GroupLayer>,
}
impl Default for GroupLayer {
    fn default() -> Self {
        let id = FuzzID::default();
        Self {
            name: format!("Group {}", id.id().wrapping_add(1)),
            id,
            blend: None,
        }
    }
}
pub struct StrokeLayer {
    name: String,
    blend: Blend,

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
    pub fn id(&self) -> FuzzID<LayerNode> {
        match self {
            Self::Group { id, .. } => *id,
            Self::StrokeLayer { id, .. } => *id,
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
        at: FuzzID<LayerNode>,
    ) -> bool {
        //Find search candidates
        let nodes_to_search = if traverse_stack.is_empty() {
            self.top_level.as_slice()
        } else {
            // Return false if last element is not a group (shouldn't occur)
            let LayerNode::Group{children, ..} = traverse_stack.last().clone().unwrap() else {return false};

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
    fn find<'a>(&'a self, at: FuzzID<LayerNode>) -> Option<Vec<&'a LayerNode>> {
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
    fn insert_node_at(&mut self, at: FuzzID<LayerNode>, node: LayerNode) {
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
                            let Some(LayerNode::Group { children: siblings, .. }) = path.get(path.len() - 2)
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
                        let Some((idx, _)) = siblings.iter().enumerate().find(|(_, node)| node.id() == at)
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
    pub fn insert_layer(&mut self, layer: impl Into<LayerNode>) -> FuzzID<LayerNode> {
        let node = layer.into();
        let node_id = node.id();
        self.insert_node(node);
        node_id
    }
    /// Insert the group at the given position
    /// If the position is a group, insert at the highest position in the group
    /// If the position is a layer, insert above it.
    /// If the position doesn't exist, behaves as `insert_group`.
    pub fn insert_layer_at(
        &mut self,
        at: FuzzID<LayerNode>,
        layer: impl Into<LayerNode>,
    ) -> FuzzID<LayerNode> {
        let node = layer.into();
        let node_id = node.id();
        self.insert_node_at(at, node);
        node_id
    }
    pub fn mut_children_of<'a>(
        &'a mut self,
        parent: FuzzID<LayerNode>,
    ) -> Option<&'a mut [LayerNode]> {
        let path = self.find(parent)?;

        // Get children, or return none if not found or not a group
        let Some(LayerNode::Group { children, ..}) = path.last()
            else {return None};

        unsafe {
            // Safety - the return value continues to mutably borrow self,
            // so no other access can occur.
            Some((*children.get()).as_mut_slice())
        }
    }
    /// Remove and return the node of the given ID. None if not found.
    pub fn remove(&mut self, at: FuzzID<LayerNode>) -> Option<LayerNode> {
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
            let LayerNode::Group { children: siblings, .. } = path.get(path.len() - 2)?
                else {return None};

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

#[derive(PartialEq, Eq, Hash, strum::AsRefStr, strum::EnumIter, Copy, Clone)]
pub enum BrushKind {
    Stamped,
    Rolled,
}
pub enum BrushStyle {
    Stamped { spacing: f32 },
    Rolled,
}
impl BrushStyle {
    pub fn default_for(brush_kind: BrushKind) -> Self {
        match brush_kind {
            BrushKind::Stamped => Self::Stamped { spacing: 5.0 },
            BrushKind::Rolled => Self::Rolled,
        }
    }
    pub fn brush_kind(&self) -> BrushKind {
        match self {
            Self::Stamped { .. } => BrushKind::Stamped,
            Self::Rolled => BrushKind::Rolled,
        }
    }
}
impl Default for BrushStyle {
    fn default() -> Self {
        Self::default_for(BrushKind::Stamped)
    }
}
pub struct Brush {
    name: String,

    style: BrushStyle,

    id: FuzzID<Brush>,

    //Globally unique ID, for allowing files to be shared after serialization
    universal_id: uuid::Uuid,
}
impl Default for Brush {
    fn default() -> Self {
        let id = FuzzID::default();
        Self {
            name: format!("Brush {}", id.id()),
            style: Default::default(),
            id,
            universal_id: uuid::Uuid::new_v4(),
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
            name: format!("New Document {}", id.id().wrapping_add(1)),
            id,
        }
    }
}
struct PerDocumentInterface {
    zoom: f32,
    rotate: f32,

    focused_subtree: Option<FuzzID<LayerNode>>,
    cur_layer: Option<FuzzID<LayerNode>>,
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
    cur_brush: Option<FuzzID<Brush>>,
    brushes: Vec<Brush>,

    cur_document: Option<FuzzID<Document>>,
    documents: Vec<Document>,

    document_interfaces: std::collections::HashMap<FuzzID<Document>, PerDocumentInterface>,
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
        }
    }
}

impl DocumentUserInterface {
    fn target_layer(&self) -> Option<FuzzID<LayerNode>> {
        let id = self.cur_document?;
        // Selected layer of the currently focused document, if any
        self.document_interfaces.get(&id)?.cur_layer
    }
    fn ui_layer_blend(ui: &mut egui::Ui, id: impl std::hash::Hash, blend: &mut Blend) {
        ui.horizontal(|ui| {
            //Checkerboard icon, for alpha clipping
            ui.toggle_value(&mut blend.alpha_clip, "▓")
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
                    for blend_mode in <crate::BlendMode as strum::IntoEnumIterator>::iter() {
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
                            ui.selectable_value(&mut document_interface.cur_layer, Some(*id), "🗀");
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

                            Self::ui_layer_blend(ui, id, blend);
                        } else {
                            layer.blend = None;
                        }

                        //Safety - No concurrent access to these nodes.
                        //TODO: iter api so as to not expose unsafe innards.
                        let children = unsafe { &mut *children.get() };
                        egui::CollapsingHeader::new("Children")
                            .default_open(true)
                            .enabled(!children.is_empty())
                            .show_unindented(ui, |ui| {
                                Self::ui_layer_slice(ui, document_interface, children);
                            });
                    }
                    LayerNode::StrokeLayer { layer, id } => {
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut document_interface.cur_layer, Some(*id), "✏");
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
                    document_interface.focused_subtree = Some(layer.id());
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
                                .insert(document.id, Default::default());
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
                let Some(document) = self.cur_document else {return};

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
            });
        });

        egui::SidePanel::right("Layers").show(&ctx, |ui| {
            ui.label("Layers");
            ui.separator();

            // if there is no current document, there is nothing for us to do here
            let Some(document_id) = self.cur_document else {return};

            // Find the document, otherwise clear selection
            let Some(document) = self.documents.iter_mut().find(|doc| doc.id == document_id)
                else {
                    self.cur_document = None;
                    return
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
                    .and_then(|tree| document.layers.mut_children_of(tree))
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
                    let brush = Brush::default();
                    self.brushes.push(brush);
                }
                if ui.button("✖").on_hover_text("Delete brush").clicked() {
                    if let Some(id) = self.cur_brush.take() {
                        self.brushes.retain(|brush| brush.id != id);
                    }
                };
            });
            ui.separator();

            for brush in self.brushes.iter_mut() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.cur_brush, Some(brush.id), "");
                        ui.text_edit_singleline(&mut brush.name);
                    })
                    .response
                    .on_hover_ui(|ui| {
                        ui.label(format!("{}", brush.id));

                        //Smol optimization to avoid formatters
                        let mut buf = uuid::Uuid::encode_buffer();
                        let uuid = brush.universal_id.as_hyphenated().encode_upper(&mut buf);
                        ui.label(&uuid[..]);
                    });
                    egui::CollapsingHeader::new("Settings")
                        .id_source(brush.id)
                        .default_open(true)
                        .show(ui, |ui| {
                            let mut brush_kind = brush.style.brush_kind();

                            egui::ComboBox::new(brush.id, "")
                                .selected_text(brush_kind.as_ref())
                                .show_ui(ui, |ui| {
                                    for kind in <BrushKind as strum::IntoEnumIterator>::iter() {
                                        ui.selectable_value(&mut brush_kind, kind, kind.as_ref());
                                    }
                                });

                            //Changed by user, switch to defaults for the new kind
                            if brush_kind != brush.style.brush_kind() {
                                brush.style = BrushStyle::default_for(brush_kind);
                            }

                            match &mut brush.style {
                                BrushStyle::Stamped { spacing } => {
                                    let slider = egui::widgets::Slider::new(spacing, 0.1..=200.0)
                                        .clamp_to_range(true)
                                        .logarithmic(true)
                                        .suffix("px");

                                    ui.add(slider);
                                }
                                BrushStyle::Rolled => {}
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
                            .multiply_with_opacity(if self.cur_document == Some(document.id) {
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
                                    Some(document.id),
                                    &document.name,
                                );
                                if ui.small_button("✖").clicked() {
                                    deleted_ids.push(document.id);
                                    //Disselect if deleted.
                                    if self.cur_document == Some(document.id) {
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
                        .retain(|document| !deleted_ids.contains(&document.id));
                    for id in deleted_ids.into_iter() {
                        self.document_interfaces.remove(&id);
                    }
                });
            });
        });
    }
}

#[derive(vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct StrokePointUnpacked {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32_SFLOAT)]
    pressure: f32,
}

mod test_renderer_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
        #version 460

        layout(push_constant) uniform Matrix {
            mat4 mvp;
        } push_matrix;

        layout(location = 0) in vec2 pos;
        layout(location = 1) in float pressure;

        layout(location = 0) flat out vec4 color;

        void main() {
            vec4 position_2d = push_matrix.mvp * vec4(pos, 0.0, 1.0);
            color = vec4(0.0, 0.0, 0.0, pressure);
            gl_Position = vec4(position_2d.xy, 0.0, 1.0);
        }"
    }
}
mod test_renderer_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
        #version 460
        layout(location = 0) flat in vec4 color;

        layout(location = 0) out vec4 out_color;

        void main() {
            out_color = color;
        }"
    }
}

const DOCUMENT_DIMENSION: u32 = 512;

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

struct SillyDocument {
    verts: Vec<StrokePointUnpacked>,
    indices: Vec<u32>,
}
struct SillyDocumentRenderer {
    render_context: Arc<render_device::RenderContext>,
    pipeline: Arc<vk::GraphicsPipeline>,
    render_pass: Arc<vk::RenderPass>,

    ms_attachment_view: Arc<vk::ImageView<vk::AttachmentImage>>,
}
impl SillyDocumentRenderer {
    fn new(render_context: Arc<render_device::RenderContext>) -> AnyResult<Self> {
        let document_format = vk::Format::R16G16B16A16_SFLOAT;
        let document_dimension = DOCUMENT_DIMENSION;
        let ms_attachment = vk::AttachmentImage::transient_multisampled(
            render_context.allocators().memory(),
            [DOCUMENT_DIMENSION; 2],
            vk::SampleCount::Sample8,
            document_format,
        )?;
        let ms_attachment_view = vk::ImageView::new_default(ms_attachment)?;

        let render_pass = vulkano::single_pass_renderpass!(
            render_context.device().clone(),
            attachments: {
                document: {
                    load: Clear,
                    store: DontCare,
                    format: document_format,
                    samples: 8,
                },
                resolve: {
                    load: DontCare,
                    store: Store,
                    format: document_format,
                    samples: 1,
                },
            },
            pass: {
                color: [document],
                depth_stencil: {},
                resolve: [resolve],
            },
        )?;

        let vert = test_renderer_vert::load(render_context.device().clone())?;
        let frag = test_renderer_frag::load(render_context.device().clone())?;

        let mut blend_premul = vk::ColorBlendState::new(1);
        blend_premul.attachments[0].blend = Some(vk::AttachmentBlend {
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
            .render_pass(render_pass.clone().first_subpass())
            .vertex_shader(
                vert.entry_point("main").unwrap(),
                test_renderer_vert::SpecializationConstants::default(),
            )
            .fragment_shader(
                frag.entry_point("main").unwrap(),
                test_renderer_frag::SpecializationConstants::default(),
            )
            .vertex_input_state(StrokePointUnpacked::per_vertex())
            .input_assembly_state(vk::InputAssemblyState {
                topology: vk::PartialStateMode::Fixed(vk::PrimitiveTopology::LineStrip),
                primitive_restart_enable: vk::StateMode::Fixed(true),
                ..Default::default()
            })
            .color_blend_state(blend_premul)
            .multisample_state(vk::MultisampleState {
                alpha_to_coverage_enable: false,
                alpha_to_one_enable: false,
                rasterization_samples: vk::SampleCount::Sample8,
                sample_shading: None,
                ..Default::default()
            })
            .rasterization_state(vk::RasterizationState {
                line_width: vk::StateMode::Fixed(4.0),
                line_rasterization_mode:
                    vulkano::pipeline::graphics::rasterization::LineRasterizationMode::Rectangular,
                ..vk::RasterizationState::default()
            })
            .viewport_state(vk::ViewportState::viewport_fixed_scissor_irrelevant([
                vk::Viewport {
                    depth_range: 0.0..1.0,
                    dimensions: [document_dimension as f32, document_dimension as f32],
                    origin: [0.0; 2],
                },
            ]))
            .build(render_context.device().clone())?;

        Ok(Self {
            render_context,
            pipeline,
            render_pass,

            ms_attachment_view,
        })
    }
    fn draw(
        &self,
        doc: &SillyDocument,
        buff: Arc<vk::ImageView<vk::StorageImage>>,
    ) -> AnyResult<vk::sync::future::FenceSignalFuture<impl vk::sync::GpuFuture>> {
        let matrix = cgmath::ortho(
            0.0,
            DOCUMENT_DIMENSION as f32,
            DOCUMENT_DIMENSION as f32,
            0.0,
            -1.0,
            0.0,
        );

        let points_buf = vk::Buffer::from_iter(
            self.render_context.allocators().memory(),
            vulkano::buffer::BufferCreateInfo {
                usage: vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                usage: vk::MemoryUsage::Upload,
                ..Default::default()
            },
            doc.verts.iter().copied(),
        )?;
        let indices = vk::Buffer::from_iter(
            self.render_context.allocators().memory(),
            vulkano::buffer::BufferCreateInfo {
                usage: vk::BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                usage: vk::MemoryUsage::Upload,
                ..Default::default()
            },
            doc.indices.iter().copied(),
        )?;

        let document_framebuffer = vk::Framebuffer::new(
            self.render_pass.clone(),
            vk::FramebufferCreateInfo {
                attachments: vec![self.ms_attachment_view.clone(), buff.clone()],
                ..Default::default()
            },
        )?;

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.render_context.allocators().command_buffer(),
            self.render_context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer
            .begin_render_pass(
                vk::RenderPassBeginInfo {
                    clear_values: vec![Some(vk::ClearValue::Float([0.0; 4])), None],
                    ..vk::RenderPassBeginInfo::framebuffer(document_framebuffer)
                },
                vk::SubpassContents::Inline,
            )?
            .bind_pipeline_graphics(self.pipeline.clone())
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                test_renderer_vert::Matrix { mvp: matrix.into() },
            )
            .bind_vertex_buffers(0, [points_buf])
            .bind_index_buffer(indices)
            .draw_indexed(doc.indices.len() as u32, 1, 0, 0, 0)?
            .end_render_pass()?;
        let command_buffer = command_buffer.build()?;

        Ok(self
            .render_context
            .now()
            .then_execute(
                self.render_context.queues().graphics().queue().clone(),
                command_buffer,
            )?
            .then_signal_fence_and_flush()?)
    }
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

    let mut doc = SillyDocument {
        indices: vec![],
        verts: vec![],
    };

    let renderer = SillyDocumentRenderer::new(renderer.clone()).unwrap();

    let mut was_pressed = false;
    loop {
        match runtime.block_on(event_stream.recv()) {
            Ok(event_frame) => {
                let mut changed = false;

                let matrix = document_preview.read().get_matrix().invert().unwrap();
                for event in event_frame.iter() {
                    // released, append a primitive restart command
                    if was_pressed && !event.pressed {
                        doc.indices.push(u32::MAX);
                    }
                    if event.pressed {
                        let pos = matrix * cgmath::vec4(event.pos.0, event.pos.1, 0.0, 1.0);

                        doc.verts.push(StrokePointUnpacked {
                            pos: [
                                pos.x * DOCUMENT_DIMENSION as f32,
                                (1.0 - pos.y) * DOCUMENT_DIMENSION as f32,
                            ],
                            pressure: event.pressure.unwrap_or(0.0),
                        });
                        doc.indices.push((doc.verts.len() - 1) as u32);
                        changed = true;
                    }

                    was_pressed = event.pressed;
                }

                if changed {
                    let buff = runtime.block_on(document_preview.read().get_writeable_buffer());
                    renderer.draw(&doc, buff).unwrap().wait(None).unwrap();

                    document_preview.read().swap();
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

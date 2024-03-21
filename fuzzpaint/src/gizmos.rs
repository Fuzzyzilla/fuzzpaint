//! # Gizmos
//!
//! Represents an interactive overlay atop the document editing workspace but behind the UI workspace. Useful for interactive
//! elements, mouse hit detection, rendering immediate previews, ect. without re-rendering the document.
//!
//! More complex gizmos are made up of simple composable elements.
//
// (Todo: Should crate::document_viewport_proxy be a kind of gizmo? the parallels are clear...)

pub mod renderer;
pub mod transform;
use transform::GizmoTransform;

pub use winit::window::CursorIcon;

pub enum MeshMode {
    Triangles,
    WideLineStrip(std::sync::Arc<[renderer::WideLineVertex]>),
    Shape(RenderShape),
    None,
}
pub enum TextureMode {
    /// Simple solid color
    Solid([u8; 4]),
    /// Use a texture. Will only ever be used for read operations,
    /// but note that there is no current method to query when this resource
    /// is done being used!
    Texture {
        view: std::sync::Arc<crate::vk::ImageView>,
        modulate: [u8; 4],
    },
    /// Screenspace ant-trail effect.
    AntTrail,
}
impl TextureMode {
    #[must_use]
    pub fn white() -> Self {
        Self::Solid([255; 4])
    }
    #[must_use]
    pub fn transparent() -> Self {
        Self::Solid([0; 4])
    }
}
#[derive(Copy, Clone)]
pub enum RenderShape {
    Rectangle {
        position: ultraviolet::Vec2,
        size: ultraviolet::Vec2,
        rotation: f32,
    },
    Ellipse {
        origin: ultraviolet::Vec2,
        radii: ultraviolet::Vec2,
        rotation: f32,
    },
}

pub struct Visual {
    pub mesh: MeshMode,
    pub texture: TextureMode,
}
impl Visual {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            mesh: MeshMode::None,
            texture: TextureMode::transparent(),
        }
    }
}

/// How can a gizmo be interacted with by the mouse?
pub enum GizmoInteraction {
    None,
    /// Can be dragged, and arbitrarily constrained.
    Move,
    /// Can be clicked to open
    Open,
    /// Both `Move`-able and `Open`-able.
    MoveOpen,
    /// Can be rotated around its origin by dragging, can be arbitrarily constrained.
    Rotate,
}

/// The shape of a gizmo's hit window.
/// In local coordinates, determined by `GizmoTransformPinning`
pub enum GizmoShape {
    /// Hollow ring - can be used for circles when inner=0
    Ring {
        inner: f32,
        outer: f32,
    },
    Rectangle {
        min: [f32; 2],
        max: [f32; 2],
    },
    None,
}
impl GizmoShape {
    #[must_use]
    pub fn hit(&self, local: [f32; 2]) -> bool {
        match self {
            Self::None => false,
            Self::Rectangle {
                min: [x0, y0],
                max: [x1, y1],
            } => (local[0] > *x0 && local[0] < *x1) && (local[1] > *y0 && local[1] < *y1),
            Self::Ring { inner, outer } => {
                let dist_sq = local[0] * local[0] + local[1] * local[1];

                dist_sq > inner * inner && dist_sq < outer * outer
            }
        }
    }
}

use std::ops::ControlFlow;

/// A kind of inverse iterator, where the visitor will be passed down the whole
/// tree to visit every gizmo in order.
pub trait GizmoVisitor<T> {
    /// Visit a [Gizmo]. Return Some to short circuit, None to continue.
    fn visit_gizmo(&mut self, gizmo: &Gizmo) -> ControlFlow<T>;
    /// Visit a [Collection]. Return Some to short circuit, None to continue.
    fn visit_collection(&mut self, gizmo: &Collection) -> ControlFlow<T>;
    /// The most recent [Collection] has been fully visited. Return Some to short circuit, None to continue.
    fn end_collection(&mut self, gizmo: &Collection) -> ControlFlow<T>;
}

/// [`GizmoVisitor`] except it accesses the Gizmos as mutable references.
pub trait MutableGizmoVisitor<T> {
    /// Visit a [Gizmo]. Return Some to short circuit, None to continue.
    fn visit_gizmo_mut(&mut self, gizmo: &mut Gizmo) -> ControlFlow<T>;
    /// Visit a [Collection]. Return Some to short circuit, None to continue.
    fn visit_collection_mut(&mut self, gizmo: &mut Collection) -> ControlFlow<T>;
    /// The most recent [Collection] has been fully visited. Return Some to short circuit, None to continue.
    fn end_collection_mut(&mut self, gizmo: &mut Collection) -> ControlFlow<T>;
}

pub struct Gizmo {
    pub visual: Visual,

    pub interaction: GizmoInteraction,
    pub hit_shape: GizmoShape,

    pub hover_cursor: CursorOrInvisible,
    pub grab_cursor: CursorOrInvisible,

    pub transform: GizmoTransform,
}
impl Default for Gizmo {
    fn default() -> Self {
        Self {
            visual: Visual::empty(),
            hit_shape: GizmoShape::None,
            grab_cursor: CursorOrInvisible::default(),
            hover_cursor: CursorOrInvisible::default(),
            interaction: GizmoInteraction::None,
            transform: transform::GizmoTransform::inherit_all(),
        }
    }
}

/// A collection of many gizmos. It itself is a Gizmo,
/// meaning Collections-in-Collections is supported.
pub struct Collection {
    pub transform: GizmoTransform,
    /// Children of this gizmo, sorted top to bottom.
    children: Vec<AnyGizmo>,
}
impl Collection {
    #[must_use]
    pub fn new(transform: GizmoTransform) -> Self {
        Self {
            transform,
            children: Vec::new(),
        }
    }
    pub fn push_top(&mut self, other: impl Into<AnyGizmo>) {
        self.children.insert(0, other.into());
    }
    pub fn push_bottom(&mut self, other: impl Into<AnyGizmo>) {
        self.children.push(other.into());
    }
}

// mem inefficient, implementation detail uwu
/// must be public for `Into<AnyGizmo>` in Collections interface but don't use >:(
pub enum AnyGizmo {
    Gizmo(Gizmo),
    Collection(Collection),
}
impl From<Gizmo> for AnyGizmo {
    fn from(value: Gizmo) -> Self {
        Self::Gizmo(value)
    }
}
impl From<Collection> for AnyGizmo {
    fn from(value: Collection) -> Self {
        Self::Collection(value)
    }
}

/// None to hide the cursor, or Some to choose a winit cursor.
#[derive(Copy, Clone)]
pub enum CursorOrInvisible {
    Icon(CursorIcon),
    Invisible,
}
impl Default for CursorOrInvisible {
    fn default() -> Self {
        Self::Icon(CursorIcon::Default)
    }
}

/// A tree that can be visited (in several modes) by a [`GizmoVisitor`].
pub trait GizmoTree {
    /// Pass the visitor to self and all children!
    /// Should visit in painters order, back-to-front.
    /// Returns Some with the short circuit value, or None if never short circuited.
    fn visit_painter<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T>;
    /// Pass the visitor to self and all children!
    /// Should visit in hit order, front-to-back.
    /// Returns Some with the short circuit value, or None if never short circuited.
    fn visit_hit<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T>;
}
pub trait MutGizmoTree {
    /// Pass the visitor to self and all children!
    /// Should visit in painters order, back-to-front.
    /// Returns Some with the short circuit value, or None if never short circuited.
    fn visit_painter_mut<T>(&mut self, visitor: &mut impl MutableGizmoVisitor<T>)
        -> ControlFlow<T>;
    /// Pass the visitor to self and all children!
    /// Should visit in hit order, front-to-back.
    /// Returns Some with the short circuit value, or None if never short circuited.
    fn visit_hit_mut<T>(&mut self, visitor: &mut impl MutableGizmoVisitor<T>) -> ControlFlow<T>;
}

impl GizmoTree for Gizmo {
    fn visit_painter<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T> {
        visitor.visit_gizmo(self)
    }

    fn visit_hit<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T> {
        visitor.visit_gizmo(self)
    }
}
impl MutGizmoTree for Gizmo {
    fn visit_painter_mut<T>(
        &mut self,
        visitor: &mut impl MutableGizmoVisitor<T>,
    ) -> ControlFlow<T> {
        visitor.visit_gizmo_mut(self)
    }

    fn visit_hit_mut<T>(&mut self, visitor: &mut impl MutableGizmoVisitor<T>) -> ControlFlow<T> {
        visitor.visit_gizmo_mut(self)
    }
}

impl GizmoTree for Collection {
    fn visit_painter<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T> {
        visitor.visit_collection(self)?;

        // In painters order- reverse the children
        for child in self.children.iter().rev() {
            child.visit_painter(visitor)?;
        }

        visitor.end_collection(self)
    }

    fn visit_hit<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T> {
        visitor.visit_collection(self)?;

        // In hit order- don't reverse the children
        for child in &self.children {
            child.visit_hit(visitor)?;
        }

        visitor.end_collection(self)
    }
}
impl MutGizmoTree for Collection {
    fn visit_painter_mut<T>(
        &mut self,
        visitor: &mut impl MutableGizmoVisitor<T>,
    ) -> ControlFlow<T> {
        visitor.visit_collection_mut(self)?;

        // In painters order- reverse the children
        for child in self.children.iter_mut().rev() {
            child.visit_painter_mut(visitor)?;
        }

        visitor.end_collection_mut(self)
    }

    fn visit_hit_mut<T>(&mut self, visitor: &mut impl MutableGizmoVisitor<T>) -> ControlFlow<T> {
        visitor.visit_collection_mut(self)?;

        // In hit order- don't reverse the children
        for child in &mut self.children {
            child.visit_hit_mut(visitor)?;
        }

        visitor.end_collection_mut(self)
    }
}

impl GizmoTree for AnyGizmo {
    fn visit_painter<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T> {
        match self {
            AnyGizmo::Collection(g) => g.visit_painter(visitor),
            AnyGizmo::Gizmo(g) => g.visit_painter(visitor),
        }
    }

    fn visit_hit<T>(&self, visitor: &mut impl GizmoVisitor<T>) -> ControlFlow<T> {
        match self {
            AnyGizmo::Collection(g) => g.visit_hit(visitor),
            AnyGizmo::Gizmo(g) => g.visit_hit(visitor),
        }
    }
}
impl MutGizmoTree for AnyGizmo {
    fn visit_painter_mut<T>(
        &mut self,
        visitor: &mut impl MutableGizmoVisitor<T>,
    ) -> ControlFlow<T> {
        match self {
            AnyGizmo::Collection(g) => g.visit_painter_mut(visitor),
            AnyGizmo::Gizmo(g) => g.visit_painter_mut(visitor),
        }
    }

    fn visit_hit_mut<T>(&mut self, visitor: &mut impl MutableGizmoVisitor<T>) -> ControlFlow<T> {
        match self {
            AnyGizmo::Collection(g) => g.visit_hit_mut(visitor),
            AnyGizmo::Gizmo(g) => g.visit_hit_mut(visitor),
        }
    }
}

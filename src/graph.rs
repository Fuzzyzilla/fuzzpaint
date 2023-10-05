pub enum LeafType {
    StrokeLayer {
        blend: crate::Blend,
        source: crate::WeakID<crate::StrokeLayer>,
    },
    SolidColor {
        blend: crate::Blend,
        // Crate-wide color type would be nice :O
        source: [f32; 4],
    },
    // The name of the note is the note!
    Note,
}
impl LeafType {
    pub fn blend_of(&self) -> Option<crate::Blend> {
        match self {
            Self::StrokeLayer { blend, .. } => Some(*blend),
            Self::SolidColor { blend, .. } => Some(*blend),
            Self::Note => None,
        }
    }
}
pub enum NodeType {
    /// Leaves are grouped for organization only, and the blend graph
    /// treats it as if it were simply it's children
    Passthrough,
    /// Leaves are rendered as a group, the output is then blended as a single image.
    GroupedBlend(crate::Blend),
}
impl NodeType {
    pub fn blend_of(&self) -> Option<crate::Blend> {
        match self {
            Self::Passthrough => None,
            Self::GroupedBlend(blend) => Some(*blend),
        }
    }
}

// Shhh.. they're secretly the same type >:3c
pub struct LeafID(id_tree::NodeId);
pub struct NodeID(id_tree::NodeId);
pub enum AnyID {
    Leaf(LeafID),
    Node(NodeID),
}
impl From<LeafID> for AnyID {
    fn from(value: LeafID) -> Self {
        Self::Leaf(value)
    }
}
impl From<NodeID> for AnyID {
    fn from(value: NodeID) -> Self {
        Self::Node(value)
    }
}
impl AnyID {
    fn into_raw(self) -> id_tree::NodeId {
        match self {
            AnyID::Leaf(LeafID(id)) => id,
            AnyID::Node(NodeID(id)) => id,
        }
    }
}

enum NodeDataTy {
    Root,
    Node(NodeType),
    Leaf(LeafType),
}
impl NodeDataTy {
    fn is_leaf(&self) -> bool {
        match self {
            Self::Leaf(..) => true,
            _ => false,
        }
    }
    fn is_node(&self) -> bool {
        match self {
            Self::Node(..) => true,
            _ => false,
        }
    }
}
struct NodeData {
    ty: NodeDataTy,
    name: String,
}

#[derive(thiserror::Error, Debug)]
pub enum TargetError {
    #[error("The target ID is not known to this blend graph!")]
    TargetNotFound,
}

#[derive(thiserror::Error, Debug)]
pub enum ReparentError {
    #[error("{}", .0)]
    TargetError(TargetError),
    #[error("The destination ID is not known to this blend graph.")]
    DestinationNotFound,
    #[error("Cannot reparent to one of the node's [grand]children!")]
    WouldCycle,
}

pub enum Location {
    /// Calculate the index and parent, such that the location
    /// referenced is the sibling above this node.
    AboveSelection(AnyID),
    /// Set as the nth child of this node, where top = 0
    IndexIntoNode(NodeID, usize),
    /// Set as the nth child of the root, where top = 0
    IndexIntoRoot(usize),
}

pub struct BlendGraph {
    tree: id_tree::Tree<NodeData>,
}
impl BlendGraph {
    pub fn add_node(
        &mut self,
        location: Location,
        node_ty: NodeType,
    ) -> Result<NodeID, TargetError> {
        todo!();
    }
    pub fn add_leaf(
        &mut self,
        location: Location,
        leaf_ty: LeafType,
    ) -> Result<NodeID, TargetError> {
        todo!();
    }
    /// Reparent the target onto a new parent.
    /// Children are brought along for the ride!
    pub fn reparent(
        &mut self,
        target: impl Into<AnyID>,
        destination: Location,
    ) -> Result<(), ReparentError> {
        todo!();
    }
    /// Get the blend of the given node, or None if no blend is assigned
    /// (for example on passthrough nodes or Note leaves)
    pub fn blend_of(&self, target: impl Into<AnyID>) -> Result<Option<crate::Blend>, TargetError> {
        todo!()
    }
}

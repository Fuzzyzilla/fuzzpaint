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
    pub fn blend(&self) -> Option<crate::Blend> {
        match self {
            Self::StrokeLayer { blend, .. } => Some(*blend),
            Self::SolidColor { blend, .. } => Some(*blend),
            Self::Note => None,
        }
    }
    pub fn blend_mut(&mut self) -> Option<&mut crate::Blend> {
        match self {
            Self::StrokeLayer { blend, .. } => Some(blend),
            Self::SolidColor { blend, .. } => Some(blend),
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
    pub fn blend(&self) -> Option<crate::Blend> {
        match self {
            Self::Passthrough => None,
            Self::GroupedBlend(blend) => Some(*blend),
        }
    }
    pub fn blend_mut(&mut self) -> Option<&mut crate::Blend> {
        match self {
            Self::Passthrough => None,
            Self::GroupedBlend(blend) => Some(blend),
        }
    }
}

// Shhh.. they're secretly the same type >:3c
#[derive(Debug, Clone)]
pub struct LeafID(id_tree::NodeId);
#[derive(Debug, Clone)]
pub struct NodeID(id_tree::NodeId);
#[derive(Debug, Clone)]
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
    pub fn blend(&self) -> Option<crate::Blend> {
        match self {
            Self::Node(n) => n.blend(),
            Self::Leaf(l) => l.blend(),
            Self::Root => None,
        }
    }
    pub fn blend_mut(&mut self) -> Option<&mut crate::Blend> {
        match self {
            Self::Node(n) => n.blend_mut(),
            Self::Leaf(l) => l.blend_mut(),
            Self::Root => None,
        }
    }
}
pub struct NodeData {
    // NOT public, as we users could break the tree by accessing this!
    ty: NodeDataTy,
    pub name: String,
}
impl NodeData {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }
    pub fn is_leaf(&self) -> bool {
        self.ty.is_leaf()
    }
    pub fn is_node(&self) -> bool {
        self.ty.is_node()
    }
    pub fn leaf(&self) -> Option<&LeafType> {
        if let NodeDataTy::Leaf(l) = &self.ty {
            Some(l)
        } else {
            None
        }
    }
    pub fn leaf_mut(&mut self) -> Option<&mut LeafType> {
        if let NodeDataTy::Leaf(l) = &mut self.ty {
            Some(l)
        } else {
            None
        }
    }
    pub fn node(&self) -> Option<&NodeType> {
        if let NodeDataTy::Node(n) = &self.ty {
            Some(n)
        } else {
            None
        }
    }
    pub fn node_mut(&mut self) -> Option<&mut NodeType> {
        if let NodeDataTy::Node(n) = &mut self.ty {
            Some(n)
        } else {
            None
        }
    }
    pub fn blend(&self) -> Option<crate::Blend> {
        self.ty.blend()
    }
    pub fn blend_mut(&mut self) -> Option<&mut crate::Blend> {
        self.ty.blend_mut()
    }
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
    pub fn new() -> Self {
        Self {
            tree: id_tree::TreeBuilder::new()
                .with_root(id_tree::Node::new(NodeData {
                    name: String::new(),
                    ty: NodeDataTy::Root,
                }))
                .build(),
        }
    }
    /// Iterate the children of the root node
    pub fn iter_top_level(&'_ self) -> impl Iterator<Item = (AnyID, &'_ NodeData)> + '_ {
        self.iter_children_of_raw(self.tree.root_node_id().unwrap())
            .unwrap()
    }
    /// Iterate the children of this node
    pub fn iter_node(
        &'_ self,
        node: &'_ NodeID,
    ) -> Option<impl Iterator<Item = (AnyID, &'_ NodeData)> + '_> {
        self.iter_children_of_raw(&node.0)
    }
    /// Iterate the children of this raw ID. A helper method for all various iters!
    fn iter_children_of_raw(
        &'_ self,
        node_id: &id_tree::NodeId,
    ) -> Option<impl Iterator<Item = (AnyID, &'_ NodeData)> + '_> {
        Some(self.tree.children_ids(node_id).ok()?.map(|node_id| {
            let node = self.tree.get(node_id).unwrap().data();
            let id = match node.ty {
                NodeDataTy::Leaf(_) => AnyID::Leaf(LeafID(node_id.clone())),
                NodeDataTy::Node(_) => AnyID::Node(NodeID(node_id.clone())),
                // Invalid tree state.
                _ => panic!("Root encountered during iteration!"),
            };
            (id, node)
        }))
    }
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

#[cfg(test)]
mod test {
    #[test]
    fn uwu() {
        let graph = super::BlendGraph::new();
        graph
            .iter_top_level()
            .for_each(|(id, data)| log::trace!("{id:?}"))
    }
}

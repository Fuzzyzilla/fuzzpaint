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
pub enum NodeType {
    /// Leaves are grouped for organization only, and the blend graph
    /// treats it as if it were it's children
    Passthrough,
    /// Leaves are rendered as a group, the output is then blended as a single image.
    GroupedBlend(crate::Blend),
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

pub enum NodeData {
    Root,
    Node(NodeType),
    Leaf(LeafType),
}
impl NodeData {
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

#[derive(thiserror::Error, Debug)]
pub enum AddError {
    #[error("The parent ID is not known to this blend graph!")]
    ParentNotFound,
    #[error("The child ID is not a child of the given parent.")]
    ChildNotFound,
}

#[derive(thiserror::Error, Debug)]
pub enum ReparentError {
    #[error("The target ID is not known to this blend graph!")]
    TargetNotFound,
    #[error("The destination ID is not known to this blend graph.")]
    DestinationNotFound,
    #[error("Cannot reparent to one of the node's [grand]children!")]
    WouldCycle,
}

pub struct BlendGraph {
    tree: id_tree::Tree<NodeData>,
}
impl BlendGraph {
    /// Add a new group of the given type into the parent, or the root if none.
    /// If above_child is provided, the group will be added above the given child. Otherwise, it will be added at the bottom of the parent.
    ///
    /// Returns the new ID, or an error if the parent node or child node is not known to this graph.
    pub fn add_group<ChildID>(
        &mut self,
        parent: Option<NodeID>,
        above_child: Option<ChildID>,
        ty: NodeType,
    ) -> Result<NodeID, AddError>
    where
        ChildID: Into<AnyID>,
    {
        /*
                let mut parent = if let Some(parent) = parent {
                    self.tree
                        .get(&parent.0)
                        .map_err(|_| AddError::ParentNotFound)?
                } else {
                    self.tree.get(self.tree.root_node_id().unwrap()).unwrap()
                };

                // Make sure we didn't get our IDs crossed up!
                assert!(parent.data().is_node());
                let new_node = id_tree::Node::new(NodeData::Node(ty));
                self.tree
                    .insert(new_node, id_tree::InsertBehavior::UnderNode(parent));

                // Find where to add
                let id = if let Some(child_id) = above_child.map(|any| Into::<AnyID>::into(any).into_raw())
                {
                    let Some((child_idx, _)) = parent
                        .as_ref()
                        .children()
                        .enumerate()
                        .find(|(_, child)| child.node_id() == child_id)
                    else {
                        return Err(AddError::ChildNotFound);
                    };
                    // add at bottom, swap it forward by child_idx+1 steps. (There is no other way to do this with this crate afaict lol)
                    let mut child = parent.prepend(NodeData::Node(ty));

                    for _ in 0..child_idx + 1 {
                        child.swap_next_sibling();
                    }

                    child.node_id()
                } else {
                    // Just add at bottom.
                    parent.prepend(NodeData::Node(ty)).node_id()
                };
        Ok(NodeID(id))
        */
        todo!()
    }
    /// A helper for add_group, adding at the bottom. Identical to calling it with None for above_child.
    pub fn add_group_bottom(
        &mut self,
        parent: Option<NodeID>,
        ty: NodeType,
    ) -> Result<NodeID, AddError> {
        self.add_group(parent, None::<NodeID>, ty)
    }
    /// Take the given target, and set it as the last child of the destination.
    /// If destination is None, reparent to root.
    /// Any children of the target are brough along for the ride!
    ///
    /// Because of internal weirdness, this consumes the target ID and returns a new one.
    pub fn reparent_node(
        &mut self,
        target_id: NodeID,
        destination: Option<NodeID>,
    ) -> Result<NodeID, ReparentError> {
        if let Some(NodeID(destination_id)) = destination {
            // Find destination
            /*
            let destination = self
                .tree
                .get(destination_id)
                .ok_or(ReparentError::DestinationNotFound)?;
            // Ensure this won't cause a cycle - target is not an ancestor of dest
            if destination
                .ancestors()
                .any(|node| node.node_id() == target_id.0)
            {
                return Err(ReparentError::WouldCycle);
            }
            // Find target
            let mut target = self
                .tree
                .get_mut(target_id.0)
                .ok_or(ReparentError::TargetNotFound)?;
            // Collect children to node
            let children: Vec<_> = target.as_ref().children().collect();
            // Delete
            target.make_last_sibling();
            let data = target
                .parent()
                .unwrap()
                .remove_last(slab_tree::RemoveBehavior::OrphanChildren)
                .unwrap();
            // Find destination (again)
            let mut destination = self.tree.get_mut(destination_id).unwrap();
            let mut new_node = destination.append(data);
            for child in children.into_iter() {
                // Whoopsie. Can't set parent. Neeeeeeeeeeeeeeeeeed other tree lib
            }
            */
            todo!()
        } else {
            // Reparent to root
            todo!()
        }
    }
}

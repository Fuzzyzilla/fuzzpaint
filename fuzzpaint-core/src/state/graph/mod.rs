//! # Graph
//!
//! The layers of the document form a graph, with leaves on the lowest level (like stroke layers, fill layers, ect)
//! and groups forming upper levels. Leaves are not allowed to have children.

pub mod commands;
mod stable_id;
pub mod writer;

use super::transform;
use crate::blend::Blend;
// Re-export the various public ids
// FuzzNodeID is NOT public!
pub use stable_id::{AnyID, LeafID, NodeID};

#[derive(Clone, PartialEq, Debug)]
pub enum LeafType {
    StrokeLayer {
        blend: Blend,
        collection: crate::state::stroke_collection::StrokeCollectionID,
        /// Transform points before tessellation.
        inner_transform: transform::Similarity,
        /// Tramsform points after tessellation.
        outer_transform: transform::Matrix,
    },
    SolidColor {
        blend: Blend,
        source: crate::color::ColorOrPalette,
    },
    Text {
        blend: Blend,
        // Horrible testing interface, this should be much richer
        // (probably a ref to another document state specifying a WYSIWYG
        // text block with faces and sizes and colors and...)
        text: String,
        // "em" is a physical unit. Currently, we have no means to deal with this fact.
        // Allow the user to specify manually.
        px_per_em: f32,
        outer_transform: transform::Matrix,
    },
    // The name of the note is the note!
    Note,
}
impl LeafType {
    #[must_use]
    pub fn blend(&self) -> Option<Blend> {
        match self {
            Self::StrokeLayer { blend, .. }
            | Self::SolidColor { blend, .. }
            | Self::Text { blend, .. } => Some(*blend),
            Self::Note => None,
        }
    }
    pub fn blend_mut(&mut self) -> Option<&mut Blend> {
        match self {
            Self::StrokeLayer { blend, .. }
            | Self::SolidColor { blend, .. }
            | Self::Text { blend, .. } => Some(blend),
            Self::Note => None,
        }
    }
}
#[derive(Clone, PartialEq, Debug)]
pub enum NodeType {
    /// Leaves are grouped for organization only, and the blend graph
    /// treats it as if it were simply it's children
    Passthrough,
    /// Leaves are rendered as a group, the output is then blended as a single image.
    GroupedBlend(Blend),
}
impl NodeType {
    #[must_use]
    pub fn blend(&self) -> Option<Blend> {
        match self {
            Self::Passthrough => None,
            Self::GroupedBlend(blend) => Some(*blend),
        }
    }
    #[must_use]
    pub fn blend_mut(&mut self) -> Option<&mut Blend> {
        match self {
            Self::Passthrough => None,
            Self::GroupedBlend(blend) => Some(blend),
        }
    }
}

#[derive(Clone, PartialEq)]
enum NodeDataTy {
    Root,
    Node(NodeType),
    Leaf(LeafType),
}
impl NodeDataTy {
    fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(..))
    }
    fn is_node(&self) -> bool {
        matches!(self, Self::Node(..))
    }
    pub fn blend(&self) -> Option<Blend> {
        match self {
            Self::Node(n) => n.blend(),
            Self::Leaf(l) => l.blend(),
            Self::Root => None,
        }
    }
    pub fn blend_mut(&mut self) -> Option<&mut Blend> {
        match self {
            Self::Node(n) => n.blend_mut(),
            Self::Leaf(l) => l.blend_mut(),
            Self::Root => None,
        }
    }
}
#[derive(Clone)]
pub struct NodeData {
    // NOT public, as we users could break the tree by mutating this!
    ty: NodeDataTy,
    // NOT public, the user could break the command queue by mutating this!
    /// Represents whether the command that created this node has been undone.
    deleted: bool,
    pub name: String,
}
impl NodeData {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.ty.is_leaf()
    }
    #[must_use]
    pub fn is_node(&self) -> bool {
        self.ty.is_node()
    }
    /// Take ownership of the node, converting into a leaf. Returns ownership of
    /// self on failure
    pub fn into_leaf(self) -> Result<LeafType, Self> {
        match self.ty {
            NodeDataTy::Leaf(l) => Ok(l),
            _ => Err(self),
        }
    }
    #[must_use]
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
    /// Take ownership of the node, converting into a node. Returns ownership of
    /// self on failure
    pub fn into_node(self) -> Result<NodeType, Self> {
        match self.ty {
            NodeDataTy::Node(n) => Ok(n),
            _ => Err(self),
        }
    }
    #[must_use]
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
    #[must_use]
    pub fn blend(&self) -> Option<Blend> {
        self.ty.blend()
    }
    pub fn blend_mut(&mut self) -> Option<&mut Blend> {
        self.ty.blend_mut()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum TargetError {
    #[error("ID not found")]
    TargetNotFound,
    #[error("target ID is deleted")]
    TargetDeleted,
}

#[derive(thiserror::Error, Debug)]
pub enum ReparentError {
    #[error("target not found: {}", .0)]
    TargetError(TargetError),
    #[error("destination not found: {}", .0)]
    DestinationError(TargetError),
    #[error("can't reparent to the node's own [grand]children")]
    WouldCycle,
}

#[derive(Copy, Clone)]
pub enum Location<'a> {
    /// Calculate the index and parent, such that the location
    /// referenced is the sibling above this node.
    AboveSelection(&'a AnyID),
    /// Set as the nth child of this node, where top = 0
    ///
    /// An index too large will be clamped to the bottom position.
    IndexIntoNode(&'a NodeID, usize),
    /// Set as the nth child of the root, where top = 0
    ///
    /// An index too large will be clamped to the bottom position.
    IndexIntoRoot(usize),
}

pub struct BlendGraph {
    tree: id_tree::Tree<NodeData>,
    ids: stable_id::StableIDMap,
}
impl Default for BlendGraph {
    fn default() -> Self {
        Self {
            tree: id_tree::TreeBuilder::new()
                .with_root(id_tree::Node::new(NodeData {
                    name: String::new(),
                    ty: NodeDataTy::Root,
                    deleted: false,
                }))
                .build(),
            ids: stable_id::StableIDMap::default(),
        }
    }
}
impl BlendGraph {
    /// Iterate the children of the root node
    pub fn iter_top_level(&'_ self) -> impl Iterator<Item = (AnyID, &'_ NodeData)> + '_ {
        self.iter_children_of_raw(self.tree.root_node_id().unwrap())
            .unwrap()
    }
    /// Iterate the children of this node
    #[must_use]
    pub fn iter_node(&self, node: NodeID) -> Option<impl Iterator<Item = (AnyID, &NodeData)> + '_> {
        self.iter_children_of_raw(self.ids.tree_id_from_node(node)?)
    }
    /// Iterate all nodes, in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = (AnyID, &NodeData)> + '_ {
        self.tree
            .traverse_post_order_ids(self.tree.root_node_id().unwrap())
            .unwrap()
            .filter_map(|node_id| {
                let node = self.tree.get(&node_id).unwrap().data();
                // Skip children marked as deleted
                if node.deleted {
                    None
                } else {
                    // Ignore root
                    if matches!(node.ty, NodeDataTy::Root) {
                        return None;
                    }
                    let fuz_id = self
                        .ids
                        .fuzz_id_from(&node_id)
                        // Stinky! Nothing we can do here (except filter it out?)
                        // This would be a bug, so report it with expect.
                        .expect("Unknown node encountered in iteration");
                    let id = match node.ty {
                        NodeDataTy::Leaf(_) => AnyID::Leaf(LeafID(*fuz_id)),
                        NodeDataTy::Node(_) => AnyID::Node(NodeID(*fuz_id)),
                        // Already handled above
                        NodeDataTy::Root => unreachable!(),
                    };
                    Some((id, node))
                }
            })
    }
    /// Iterate the children of this raw ID. A helper method for all various iters!
    fn iter_children_of_raw<'s>(
        &'s self,
        node_id: &id_tree::NodeId,
    ) -> Option<impl Iterator<Item = (AnyID, &'s NodeData)> + 's> {
        Some(self.tree.children_ids(node_id).ok()?.filter_map(|node_id| {
            let node = self.tree.get(node_id).unwrap().data();
            // Skip children marked as deleted
            if node.deleted {
                None
            } else {
                let fuz_id = self
                    .ids
                    .fuzz_id_from(node_id)
                    // Stinky! Nothing we can do here (except filter it out?)
                    // This would be a bug, so report it with expect.
                    .expect("Unknown node encountered in iteration");
                let id = match node.ty {
                    NodeDataTy::Leaf(_) => AnyID::Leaf(LeafID(*fuz_id)),
                    NodeDataTy::Node(_) => AnyID::Node(NodeID(*fuz_id)),
                    // Invalid tree state.
                    NodeDataTy::Root => panic!("Root encountered during iteration!"),
                };
                Some((id, node))
            }
        }))
    }
    /// Convert a location to a parent and child idx
    /// Ok result implies the node is both present and not deleted.
    fn find_location<'a>(
        &'a self,
        location: Location<'a>,
    ) -> Result<(&'a id_tree::NodeId, usize), TargetError> {
        match location {
            Location::AboveSelection(selection) => {
                let selection_tree_id = self
                    .ids
                    .tree_id_from_any(*selection)
                    .ok_or(TargetError::TargetNotFound)?;
                let node = self
                    .tree
                    .get(selection_tree_id)
                    .map_err(|_| TargetError::TargetNotFound)?;
                if node.data().deleted {
                    return Err(TargetError::TargetDeleted);
                }
                // unwrap ok - node is NOT the root, if it were that would be a structural error!
                let parent = node.parent().unwrap();
                let (idx, _) = self
                    .tree
                    .children_ids(parent)
                    // unwrap ok - parent will be in the tree if the child was! (checked above)
                    .unwrap()
                    .enumerate()
                    .find(|(_, id)| *id == selection_tree_id)
                    // Unwrap ok - the child must be a child of it's parent of course!
                    .unwrap();
                Ok((parent, idx))
            }
            Location::IndexIntoNode(node, idx) => {
                let selection_tree_id = self
                    .ids
                    .tree_id_from_node(*node)
                    .ok_or(TargetError::TargetNotFound)?;

                let node_deleted = self
                    .tree
                    .get(selection_tree_id)
                    .ok()
                    .map(|node| node.data().deleted);

                match node_deleted {
                    None => Err(TargetError::TargetNotFound),
                    Some(true) => Err(TargetError::TargetDeleted),
                    Some(false) => Ok((selection_tree_id, idx)),
                }
            }
            Location::IndexIntoRoot(idx) => Ok((self.tree.root_node_id().unwrap(), idx)),
        }
    }
    pub fn get(&self, id: impl Into<AnyID>) -> Option<&NodeData> {
        let id = Into::<AnyID>::into(id);
        let tree_id = self.ids.tree_id_from_any(id)?;
        self.tree.get(tree_id).ok().map(id_tree::Node::data)
    }
    pub fn get_mut(&mut self, id: impl Into<AnyID>) -> Option<&mut NodeData> {
        let id = Into::<AnyID>::into(id);
        let tree_id = self.ids.tree_id_from_any(id)?;
        self.tree.get_mut(tree_id).ok().map(id_tree::Node::data_mut)
    }
    pub fn get_node_mut(&mut self, id: NodeID) -> Option<&mut NodeType> {
        self.get_mut(id).and_then(NodeData::node_mut)
    }
    pub fn get_leaf_mut(&mut self, id: LeafID) -> Option<&mut LeafType> {
        self.get_mut(id).and_then(NodeData::leaf_mut)
    }
    pub fn add_node(
        &mut self,
        location: Location,
        name: String,
        ty: NodeType,
    ) -> Result<NodeID, TargetError> {
        let node = id_tree::Node::new(NodeData {
            name,
            deleted: false,
            ty: NodeDataTy::Node(ty),
        });
        // Convert this location to a parent ID and a child idx.
        let (parent_id, idx) = self.find_location(location)?;
        let parent_id = parent_id.to_owned();

        // Check for errors again, since IndexIntoNode hasn't been checked yet.
        let new_node = self
            .tree
            .insert(node, id_tree::InsertBehavior::UnderNode(&parent_id))
            .map_err(|_| TargetError::TargetNotFound)?;

        // unwrap ok - we just added it, of course it will be found!
        self.tree.make_nth_sibling(&new_node, idx).unwrap();

        Ok(NodeID(*self.ids.get_or_insert_tree_id(new_node)))
    }
    pub fn add_leaf(
        &mut self,
        location: Location,
        name: String,
        ty: LeafType,
    ) -> Result<LeafID, TargetError> {
        let node = id_tree::Node::new(NodeData {
            name,
            deleted: false,
            ty: NodeDataTy::Leaf(ty),
        });
        // Convert this location to a parent ID and a child idx.
        let (parent_id, idx) = self.find_location(location)?;
        let parent_id = parent_id.to_owned();

        // Check for errors again, since IndexIntoNode hasn't been checked yet.
        let new_node = self
            .tree
            .insert(node, id_tree::InsertBehavior::UnderNode(&parent_id))
            .map_err(|_| TargetError::TargetNotFound)?;

        // unwrap ok - we just added it, of course it will be found!
        self.tree.make_nth_sibling(&new_node, idx).unwrap();

        Ok(LeafID(*self.ids.get_or_insert_tree_id(new_node)))
    }
    // Get the (parent, idx) of the node. Parent is None if root is the parent.
    fn location_of(&self, id: impl Into<AnyID>) -> Option<(Option<NodeID>, usize)> {
        let tree_id = self.ids.tree_id_from_any(id.into())?;
        let node = self.tree.get(tree_id).ok()?;
        // Should never return None - user shouldn't have access to root ID!
        let parent = node.parent().unwrap();
        let child_idx = self
            .tree
            .children_ids(parent)
            .unwrap()
            .position(|child_id| child_id == tree_id)
            .unwrap();

        // Will be none if parent is root.
        let parent_id = self.ids.fuzz_id_from(parent);

        Some((parent_id.map(|id| NodeID(*id)), child_idx))
    }
    /// Reparent the target onto a new parent.
    /// Children are brought along for the ride!
    pub fn reparent(
        &mut self,
        target: impl Into<AnyID>,
        destination: Location,
    ) -> Result<(), ReparentError> {
        let target_id = Into::<AnyID>::into(target);
        let target_tree_id = self
            .ids
            .tree_id_from_any(target_id)
            .ok_or(ReparentError::TargetError(TargetError::TargetNotFound))?;
        let (destination_id, idx) = self
            .find_location(destination)
            .map_err(ReparentError::DestinationError)?;
        // Are we trying to reparent to one of this node's own children
        // or itself?
        if std::iter::once(destination_id)
            .chain(
                self.tree
                    .ancestor_ids(destination_id)
                    .map_err(|_| ReparentError::DestinationError(TargetError::TargetNotFound))?,
            )
            .any(|ancestor| ancestor == target_tree_id)
        {
            return Err(ReparentError::WouldCycle);
        }
        let destination_id = destination_id.to_owned();

        // Destination is checked, target is not.
        self.tree
            .move_node(
                target_tree_id,
                id_tree::MoveBehavior::ToParent(&destination_id),
            )
            .map_err(|_| ReparentError::DestinationError(TargetError::TargetNotFound))?;

        // unwrap ok - move_node already checked presence of target.
        self.tree.make_nth_sibling(target_tree_id, idx).unwrap();

        Ok(())
    }
    /// Get the blend of the given node, or None if no blend is assigned
    /// (for example on passthrough nodes or Note leaves)
    pub fn blend_of(&self, target: impl Into<AnyID>) -> Result<Option<Blend>, TargetError> {
        let node_data = self
            .tree
            .get(
                self.ids
                    .tree_id_from_any(target.into())
                    .ok_or(TargetError::TargetNotFound)?,
            )
            .map_err(|_| TargetError::TargetNotFound)?
            .data();
        if node_data.deleted {
            Err(TargetError::TargetDeleted)
        } else {
            Ok(node_data.blend())
        }
    }
}
/// Very expensive clone impl!
impl Clone for BlendGraph {
    fn clone(&self) -> Self {
        let tree_clone = self.tree.clone();
        let mut new_ids = stable_id::StableIDMap::with_capacity(self.ids.capacity());

        // id_tree's NodeIds get scrombled when cloning, but we want the old FuzzID based references
        // to still work. Reconstruct!
        self.tree
            .traverse_post_order_ids(self.tree.root_node_id().unwrap())
            .unwrap()
            .zip(
                tree_clone
                    .traverse_post_order_ids(tree_clone.root_node_id().unwrap())
                    .unwrap(),
            )
            .for_each(|(original_id, new_id)| {
                // get the FuzzID that corresponds with this node
                if let Some(original_fuzz_id) = self.ids.fuzz_id_from(&original_id) {
                    new_ids.insert_pair(new_id, *original_fuzz_id);
                }
            });
        Self {
            tree: tree_clone,
            ids: new_ids,
        }
    }
}

impl crate::commands::CommandConsumer<commands::Command> for BlendGraph {
    fn apply(
        &mut self,
        command: crate::commands::DoUndo<'_, commands::Command>,
    ) -> Result<(), crate::commands::CommandError> {
        use crate::commands::{CommandError, DoUndo};
        use commands::Command;

        match command {
            DoUndo::Do(Command::BlendChanged { from, to, target })
            | DoUndo::Undo(Command::BlendChanged {
                to: from,
                from: to,
                target,
            }) => {
                // Get a mut reference to the node's blend, checking that it matches
                // old state before assigning new state.

                // if-let chains pls...
                let Some(node) = self.get_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                if node.deleted {
                    return Err(CommandError::MismatchedState);
                }
                let Some(blend) = node.blend_mut() else {
                    return Err(CommandError::MismatchedState);
                };
                if blend != from {
                    return Err(CommandError::MismatchedState);
                }
                *blend = *to;
                Ok(())
            }
            DoUndo::Do(Command::LeafInnerTransformChanged {
                target,
                old_transform,
                new_transform,
            })
            | DoUndo::Undo(Command::LeafInnerTransformChanged {
                target,
                old_transform: new_transform,
                new_transform: old_transform,
            }) => {
                let Some(node) = self.get_leaf_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };

                match node {
                    LeafType::StrokeLayer {
                        inner_transform, ..
                    } => {
                        // If NaN This becomes problematic.
                        if inner_transform != old_transform {
                            Err(CommandError::MismatchedState)
                        } else {
                            *inner_transform = *new_transform;
                            Ok(())
                        }
                    }
                    LeafType::Note | LeafType::SolidColor { .. } | LeafType::Text { .. } => {
                        Err(CommandError::MismatchedState)
                    }
                }
            }
            DoUndo::Do(Command::LeafOuterTransformChanged {
                target,
                old_transform,
                new_transform,
            })
            | DoUndo::Undo(Command::LeafOuterTransformChanged {
                target,
                old_transform: new_transform,
                new_transform: old_transform,
            }) => {
                let Some(node) = self.get_leaf_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };

                match node {
                    LeafType::StrokeLayer {
                        outer_transform, ..
                    }
                    | LeafType::Text {
                        outer_transform, ..
                    } => {
                        // If NaN This becomes problematic.
                        if outer_transform != old_transform {
                            Err(CommandError::MismatchedState)
                        } else {
                            *outer_transform = *new_transform;
                            Ok(())
                        }
                    }
                    LeafType::Note | LeafType::SolidColor { .. } => {
                        Err(CommandError::MismatchedState)
                    }
                }
            }
            DoUndo::Do(Command::LeafTyChanged { target, old_ty, ty })
            | DoUndo::Undo(Command::LeafTyChanged {
                old_ty: ty,
                ty: old_ty,
                target,
            }) => {
                // Get a mut reference to the leaf ty, checking that it matches
                // old state before assigning new state.
                let Some(node) = self.get_leaf_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                if node != old_ty {
                    return Err(CommandError::MismatchedState);
                }
                *node = ty.clone();
                Ok(())
            }
            DoUndo::Do(Command::NodeTyChanged { target, old_ty, ty })
            | DoUndo::Undo(Command::NodeTyChanged {
                old_ty: ty,
                ty: old_ty,
                target,
            }) => {
                // Get a mut reference to the node ty, checking that it matches
                // old state before assigning new state.
                let Some(node) = self.get_node_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                if node != old_ty {
                    return Err(CommandError::MismatchedState);
                }
                *node = ty.clone();
                Ok(())
            }
            DoUndo::Do(Command::NodeCreated {
                target,
                ty,
                child_idx: _, // FIXME!
                destination: _,
            }) => {
                // This case only reachable with undo then redo.
                // Clear the deleted flag!
                let Some(node) = self.get_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                if node.node() != Some(ty) || !node.deleted {
                    return Err(CommandError::MismatchedState);
                }
                // todo: check child_idx and destination.
                // problem is it won't necessarily still be in that location
                // because of newly appended nodes!
                node.deleted = false;
                Ok(())
            }
            DoUndo::Undo(Command::NodeCreated {
                target,
                ty,
                child_idx: _, // FIXME!
                destination: _,
            }) => {
                let Some(node) = self.get_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                if node.node() != Some(ty) || node.deleted {
                    return Err(CommandError::MismatchedState);
                }
                // todo: check child_idx and destination.
                // problem is it won't necessarily still be in that location
                // because of newly appended nodes!
                node.deleted = true;
                Ok(())
            }
            DoUndo::Do(Command::LeafCreated {
                target,
                ty,
                child_idx: _, // FIXME!
                destination: _,
            }) => {
                // This case only reachable with undo then redo.
                // Clear the deleted flag!
                let Some(node) = self.get_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                if node.leaf() != Some(ty) || !node.deleted {
                    return Err(CommandError::MismatchedState);
                }
                // todo: check child_idx and destination.
                // problem is it won't necessarily still be in that location
                // because of newly appended nodes!
                node.deleted = false;
                Ok(())
            }
            DoUndo::Undo(Command::LeafCreated {
                target,
                ty,
                child_idx: _, // FIXME!
                destination: _,
            }) => {
                let Some(node) = self.get_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                if node.leaf() != Some(ty) || node.deleted {
                    return Err(CommandError::MismatchedState);
                }
                // todo: check child_idx and destination.
                // problem is it won't necessarily still be in that location
                // because of newly appended nodes!
                node.deleted = true;
                Ok(())
            }
            DoUndo::Do(Command::Reparent {
                target,
                new_parent,
                new_child_idx,
                old_parent,
                old_child_idx: _, // FIXME!
            })
            | DoUndo::Undo(Command::Reparent {
                target,
                old_parent: new_parent,
                old_child_idx: new_child_idx,
                new_parent: old_parent,
                new_child_idx: _, // FIXME!
            }) => {
                // Get the expected current parent's nodeId, or none if root is parent.
                let old_parent_tree_id = match old_parent.map(|p| self.ids.tree_id_from_node(p)) {
                    Some(None) => return Err(CommandError::UnknownResource),
                    Some(Some(id)) => Some(id),
                    None => None,
                };

                // Compare this to the actual parent.
                if self
                    .tree
                    .get(
                        self.ids
                            .tree_id_from_any(*target)
                            .ok_or(CommandError::UnknownResource)?,
                    )
                    .map_err(|_| CommandError::UnknownResource)?
                    .parent()
                    // Make the root "None"
                    .filter(|id| Some(*id) != self.tree.root_node_id())
                    != old_parent_tree_id
                {
                    return Err(CommandError::MismatchedState);
                };

                // todo: check old_child_idx.
                // problem is it won't necessarily still be in that location
                // because of newly appended nodes!

                let result = self.reparent(
                    *target,
                    match new_parent {
                        Some(parent) => Location::IndexIntoNode(parent, *new_child_idx),
                        None => Location::IndexIntoRoot(*new_child_idx),
                    },
                );

                match result {
                    Err(
                        ReparentError::TargetError(TargetError::TargetNotFound)
                        | ReparentError::DestinationError(TargetError::TargetNotFound),
                    ) => Err(CommandError::UnknownResource),
                    Err(
                        ReparentError::TargetError(TargetError::TargetDeleted)
                        | ReparentError::DestinationError(TargetError::TargetDeleted)
                        | ReparentError::WouldCycle,
                    ) => Err(CommandError::MismatchedState),
                    Ok(()) => Ok(()),
                }
            }
            DoUndo::Do(Command::AnyDeleted { target }) => {
                let Some(node) = self.get_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                // Can't delete a deleted node
                if node.deleted {
                    return Err(CommandError::MismatchedState);
                }
                node.deleted = true;
                Ok(())
            }
            DoUndo::Undo(Command::AnyDeleted { target }) => {
                let Some(node) = self.get_mut(*target) else {
                    return Err(CommandError::UnknownResource);
                };
                // Can't un-delete a non deleted node
                if !node.deleted {
                    return Err(CommandError::MismatchedState);
                }
                node.deleted = false;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn id_transitivity() {
        let mut graph = BlendGraph::default();
        let soup_id = graph
            .add_leaf(
                Location::IndexIntoRoot(0),
                "Soup!".to_string(),
                LeafType::Note,
            )
            .unwrap();

        let clone = graph.clone();
        assert_eq!(clone.get(soup_id).map(NodeData::name), Some("Soup!"));
    }
}

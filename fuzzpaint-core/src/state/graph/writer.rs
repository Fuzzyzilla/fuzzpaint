use super::{commands::Command, TargetError};
use crate::queue::writer::CommandWrite;

#[derive(thiserror::Error, Debug)]
pub enum CommandError<Err: std::error::Error> {
    #[error(transparent)]
    Inner(#[from] Err),
    #[error("command cannot be applied to the current state")]
    MismatchedState,
}

type Result<T, Err> = std::result::Result<T, CommandError<Err>>;

pub struct GraphWriter<'a, Write: CommandWrite<Command>> {
    writer: Write,
    graph: &'a mut super::BlendGraph,
}
// Inherit immutable methods from the graph, so that the writing code can read the graph
// to inform it's modifications.
impl<'a, Write: CommandWrite<Command>> std::ops::Deref for GraphWriter<'a, Write> {
    type Target = super::BlendGraph;
    fn deref(&self) -> &Self::Target {
        self.graph
    }
}
impl<'a, Write: CommandWrite<Command>> GraphWriter<'a, Write> {
    pub fn new(writer: Write, graph: &'a mut super::BlendGraph) -> Self {
        Self { writer, graph }
    }
    /// Access the name of a node, or None if not found.
    /// Name changes are NOT tracked by the command queue, however
    /// this is still the most correct way to access the Graph mutably.
    pub fn name_mut(&mut self, target: super::AnyID) -> Option<&mut String> {
        self.graph.get_mut(target).map(super::NodeData::name_mut)
    }
    /// Change the blend of any node or leaf. Does not insert a command
    /// if the blend is identical to what it was before!
    /// Returns `MismatchedState` if the chosen node does not have a blend property to modify.
    pub fn change_blend(
        &mut self,
        target: super::AnyID,
        to: crate::blend::Blend,
    ) -> Result<(), TargetError> {
        // Get node, check it's not deleted
        let node = self
            .graph
            .get_mut(target)
            .ok_or(TargetError::TargetNotFound)?;
        // Ensure not deleted
        if node.deleted {
            return Err(TargetError::TargetDeleted.into());
        }
        // No blend attribute on this node = err!
        let Some(blend) = node.blend_mut() else {
            return Err(CommandError::MismatchedState);
        };
        // Perform the change if it's not already matching:
        let from = *blend;
        if from != to {
            *blend = to;
            // Insert command
            self.writer
                .write(Command::BlendChanged { from, to, target });
        }
        Ok(())
    }
    pub fn reparent(
        &mut self,
        target: super::AnyID,
        location: super::Location<'_>,
    ) -> Result<(), super::ReparentError> {
        // Get original idx
        let (old_parent, old_idx) =
            self.graph
                .location_of(target)
                .ok_or(super::ReparentError::TargetError(
                    TargetError::TargetNotFound,
                ))?;
        // Ensure not deleted.
        if self
            .graph
            .get(target)
            .ok_or(super::ReparentError::TargetError(
                TargetError::TargetNotFound,
            ))?
            .deleted
        {
            return Err(CommandError::MismatchedState);
        }

        // perform reparent
        self.graph.reparent(target, location)?;
        // Cannot return err if this fails, that would result in mismatched state and command tree.
        let (new_parent, new_idx) = self.graph.location_of(target).unwrap();

        // Only write if changed
        if (new_parent, new_idx) != (old_parent, old_idx) {
            self.writer.write(Command::Reparent {
                target,
                new_parent,
                new_child_idx: new_idx,
                old_parent,
                old_child_idx: old_idx,
            });
        }

        Ok(())
    }
    pub fn add_leaf(
        &mut self,
        leaf_ty: super::LeafType,
        location: super::Location<'_>,
        name: impl Into<String>,
    ) -> Result<super::LeafID, TargetError> {
        let new_id = self
            .graph
            .add_leaf(location, name.into(), leaf_ty.clone())?;
        // Is this useful? Roundtrips the Location into a true location.
        // Allows the graph to determine Location behavior rather than duplicating it :3
        let (parent, idx) = self.graph.location_of(new_id).unwrap();

        self.writer.write(Command::LeafCreated {
            target: new_id,
            ty: leaf_ty,
            destination: parent,
            child_idx: idx,
        });

        Ok(new_id)
    }
    pub fn set_leaf(
        &mut self,
        target: super::LeafID,
        to: super::LeafType,
    ) -> Result<(), TargetError> {
        let node = self
            .graph
            .get_mut(target)
            .ok_or(TargetError::TargetNotFound)?;
        // Ensure not deleted
        if node.deleted {
            return Err(TargetError::TargetDeleted.into());
        };
        // Is this a possible error state?
        let Some(leaf_ty) = node.leaf_mut() else {
            return Err(CommandError::MismatchedState);
        };
        let from = leaf_ty.clone();
        // Perform the change if it's not already matching:
        if to != from {
            *leaf_ty = to.clone();
            self.writer.write(Command::LeafTyChanged {
                target,
                old_ty: from,
                ty: to,
            });
        }

        Ok(())
    }
    pub fn add_node(
        &mut self,
        node_ty: super::NodeType,
        location: super::Location<'_>,
        name: impl Into<String>,
    ) -> Result<super::NodeID, TargetError> {
        let new_id = self
            .graph
            .add_node(location, name.into(), node_ty.clone())?;
        // Is this useful? Roundtrips the Location into a true location.
        // Allows the graph to determine Location behavior rather than duplicating it :3
        let (parent, idx) = self.graph.location_of(new_id).unwrap();

        self.writer.write(Command::NodeCreated {
            target: new_id,
            ty: node_ty,
            destination: parent,
            child_idx: idx,
        });

        Ok(new_id)
    }
    pub fn set_node(
        &mut self,
        target: super::NodeID,
        to: super::NodeType,
    ) -> Result<(), TargetError> {
        let node = self
            .graph
            .get_mut(target)
            .ok_or(TargetError::TargetNotFound)?;
        // Ensure not deleted
        if node.deleted {
            return Err(TargetError::TargetDeleted.into());
        };
        // Is this a possible error state?
        let Some(node_ty) = node.node_mut() else {
            return Err(CommandError::MismatchedState);
        };
        let from = node_ty.clone();
        // Perform the change if it's not already matching:
        if to != from {
            *node_ty = to.clone();
            self.writer.write(Command::NodeTyChanged {
                target,
                old_ty: from,
                ty: to,
            });
        }

        Ok(())
    }
    pub fn delete(&mut self, target: super::AnyID) -> Result<(), TargetError> {
        let node = self
            .graph
            .get_mut(target)
            .ok_or(TargetError::TargetNotFound)?;
        if node.deleted {
            return Err(CommandError::MismatchedState);
        }
        node.deleted = true;

        self.writer.write(Command::AnyDeleted { target });

        Ok(())
    }
    pub fn set_inner_transform(
        &mut self,
        target: super::LeafID,
        transform: crate::state::transform::Similarity,
    ) -> Result<(), TargetError> {
        let node = self
            .graph
            .get_mut(target)
            .ok_or(TargetError::TargetNotFound)?;
        if node.deleted {
            return Err(TargetError::TargetDeleted.into());
        }

        let Some(leaf) = node.leaf_mut() else {
            return Err(CommandError::MismatchedState);
        };
        match leaf {
            super::LeafType::StrokeLayer {
                inner_transform, ..
            } => {
                let old = *inner_transform;
                if old == transform {
                    return Ok(());
                }
                *inner_transform = transform;
                self.writer.write(Command::LeafInnerTransformChanged {
                    target,
                    old_transform: old,
                    new_transform: transform,
                });
                Ok(())
            }
            _ => Err(CommandError::MismatchedState),
        }
    }
    pub fn set_outer_transform(
        &mut self,
        target: super::LeafID,
        transform: crate::state::transform::Matrix,
    ) -> Result<(), TargetError> {
        let node = self
            .graph
            .get_mut(target)
            .ok_or(TargetError::TargetNotFound)?;
        if node.deleted {
            return Err(TargetError::TargetDeleted.into());
        }

        let Some(leaf) = node.leaf_mut() else {
            return Err(CommandError::MismatchedState);
        };
        match leaf {
            super::LeafType::StrokeLayer {
                outer_transform, ..
            }
            | super::LeafType::Text {
                outer_transform, ..
            } => {
                let old = *outer_transform;
                if old == transform {
                    return Ok(());
                }
                *outer_transform = transform;
                self.writer.write(Command::LeafOuterTransformChanged {
                    target,
                    old_transform: old,
                    new_transform: transform,
                });
                Ok(())
            }
            _ => Err(CommandError::MismatchedState),
        }
    }
}

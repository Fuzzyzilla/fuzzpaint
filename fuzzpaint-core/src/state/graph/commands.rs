#[derive(Clone, Debug)]
pub enum Command {
    BlendChanged {
        from: crate::blend::Blend,
        to: crate::blend::Blend,
        target: super::AnyID,
    },
    Reparent {
        target: super::AnyID,
        /// New parent, or None if root.
        new_parent: Option<super::NodeID>,
        new_child_idx: usize,
        /// Old parent, or None if root.
        old_parent: Option<super::NodeID>,
        old_child_idx: usize,
    },
    LeafCreated {
        target: super::LeafID,
        ty: super::LeafType,
        /// New parent, or None if root.
        destination: Option<super::NodeID>,
        child_idx: usize,
    },
    LeafTyChanged {
        target: super::LeafID,
        old_ty: super::LeafType,
        ty: super::LeafType,
    },
    NodeCreated {
        target: super::NodeID,
        ty: super::NodeType,
        /// New parent, or None if root.
        destination: Option<super::NodeID>,
        child_idx: usize,
    },
    NodeTyChanged {
        target: super::NodeID,
        old_ty: super::NodeType,
        ty: super::NodeType,
    },
    AnyDeleted {
        target: super::AnyID,
    },
}

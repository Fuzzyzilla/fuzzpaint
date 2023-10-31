//! Algs to help with rendering.
//!
//! Performs no rendering itself, merely provides descriptions of the optimal way to render the graph
//! based on a description of the changes.

pub enum Changes<'a> {
    /// A node or leaf's blend mode was changed. Includes going from passthrough to blend, or vice-versa.
    BlendChanged(&'a super::AnyID),
    /// A leaf's contents where changed (not including blend)
    LeafChanged(&'a super::LeafID),
}
impl Changes<'_> {
    fn any_id(&self) -> super::AnyID {
        match self {
            Self::BlendChanged(any) => (*any).clone(),
            Self::LeafChanged(leaf) => (*leaf).clone().into(),
        }
    }
}
/// Get a list of all the objects dirtied by the set of changes, in no particular order.
/// Includes a deduplicated set of parents, grandparents, ect. of modified leaves and nodes, including the modified objects themselves.
/// Unknown nodes will be silently ignored, and will not be included in the returned set.
pub fn dirtied_by(
    graph: &super::BlendGraph,
    changes: &[Changes],
) -> hashbrown::HashSet<super::AnyID> {
    let mut dirtied = hashbrown::HashSet::<super::AnyID>::default();
    'next_change: for change in changes {
        let change_any_id = change.any_id();
        let Some(change_tree_id) = graph.ids.tree_id_from_any(&change_any_id) else {
            continue 'next_change;
        };
        // Get ancestors if any, skipping if not found.
        if let Ok(ancestors) = graph.tree.ancestor_ids(change_tree_id) {
            // Found! insert self into dirtied list.
            dirtied.insert(change_any_id);
            // Insert all ancestors (except root) with appropriate ID type:
            'next_ancestor: for ancestor in ancestors {
                let Ok(ancestor_node) = graph.tree.get(ancestor) else {
                    continue 'next_ancestor;
                };
                let Some(ancestor_fuzz_id) = graph.ids.fuzz_id_from(ancestor).cloned() else {
                    continue 'next_ancestor;
                };
                match ancestor_node.data() {
                    super::NodeData {
                        ty: super::NodeDataTy::Leaf(..),
                        ..
                    } => {
                        dirtied.insert(super::AnyID::Leaf(super::LeafID(ancestor_fuzz_id)));
                    }
                    super::NodeData {
                        ty: super::NodeDataTy::Node(..),
                        ..
                    } => {
                        dirtied.insert(super::AnyID::Node(super::NodeID(ancestor_fuzz_id)));
                    }
                    super::NodeData {
                        ty: super::NodeDataTy::Root,
                        ..
                    } => {
                        // Don't expose the root to the user!
                        ()
                    }
                }
            }
        }
    }
    dirtied
}

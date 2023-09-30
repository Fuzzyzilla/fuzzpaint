//! Command Queue
//!
//! The global queues manage all the actions performed by the user, keeping track of commands, undo/redo state, etc.
//! The queues are the ground truth for the current state of the program and their corresponding document. Listeners to the queue
//! can be various stages of out-of-date, at any point they can view all new commands and bring themselves back to the present.
//!
//! If a listener is greatly out-of-date, the order of commands it sees may not match the exact order of events, but the outcome
//! will be the same. (For example, an unobserved undo followed by a redo will result in neither being reported).
//!
//! There exists one command queue per document.

pub struct CommandAtomsWriter {}
struct DocumentCommandQueueInner {
    /// Tree structure of commands, where undos create branches.
    /// "First child" represents earlier series of commands that were undone, "last" is the most recent.
    /// More than two branches are allowed, of course!
    command_tree: slab_tree::Tree<super::Command>,
    // "Pointer" into the tree where the most recent command took place.
    cursor: slab_tree::NodeId,
}
pub struct DocumentCommandQueue {
    /// Mutable inner bits.
    inner: std::sync::Arc<parking_lot::RwLock<DocumentCommandQueueInner>>,
    document: crate::FuzzID<crate::Document>,
}
impl DocumentCommandQueue {
    /// Atomically push write some number of commands in an Atoms scope, such that they are treated as one larger command.
    /// Ordering between threads is not guarunteed, other than the fact that no commands from other threads are written inside the resulting atoms scope.
    pub fn write_atoms(&self, f: impl FnOnce(&mut CommandAtomsWriter)) {}
}
pub struct DocumentCommandListener {
    document: crate::WeakID<crate::Document>,
    // Cursor into the tree that this listener has last seen,
    // When more events are requested, the path to the "true" cursor is found and traversed.
    cursor: slab_tree::NodeId,
    inner: std::sync::Arc<parking_lot::RwLock<DocumentCommandQueueInner>>,
}
impl DocumentCommandListener {
    pub fn update(&mut self, f: impl FnMut(super::DoUndo<'_>)) {
        let lock = self.inner.read();
        let traverse = traverse(&lock.command_tree, self.cursor, lock.cursor).unwrap();
        self.cursor = lock.cursor;
        traverse.for_each(f)
    }
}

// Traverses the shortest path from one tree node to another.
// A traversal is an optional walk up to the closest ancestor, followed by walking down.
struct TreeTraverser<'t> {
    // current point of the traversal
    cur: slab_tree::NodeRef<'t, super::Command>,
    tree: &'t slab_tree::Tree<super::Command>,

    // Common ancestor. May be equal to end, but never equal to start (we'd be walking down then).
    // Or None if we're walking down (i.e. start *is* the common ancestor)
    ancestor: Option<slab_tree::NodeId>,
    // Path from the end up to the ancestor. Includes the ID of the branch point and the child idx.
    path_down: Vec<(slab_tree::NodeId, usize)>,
    // destination of the traversal.
    end: slab_tree::NodeId,
}

impl<'t> Iterator for TreeTraverser<'t> {
    type Item = super::DoUndo<'t>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ancestor) = self.ancestor {
            // Ancestor is Some, we're going up!
            // Undo, then move cur
            let result = super::DoUndo::Undo(self.cur.data());
            // Move up. Parent will be some, as we know there's a common ancestor.
            // Kinda silly code, as NodeRef.parent borrows the ref, not the tree.
            self.cur = self.tree.get(self.cur.parent()?.node_id())?;

            // We've reached the top of traversal! Go down now.
            if self.cur.node_id() == ancestor {
                self.ancestor = None;
            }

            Some(result)
        } else {
            // We made it!
            if self.cur.node_id() == self.end {
                return None;
            }

            // Ancestor is None, going down.
            // Move cur, then "Do" (opposite order)

            // Find the index of cur to go to next.
            // Kinda a logic mess lol
            // Last item will be next path to go down. Only consume it if the node id matches,
            // Otherwise default to first child.
            let child_idx = if let Some((node_id, child_idx)) = self.path_down.last().cloned() {
                if node_id == self.cur.node_id() {
                    // Consume path, return
                    self.path_down.pop();
                    Some(child_idx)
                } else {
                    None
                }
            } else {
                None
            };
            // Default to first child
            let child_idx = child_idx.unwrap_or(0);
            // Move
            self.cur = self
                .tree
                .get(self.cur.children().nth(child_idx)?.node_id())?;

            let result = super::DoUndo::Do(self.cur.data());

            Some(result)
        }
    }
}

/// Find the ID of the nearest ancestor of A and B, or None if the IDs do not come from the same tree.
/// The endpoints themselves could be the ancestor, if one is a parent of another!
fn nearest_ancestor(
    tree: &slab_tree::Tree<super::Command>,
    a: slab_tree::NodeId,
    b: slab_tree::NodeId,
) -> Option<slab_tree::NodeId> {
    let a_node = tree.get(a)?;
    let b_node = tree.get(b)?;

    // Collect the ID of A, followed by the ancestors of A.
    let parents_of_a: Vec<_> = std::iter::once(a)
        .chain(a_node.ancestors().map(|node| node.node_id()))
        .collect();
    // Iterate over the ID of B, followed by the ancestors of B, and find the first one
    // that is shared. Because of traversal order this will be the nearest ancestor!
    // Won't be found if they come from different trees within the same structure.
    std::iter::once(b)
        .chain(b_node.ancestors().map(|node| node.node_id()))
        .find(|b_ancestor| parents_of_a.contains(b_ancestor))
}

/// Create an iterator that traverses the shortest path between start and end nodes, or None if the start
/// and end nodes are not from the same tree.
fn traverse<'t>(
    tree: &'t slab_tree::Tree<super::Command>,
    start: slab_tree::NodeId,
    end: slab_tree::NodeId,
) -> Option<TreeTraverser<'t>> {
    let ancestor = nearest_ancestor(tree, start, end)?;

    // Find the path from the ancestor to the end.
    // This is expensive! :O Ma
    let path_down = {
        let mut path_down = Vec::<(slab_tree::NodeId, usize)>::new();
        // Will be some - nearest_ancestor already checked.
        let mut cur_ref = tree.get(end)?;
        loop {
            // Will be Some, as we know there's a common ancestor. Will break before this becomes None.
            let parent = cur_ref.parent()?;
            // Will be found - of course the child is a child of it's parent :P
            let (child_idx, _) = parent
                .children()
                .enumerate()
                .find(|(_, node)| node.node_id() == cur_ref.node_id())?;
            // Default to the zero'th child. That way, nodes with only one child won't
            // be collected, otherwise we're just storing the whole tree! :P
            if child_idx != 0 {
                path_down.push((parent.node_id(), child_idx));
            }

            if parent.node_id() == ancestor {
                break;
            }
            cur_ref = tree.get(parent.node_id())?;
        }
        path_down
    };

    Some(TreeTraverser {
        cur: tree.get(start)?,
        tree,
        ancestor: (ancestor != start).then_some(ancestor),
        path_down,
        end,
    })
}

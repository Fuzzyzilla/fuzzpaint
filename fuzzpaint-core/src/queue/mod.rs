//! Command Queue
//!
//! The global queues manage all the actions performed by the user, keeping track of commands, undo/redo state, etc.
//! The queues are the ground truth for the current state of the program and their corresponding document. Listeners to the queue
//! can be various stages of out-of-date, at any point they can view all new commands and bring themselves back to the present.
//!
//! If a listener is greatly out-of-date, the order of commands it sees may not match the exact order of events, but the outcome
//! will be the same. (For example, an unobserved undo followed by a redo will result in neither being reported).
//!
//! There exists one command queue per document, accessed through [provider]s

use std::sync::Arc;

use crate::{
    commands::{self, CommandConsumer},
    state,
};

mod queue_state;
pub mod state_reader;
pub mod writer;

struct DocumentCommandQueueInner {
    /// Tree structure of commands, where undos create branches.
    /// "First child" represents earlier series of commands that were undone, "last" is the most recent.
    /// More than two branches are allowed, of course!
    command_tree: slab_tree::Tree<commands::Command>,
    state: queue_state::State,
    // "Pointer" into the tree where the most recent command took place.
    root: slab_tree::NodeId,
}
pub struct DocumentCommandQueue {
    /// Mutable inner bits.
    inner: std::sync::Arc<parking_lot::RwLock<DocumentCommandQueueInner>>,
    document: state::DocumentID,
}
impl Default for DocumentCommandQueue {
    fn default() -> Self {
        let command_tree = slab_tree::TreeBuilder::new()
            .with_root(commands::Command::Dummy)
            .build();
        let root = command_tree.root_id().unwrap();
        Self {
            inner: Arc::new(
                DocumentCommandQueueInner {
                    state: queue_state::State::new(root),
                    command_tree,
                    root,
                }
                .into(),
            ),
            document: crate::FuzzID::default(),
        }
    }
}
impl DocumentCommandQueue {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    // Create a queue from data, without a history.
    #[must_use]
    pub fn from_state(
        document: state::Document,
        blend_graph: state::graph::BlendGraph,
        stroke_state: state::stroke_collection::StrokeCollectionState,
    ) -> Self {
        let command_tree = slab_tree::TreeBuilder::new()
            .with_root(commands::Command::Dummy)
            .build();
        let root = command_tree.root_id().unwrap();
        Self {
            inner: Arc::new(
                DocumentCommandQueueInner {
                    state: queue_state::State {
                        document,
                        graph: blend_graph,
                        stroke_state,
                        present: root,
                    },
                    command_tree,
                    root,
                }
                .into(),
            ),
            document: crate::FuzzID::default(),
        }
    }
    #[must_use]
    pub fn id(&self) -> state::DocumentID {
        self.document
    }
    /// Locks the queue for writing commands during the span of the closure, where each modification of the state is tracked
    /// by the command queue. If multiple commands are written, they will be written in order as a single Atoms scope.
    pub fn write_with<F, T>(&self, write: F) -> T
    where
        F: FnOnce(&mut writer::CommandQueueWriter<'_>) -> T,
    {
        let (result, _changed) = {
            let lock = self.inner.write();
            let mut writer = writer::CommandQueueWriter {
                lock,
                commands: smallvec::SmallVec::new(),
            };
            // Panic safe - `writer::CommandQueueWriter`'s Drop impl will do the cleanup ensuring the queue's commands and state are synchronized.
            // However, changes will not be notified.
            let result = write(&mut writer);
            (result, writer.changed())
        };
        result
    }
    /// A helper method to view the state as it is at this moment as a clone.
    #[must_use]
    pub fn peek_clone_state(&self) -> state_reader::CommandQueueCloneLock {
        // Unwrap OK - ref to self means queue is alive valid during this method,
        // and we don't anticipate a broken command graph ofc...
        self.listen_from_now().forward_clone_state().unwrap()
    }
    pub fn undo_n(&self, num: usize) {
        let _changed = {
            // Linearly walk up the tree num steps. Todo: a more sophisticated approach, allowing for full navigation
            // of the tree!
            let mut lock = self.inner.write();
            let DocumentCommandQueueInner {
                command_tree,
                root,
                state,
            } = &mut *lock;
            let start = state.present;
            let Some(ancestors) = command_tree.get(state.present).map(|this| this.ancestors())
            else {
                // Cursor not found - shouldn't be possible, as the tree is never trimmed!
                // This kinda means the command tree is now in an unusable state...
                panic!(
                    "Current Node {:?} not found in command tree!",
                    state.present
                );
            };
            let new_cursor = ancestors.take(num).last();
            state.present = new_cursor.map_or(*root, |node| node.node_id());
            let end = state.present;
            // Apply state changes from the commands:
            for command in traverse(command_tree, start, end).unwrap() {
                state.apply(command).unwrap();
            }

            // Changed if we ended up in a different spot!
            start != end
        };
    }
    pub fn redo_n(&self, num: usize) {
        let _changed = {
            // Step down the tree, taking the last (most recent) child every time.
            let mut lock = self.inner.write();
            let DocumentCommandQueueInner {
                command_tree,
                state,
                ..
            } = &mut *lock;
            let start = state.present;
            for _ in 0..num {
                let Some(this) = command_tree.get(state.present) else {
                    // Cursor not found - shouldn't be possible, as the tree is never trimmed!
                    // This kinda means the command tree is now in an unusable state...
                    panic!(
                        "Current Node {:?} not found in command tree!",
                        state.present
                    );
                };
                let Some(last_child) = this.last_child() else {
                    // We've gone as deep as we can go!
                    return;
                };
                state.present = last_child.node_id();
            }
            let end = state.present;
            // Apply state changes from the commands:
            for command in traverse(command_tree, start, end).unwrap() {
                state.apply(command).unwrap();
            }
            // Changed if we ended up in a different spot!
            start != end
        };
    }
    /// Create a listener that starts at the beginning of history.
    #[must_use]
    pub fn listen_from_start(&self) -> DocumentCommandListener {
        let start = self.inner.read().root;
        DocumentCommandListener {
            _document: self.document,
            cursor: start,
            inner: std::sync::Arc::downgrade(&self.inner),
        }
    }
    /// Create a listener that will only see new activity
    #[must_use]
    pub fn listen_from_now(&self) -> DocumentCommandListener {
        let start = self.inner.read().state.present;
        DocumentCommandListener {
            _document: self.document,
            cursor: start,
            inner: std::sync::Arc::downgrade(&self.inner),
        }
    }
}
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ListenerError {
    #[error("document not available")]
    DocumentClosed,
    // Hints that something has gone horribly wrong internally!
    #[error("tree malformed: {}", .0)]
    TreeMalformed(TraverseError),
}
pub struct DocumentCommandListener {
    _document: crate::state::DocumentID,
    // Cursor into the tree that this listener has last seen,
    // When more events are requested, the path to the "true" cursor is found and traversed.
    cursor: slab_tree::NodeId,
    inner: std::sync::Weak<parking_lot::RwLock<DocumentCommandQueueInner>>,
}
impl DocumentCommandListener {
    /// Locks the shared state, without forwarding this listener's point in time.
    /// See [`state_reader::CommandQueueLock`]
    pub fn peek_lock_state(&self) -> Result<state_reader::CommandQueueReadLock, ListenerError> {
        todo!()
    }
    /// Locks the shared state, bringing this listener up-to-date in the process.
    /// See [`state_reader::CommandQueueLock`]
    pub fn forward_lock_state(
        &mut self,
    ) -> Result<state_reader::CommandQueueReadLock, ListenerError> {
        todo!()
    }
    /// Locks or clones the shared state, without forwarding this listener's point in time.
    /// See [`state_reader::CommandQueueCloneLock`]
    pub fn peek_clone_state(
        &'_ self,
    ) -> Result<state_reader::CommandQueueCloneLock, ListenerError> {
        let inner = self.inner.upgrade().ok_or(ListenerError::DocumentClosed)?;
        let lock = inner.read();
        // Eagerly collect command traversal.
        let commands: Vec<state_reader::OwnedDoUndo<_>> =
            traverse(&lock.command_tree, self.cursor, lock.state.present)
                .map_err(ListenerError::TreeMalformed)?
                .map(Into::into)
                .collect();

        Ok(state_reader::CommandQueueCloneLock {
            inner: self.inner.clone(),
            commands,
            // OOOF!! unconditional big expensive clone, todo here :3
            // Could instead share the state until the moment it's changed, also sharing
            // this Arc'd state between different listeners reading the same point in time. (a highly likely scenario)
            shared_state: Arc::new(lock.state.fork()),
        })
    }
    /// Locks or clones the shared state, bringing this listener up-to-date in the process.
    /// See [`state_reader::CommandQueueCloneLock`]
    pub fn forward_clone_state(
        &'_ mut self,
    ) -> Result<state_reader::CommandQueueCloneLock, ListenerError> {
        let state = self.peek_clone_state()?;
        // Advance cursor to the present state of the lock.
        self.cursor = state.shared_state.present;
        Ok(state)
    }
    /// Moves the cursor forward up-to-date with the documnet, not reporting the changes.
    /// Returns `true` if any change occured.
    pub fn forward(&mut self) -> Result<bool, ListenerError> {
        let inner = self.inner.upgrade().ok_or(ListenerError::DocumentClosed)?;
        let lock = inner.read();

        if lock.state.present != self.cursor {
            self.cursor = lock.state.present;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

// Traverses the shortest path from one tree node to another.
// A traversal is an optional walk up to the closest ancestor, followed by walking down.
struct TreeTraverser<'t, T> {
    // current point of the traversal
    cur: slab_tree::NodeRef<'t, T>,
    tree: &'t slab_tree::Tree<T>,

    // Common ancestor. May be equal to end, but never equal to start (we'd be walking down then).
    // Or None if we're walking down (i.e. start *is* the common ancestor)
    ancestor: Option<slab_tree::NodeId>,
    // Path from the end up to the ancestor. Includes the ID of the branch point and the child idx.
    path_down: Vec<(slab_tree::NodeId, usize)>,
    // destination of the traversal.
    end: slab_tree::NodeId,
}

impl<'t, T> Iterator for TreeTraverser<'t, T> {
    type Item = commands::DoUndo<'t, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ancestor) = self.ancestor {
            // Ancestor is Some, we're going up!
            // Undo, then move cur
            let result = commands::DoUndo::Undo(self.cur.data());
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
            let child_idx = if let Some((node_id, child_idx)) = self.path_down.last().copied() {
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

            let result = commands::DoUndo::Do(self.cur.data());

            Some(result)
        }
    }
}

/// Find the ID of the nearest ancestor of A and B, or None if the IDs do not come from the same tree.
/// The endpoints themselves could be the ancestor, if one is a parent of another!
fn nearest_ancestor<T>(
    tree: &slab_tree::Tree<T>,
    a: slab_tree::NodeId,
    b: slab_tree::NodeId,
) -> Result<slab_tree::NodeId, TraverseError> {
    let a_node = tree.get(a).ok_or(TraverseError::NotFound)?;
    let b_node = tree.get(b).ok_or(TraverseError::NotFound)?;

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
        .ok_or(TraverseError::Disconnected)
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum TraverseError {
    #[error("can't traverse disconnected subtrees")]
    Disconnected,
    #[error("ID not present in tree")]
    NotFound,
}
/// Create an iterator that traverses the shortest path between start and end nodes, or None if the start
/// and end nodes are not from the same tree.
fn traverse<T>(
    tree: &slab_tree::Tree<T>,
    start: slab_tree::NodeId,
    end: slab_tree::NodeId,
) -> Result<TreeTraverser<'_, T>, TraverseError> {
    let ancestor = nearest_ancestor(tree, start, end)?;

    // Find the path from the ancestor to the end.
    // This is expensive!
    let path_down = {
        let mut path_down = Vec::<(slab_tree::NodeId, usize)>::new();
        // Early escape if end is the nearest ancestor -
        // There will be no drilling down phase of the traversal.
        if ancestor != end {
            // Will be some - nearest_ancestor already checked.
            let mut cur_ref = tree.get(end).unwrap();
            loop {
                // Will be Some, as we know there's a common ancestor. Will break before this becomes None.
                let parent = cur_ref.parent().unwrap();
                // Will be found - of course the child is a child of it's parent :P
                let (child_idx, _) = parent
                    .children()
                    .enumerate()
                    .find(|(_, node)| node.node_id() == cur_ref.node_id())
                    .unwrap();
                // Default to the zero'th child. That way, nodes with only one child won't
                // be collected, otherwise we're just storing the whole tree! :P
                if child_idx != 0 {
                    path_down.push((parent.node_id(), child_idx));
                }

                if parent.node_id() == ancestor {
                    break;
                }
                cur_ref = tree.get(parent.node_id()).unwrap();
            }
        }
        path_down
    };

    Ok(TreeTraverser {
        // Unwrap ok - Checked by nearest_ancestor
        cur: tree.get(start).unwrap(),
        tree,
        ancestor: (ancestor != start).then_some(ancestor),
        path_down,
        end,
    })
}

#[cfg(test)]
mod traversal_test {
    use super::{nearest_ancestor, traverse, TraverseError};
    ///```ignore
    ///         0     <deleted>
    ///        / \        |
    ///       /   \       |
    ///      1     2      8
    ///     /|\    |\     |
    ///    / | \   | \    |
    ///   3  4  5  6  7   9
    fn make_test_tree() -> (
        hashbrown::HashMap<i32, slab_tree::NodeId>,
        slab_tree::Tree<i32>,
    ) {
        let mut node_map = hashbrown::HashMap::with_capacity(11);
        let mut tree = slab_tree::TreeBuilder::new()
            .with_capacity(7)
            .with_root(0)
            .build();
        let mut root = tree.root_mut().unwrap();
        node_map.insert(0, root.node_id());

        let mut left = root.append(1);
        node_map.insert(1, left.node_id());
        node_map.insert(3, left.append(3).node_id());
        node_map.insert(4, left.append(4).node_id());
        node_map.insert(5, left.append(5).node_id());
        let mut right = root.append(2);
        node_map.insert(2, right.node_id());
        node_map.insert(6, right.append(6).node_id());
        node_map.insert(7, right.append(7).node_id());
        // Make floating tree fragment
        let mut float = right.append(-1);
        let mut float = float.append(8);
        node_map.insert(8, float.node_id());
        node_map.insert(9, float.append(9).node_id());
        right.remove_last(slab_tree::RemoveBehavior::OrphanChildren);

        (node_map, tree)
    }
    #[test]
    fn test_ancestor() {
        let (ids, tree) = make_test_tree();
        macro_rules! id_of {
            ($id:expr) => {
                ids.get(&$id).copied().unwrap()
            };
        }

        // At root
        assert_eq!(nearest_ancestor(&tree, id_of!(2), id_of!(1)), Ok(id_of!(0)));
        // Symmetric
        assert_eq!(nearest_ancestor(&tree, id_of!(1), id_of!(2)), Ok(id_of!(0)));
        // Disconnected tree
        assert_eq!(nearest_ancestor(&tree, id_of!(8), id_of!(9)), Ok(id_of!(8)));
        // Deeper down
        assert_eq!(nearest_ancestor(&tree, id_of!(3), id_of!(5)), Ok(id_of!(1)));
        // Identity
        assert_eq!(nearest_ancestor(&tree, id_of!(1), id_of!(1)), Ok(id_of!(1)));
        // Broken tree
        assert_eq!(
            nearest_ancestor(&tree, id_of!(1), id_of!(9)),
            Err(TraverseError::Disconnected)
        );
    }
    #[test]
    fn test_traverse() {
        use crate::commands::DoUndo;
        let (ids, tree) = make_test_tree();
        macro_rules! id_of {
            ($id:expr) => {
                ids.get(&$id).copied().unwrap()
            };
        }

        // Long traversal
        assert!(Iterator::eq(
            traverse(&tree, id_of!(7), id_of!(5)).unwrap(),
            [
                DoUndo::Undo(&7),
                DoUndo::Undo(&2),
                DoUndo::Do(&1),
                DoUndo::Do(&5)
            ]
            .into_iter()
        ));
        // Reverse traversal
        assert!(Iterator::eq(
            traverse(&tree, id_of!(5), id_of!(7)).unwrap(),
            [
                DoUndo::Undo(&5),
                DoUndo::Undo(&1),
                DoUndo::Do(&2),
                DoUndo::Do(&7),
            ]
            .into_iter()
        ));
        // End at parent
        assert!(Iterator::eq(
            traverse(&tree, id_of!(6), id_of!(0)).unwrap(),
            [DoUndo::Undo(&6), DoUndo::Undo(&2),].into_iter()
        ));
        // Start at parent
        assert!(Iterator::eq(
            traverse(&tree, id_of!(0), id_of!(3)).unwrap(),
            [DoUndo::Do(&1), DoUndo::Do(&3),].into_iter()
        ));
        // Identity
        assert!(Iterator::eq(
            traverse(&tree, id_of!(2), id_of!(2)).unwrap(),
            [].into_iter()
        ));
        // Broken tree
        assert!(traverse(&tree, id_of!(6), id_of!(9)).is_err());
    }
}

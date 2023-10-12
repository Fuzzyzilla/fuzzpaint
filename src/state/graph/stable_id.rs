//! Glue between FuzzID's and id_tree::NodeId's in order to make IDs stable across clones. This is a baaaad
//! solution, but it is necessary with the current graph impl. I need to write my own!

// Private id type! (public to super)
pub(super) type FuzzNodeID = crate::FuzzID<id_tree::NodeId>;

// Shhh.. they're secretly the same type >:3c
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LeafID(pub(super) FuzzNodeID);
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NodeID(pub(super) FuzzNodeID);
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnyID {
    Leaf(LeafID),
    Node(NodeID),
}
impl std::hash::Hash for AnyID {
    // Forego including type in the hash, as we assume the invariant that a
    // Leaf and Node may not have the same ID.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}
impl AsRef<FuzzNodeID> for LeafID {
    fn as_ref(&self) -> &FuzzNodeID {
        &self.0
    }
}
impl AsRef<FuzzNodeID> for NodeID {
    fn as_ref(&self) -> &FuzzNodeID {
        &self.0
    }
}
impl AsRef<FuzzNodeID> for AnyID {
    fn as_ref(&self) -> &FuzzNodeID {
        match self {
            AnyID::Leaf(LeafID(id)) => id,
            AnyID::Node(NodeID(id)) => id,
        }
    }
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
    fn into_raw(self) -> FuzzNodeID {
        match self {
            AnyID::Leaf(LeafID(id)) => id,
            AnyID::Node(NodeID(id)) => id,
        }
    }
}
pub(super) struct StableIDMap {
    fuzz_to_id: hashbrown::HashMap<FuzzNodeID, id_tree::NodeId>,
    id_to_fuzz: hashbrown::HashMap<id_tree::NodeId, FuzzNodeID>,
}
impl Default for StableIDMap {
    fn default() -> Self {
        Self {
            fuzz_to_id: Default::default(),
            id_to_fuzz: Default::default(),
        }
    }
}
impl StableIDMap {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            fuzz_to_id: hashbrown::HashMap::with_capacity(capacity),
            id_to_fuzz: hashbrown::HashMap::with_capacity(capacity),
        }
    }
    pub fn capacity(&self) -> usize {
        self.fuzz_to_id.capacity().max(self.id_to_fuzz.capacity())
    }
    pub fn tree_id_from_any<'s>(&'s self, any: &'_ AnyID) -> Option<&'s id_tree::NodeId> {
        self.fuzz_to_id.get(any.as_ref())
    }
    pub fn tree_id_from_node<'s>(&'s self, node: &'_ NodeID) -> Option<&'s id_tree::NodeId> {
        self.fuzz_to_id.get(node.as_ref())
    }
    pub fn tree_id_from_leaf<'s>(&'s self, leaf: &'_ LeafID) -> Option<&'s id_tree::NodeId> {
        self.fuzz_to_id.get(leaf.as_ref())
    }
    /// Returns a raw ID, as node type is not stored.
    pub fn fuzz_id_from<'s>(&'s self, tree: &'_ id_tree::NodeId) -> Option<&'s FuzzNodeID> {
        self.id_to_fuzz.get(tree)
    }
    pub fn get_or_insert_tree_id<'s>(&'s mut self, tree: id_tree::NodeId) -> &'s FuzzNodeID {
        let entry = self.id_to_fuzz.entry(tree.clone());
        match entry {
            hashbrown::hash_map::Entry::Occupied(o) => &*o.into_mut(),
            hashbrown::hash_map::Entry::Vacant(v) => {
                // Allocate a new id, update other map, and return.
                let new = v.insert(Default::default());
                self.fuzz_to_id.insert(*new, tree);
                new
            }
        }
    }
    /// Insert a specific correlation between tree id and fuzz id
    pub fn insert_pair(&mut self, tree: id_tree::NodeId, fuzz: FuzzNodeID) {
        self.fuzz_to_id.insert(fuzz.clone(), tree.clone());
        self.id_to_fuzz.insert(tree, fuzz);
    }
    pub fn erase_tree_id(&mut self, tree: &id_tree::NodeId) {
        if let Some(id) = self.id_to_fuzz.remove(tree) {
            let _ = self.fuzz_to_id.remove(&id);
        }
    }
}

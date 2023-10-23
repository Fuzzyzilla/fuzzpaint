pub struct InMemoryDocumentProvider {
    // We don't expect high contention - will only be locked for writing when a new queue is inserted.
    documents: parking_lot::RwLock<
        hashbrown::HashMap<crate::state::DocumentID, super::DocumentCommandQueue>,
    >,
}
// Todo: This sould be a trait! For out-of-process providers, network providers, etc.
// That's a ways away, though :3
impl InMemoryDocumentProvider {
    /// Create and insert a new document, returning it's new ID.
    pub fn insert_new(&self) -> crate::state::DocumentID {
        let new_document = super::DocumentCommandQueue::new();
        let new_id = new_document.id();
        self.documents.write().insert(new_id, new_document);

        new_id
    }
    /// Call the given closure on the document queue with the given ID, if found.
    pub fn inspect<F, T>(&self, id: crate::state::DocumentID, f: F) -> Option<T>
    where
        F: FnOnce(&super::DocumentCommandQueue) -> T,
    {
        Some(f(self.documents.read().get(&id)?))
    }
    /// Iterate over all the open documents, by ID.
    pub fn document_iter(&self) -> impl Iterator<Item = crate::state::DocumentID> {
        let ids: Vec<_> = self.documents.read().keys().cloned().collect();
        ids.into_iter()
    }
}

pub fn provider() -> &'static InMemoryDocumentProvider {
    static ONCE: std::sync::OnceLock<InMemoryDocumentProvider> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| InMemoryDocumentProvider {
        documents: Default::default(),
    })
}

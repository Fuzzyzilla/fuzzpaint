//! # Providers
//!
//! Providers give access to some number of document queues, as well as notifications when new ones are added,
//! old ones removed, and current ones modified. It is the source of ownership for the document data!
//! Although only one is currently implemented, this interface will allow for placing the document data in a daemon,
//! on a server, ect.

/// A provider that keeps documents in-memory.
pub struct InMemoryDocumentProvider {
    // We don't expect high contention - will only be locked for writing when a new queue is inserted.
    documents: parking_lot::RwLock<
        hashbrown::HashMap<crate::state::DocumentID, super::DocumentCommandQueue>,
    >,
    notifier: tokio::sync::broadcast::Sender<ProviderMessage>,
}
// Todo: This sould be a trait! For out-of-process providers, network providers, etc.
// That's a ways away, though :3
impl InMemoryDocumentProvider {
    /// Create and insert a new document, returning it's new ID.
    pub fn insert_new(&self) -> crate::state::DocumentID {
        let mut new_document = super::DocumentCommandQueue::new();
        new_document.send_on_change(self.notifier.clone());
        let new_id = new_document.id();
        self.documents.write().insert(new_id, new_document);
        // Notify existing listeners of the addition
        let _ = self.notifier.send(ProviderMessage::Opened(new_id));

        new_id
    }
    /// Insert a queue into this provider.
    /// If a document with this ID already exists, the untouched queue is returned as an error.
    // (weird way to do it hehe)
    pub fn insert(
        &self,
        mut queue: super::DocumentCommandQueue,
    ) -> Result<(), super::DocumentCommandQueue> {
        let id = queue.id();
        match self.documents.write().entry(id) {
            hashbrown::hash_map::Entry::Occupied(_) => return Err(queue),
            hashbrown::hash_map::Entry::Vacant(v) => {
                queue.send_on_change(self.notifier.clone());
                v.insert(queue);
            }
        }
        // Fellthrough. Must've been ok and writelock has been dropped, we can now advertise new doc!
        let _ = self.notifier.send(ProviderMessage::Opened(id));
        Ok(())
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
    /// Get a receiver for messages about change in state for all documents.
    pub fn change_notifier(&self) -> tokio::sync::broadcast::Receiver<ProviderMessage> {
        self.notifier.subscribe()
    }
    /// Broadcast a ProviderMessage::Modified with the given ID to any change listeners.
    /// Ensures the ID is valid before sending.
    pub fn touch(&self, id: crate::state::DocumentID) {
        // If valid:
        if self.documents.read().contains_key(&id) {
            // Send, ignoring error
            let _ = self.notifier.send(ProviderMessage::Modified(id));
        }
    }
}
impl Default for InMemoryDocumentProvider {
    fn default() -> Self {
        let (send, _) = tokio::sync::broadcast::channel(64);
        Self {
            documents: Default::default(),
            notifier: send,
        }
    }
}
#[derive(Copy, Clone)]
pub enum ProviderMessage {
    /// A new document has been made available to the provider.
    Opened(crate::state::DocumentID),
    /// A document has been modified, i.e. it is likely that existing listeners
    /// will see new commands.
    Modified(crate::state::DocumentID),
    /// A document is no longer available.
    Closed(crate::state::DocumentID),
}
impl ProviderMessage {
    /// Gets the document this message refers to.
    pub fn id(&self) -> crate::state::DocumentID {
        match self {
            Self::Closed(id) | Self::Modified(id) | Self::Opened(id) => *id,
        }
    }
}

pub fn provider() -> &'static InMemoryDocumentProvider {
    static ONCE: std::sync::OnceLock<InMemoryDocumentProvider> = std::sync::OnceLock::new();
    ONCE.get_or_init(Default::default)
}

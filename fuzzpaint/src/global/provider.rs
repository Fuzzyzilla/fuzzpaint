//! # Providers
//!
//! Providers give access to some number of document queues, as well as notifications when new ones are added,
//! old ones removed, and current ones modified. It is the source of ownership for the document data!
//! Although only one is currently implemented, this interface will allow for placing the document data in a daemon,
//! on a server, ect.

use fuzzpaint_core::{queue::DocumentCommandQueue, state::DocumentID};

struct PerDocument {
    queue: DocumentCommandQueue,
}
/// A provider that keeps documents in-memory.
pub struct InMemoryDocumentProvider {
    on_change: parking_lot::Mutex<bus::Bus<ProviderMessage>>,
    // We don't expect high contention - will only be locked for writing when a new queue is inserted.
    documents: parking_lot::RwLock<hashbrown::HashMap<DocumentID, PerDocument>>,
}
// Todo: This sould be a trait! For out-of-process providers, network providers, etc.
// That's a ways away, though :3
impl InMemoryDocumentProvider {
    /// Create and insert a new document, returning it's new ID.
    pub fn insert_new(&self) -> DocumentID {
        let new_document = DocumentCommandQueue::new();
        let new_id = new_document.id();
        let new_document = PerDocument {
            queue: new_document,
        };
        self.documents.write().insert(new_id, new_document);

        self.on_change
            .lock()
            .broadcast(ProviderMessage::Opened(new_id));

        new_id
    }
    /// Insert a queue into this provider.
    /// If a document with this ID already exists, the untouched queue is returned as an error.
    pub fn insert(&self, queue: DocumentCommandQueue) -> Result<(), DocumentCommandQueue> {
        let id = queue.id();
        match self.documents.write().entry(id) {
            hashbrown::hash_map::Entry::Occupied(_) => return Err(queue),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let queue = PerDocument { queue };

                v.insert(queue);
            }
        }

        self.on_change.lock().broadcast(ProviderMessage::Opened(id));

        Ok(())
    }
    /// Call the given closure on the document queue with the given ID, if found.
    pub fn inspect<F, T>(&self, id: DocumentID, f: F) -> Option<T>
    where
        F: FnOnce(&DocumentCommandQueue) -> T,
    {
        let read = self.documents.read();
        let data = read.get(&id)?;

        // Capture state now, check if it matches after user closure..
        let mut cursor = data.queue.listen_from_now();

        let result = f(&data.queue);
        // Avoid locking for any longer than we need to (next may block)
        drop(read);

        if let Ok(true) = cursor.forward() {
            self.on_change
                .lock()
                .broadcast(ProviderMessage::Modified(id));
        }

        Some(result)
    }
    /// Iterate over all the open documents, by ID.
    pub fn document_iter(&self) -> impl Iterator<Item = DocumentID> {
        let ids: Vec<_> = self.documents.read().keys().copied().collect();
        ids.into_iter()
    }
    /// Broadcast a `ProviderMessage::Modified` with the given ID to any change listeners.
    /// Ensures the ID is valid before sending.
    pub fn touch(&self, id: DocumentID) {
        if self.documents.read().contains_key(&id) {
            self.on_change
                .lock()
                .broadcast(ProviderMessage::Modified(id));
        }
    }
    /// Get a reciever of messages describing changes to the provider or it's documents.
    /// Does not recieve old messages, use [`Self::document_iter`] to get up-to-date!
    pub fn change_listener(&self) -> bus::BusReader<ProviderMessage> {
        self.on_change.lock().add_rx()
    }
}
impl Default for InMemoryDocumentProvider {
    fn default() -> Self {
        // Blocks on full, so choose a large number to avoid blocking user thread.
        let on_change = bus::Bus::new(256);
        Self {
            on_change: on_change.into(),
            documents: parking_lot::RwLock::default(),
        }
    }
}
#[derive(Copy, Clone, Debug)]
pub enum ProviderMessage {
    /// A new document has been made available to the provider.
    Opened(DocumentID),
    /// A document has been modified, i.e. it is likely that existing listeners
    /// will see new commands.
    Modified(DocumentID),
    /// A document is no longer available.
    Closed(DocumentID),
}
impl ProviderMessage {
    /// Gets the document this message refers to.
    #[must_use]
    pub fn id(&self) -> DocumentID {
        match self {
            Self::Closed(id) | Self::Modified(id) | Self::Opened(id) => *id,
        }
    }
}

pub fn provider() -> &'static InMemoryDocumentProvider {
    static ONCE: std::sync::OnceLock<InMemoryDocumentProvider> = std::sync::OnceLock::new();
    ONCE.get_or_init(Default::default)
}

//! #IDs
//! For many purposes, a unique ID is needed. This is implemented in this module via the `FuzzID<T>` type,
//! which generates unique IDs namespaced by the type T. Order of IDs is not guaranteed.
//!
//! As cloning and re-use of an ID is a logic error, a weak, clonable + Copyable version is provided with `FuzzID::weak`.

// Collection of pending IDs by type.
static ID_SERVER: std::sync::OnceLock<
    parking_lot::RwLock<hashbrown::HashMap<std::any::TypeId, std::sync::atomic::AtomicU64>>,
> = std::sync::OnceLock::new();

/// ID that is guarunteed unique within this execution of the program.
/// IDs with different types may share a value but should not be considered equal.
pub struct FuzzID<T: std::any::Any> {
    id: std::num::NonZeroU64,
    // Namespace marker
    _phantom: std::marker::PhantomData<T>,
}
impl<T: std::any::Any> Clone for FuzzID<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: std::any::Any> Copy for FuzzID<T> {}
impl<T: std::any::Any> std::cmp::PartialEq<FuzzID<T>> for FuzzID<T> {
    fn eq(&self, other: &FuzzID<T>) -> bool {
        // Namespace already checked at compile time - Self::T == Other::T of course!
        self.id == other.id
    }
}
impl<T: std::any::Any> std::cmp::Eq for FuzzID<T> {}

impl<T: std::any::Any> std::hash::Hash for FuzzID<T> {
    /// A note on hashes - this relies on the internal representation of TypeID,
    /// which is unstable between compilations. Do NOT serialize or otherwise rely on
    /// comparisons between hashes from different executions of the program.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::TypeId::of::<T>().hash(state);
        self.id.hash(state)
    }
}

impl<T: std::any::Any> FuzzID<T> {
    /// Get the raw numeric value of this ID.
    /// IDs from differing namespaces may share the same numeric ID!
    pub fn id(&self) -> u64 {
        self.id.get()
    }
}
impl<T: std::any::Any> Default for FuzzID<T> {
    fn default() -> Self {
        let map = ID_SERVER.get_or_init(Default::default);
        // ID of zero will be invalid, start at one and go up.
        let id = {
            let read = map.upgradable_read();
            let ty = std::any::TypeId::of::<T>();
            if let Some(atomic) = read.get(&ty) {
                //We don't really care about the order things happen in, it just needs
                //to be unique.
                atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            } else {
                // We need to insert into the map - transition to exclusive access
                let mut write = parking_lot::RwLockUpgradableReadGuard::upgrade(read);
                // Initialize at 2, return ID 1
                write.insert(ty, 2.into());
                1
            }
        };

        // Incredibly unrealistic for this to fail - At one brush stroke per second, 24/7/365, it will take
        // half a trillion years to overflow!
        let Some(id) = std::num::NonZeroU64::new(id) else {
            log::error!("{} ID overflow! Aborting!", std::any::type_name::<T>());
            log::logger().flush();
            // Panic is not enough - we cannot allow any threads to continue, global state is unfixably borked!
            // We could instead return an option type, allowing threads to clean up properly but still preventing
            // future ID allocations. However, this failure case is so absurd I don't think it's worth degrading the usability
            // of this API.
            std::process::abort();
        };

        Self {
            id,
            _phantom: Default::default(),
        }
    }
}
impl<T: std::any::Any> std::fmt::Display for FuzzID<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //Unwrap here is safe - the rsplit will always return at least one element, even for empty strings.
        write!(
            f,
            "{}#{}",
            std::any::type_name::<T>().rsplit("::").next().unwrap(),
            self.id
        )
    }
}

impl<T: std::any::Any> std::fmt::Debug for FuzzID<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <FuzzID<T> as std::fmt::Display>::fmt(self, f)
    }
}

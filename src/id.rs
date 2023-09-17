//! #IDs
//! For many purposes, a unique ID is needed. This is implemented in this module via the FuzzID<T> type,
//! which generates unique IDs namespaced by the type T. Order of IDs is not guaranteed.
//!
//! As cloning and re-use of an ID is a logic error, a weak, clonable + Copyable version is provided with `FuzzID::weak`.

// Collection of pending IDs by type.
static ID_SERVER: std::sync::OnceLock<
    parking_lot::RwLock<std::collections::HashMap<std::any::TypeId, std::sync::atomic::AtomicU64>>,
> = std::sync::OnceLock::new();

/// ID that is guarunteed unique within this execution of the program.
/// IDs with different types may share a value but should not be considered equal.
pub struct FuzzID<T: std::any::Any> {
    id: u64,
    // Namespace marker
    _phantom: std::marker::PhantomData<T>,
}
impl<T: std::any::Any> std::cmp::PartialEq<FuzzID<T>> for FuzzID<T> {
    fn eq(&self, other: &FuzzID<T>) -> bool {
        self.weak() == other.weak()
    }
}
impl<T: std::any::Any> std::cmp::Eq for FuzzID<T>{}
impl<T: std::any::Any> std::cmp::PartialEq<WeakID<T>> for &FuzzID<T> {
    fn eq(&self, other: &WeakID<T>) -> bool {
        self.weak() == *other
    }
}
impl<T: std::any::Any> std::cmp::PartialEq<FuzzID<T>> for WeakID<T> {
    fn eq(&self, other: &FuzzID<T>) -> bool {
        *self == other.weak()
    }
}
impl<T: std::any::Any> std::hash::Hash for FuzzID<T> {
    /// A note on hashes - this relies on the internal representation of TypeID,
    /// which is unstable between compilations. Do NOT serialize or otherwise rely on
    /// comparisons between hashes from different executions of the program.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.weak().hash(state);
    }
}

/// Result of a cloned ID. Cannot be used to make an object with a duplicated ID.
pub struct WeakID<T: std::any::Any> {
    id: u64,
    // Namespace marker
    _phantom: std::marker::PhantomData<T>,
}
impl<T: std::any::Any> WeakID<T> {
    pub fn empty() -> Self {
        Self {
            id : 0,
            _phantom: Default::default(),
        }
    }
}
impl<T: std::any::Any> Clone for WeakID<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            _phantom: self._phantom,
        }
    }
}
impl<T: std::any::Any> Copy for WeakID<T> {}
impl<T: std::any::Any> std::cmp::PartialEq for WeakID<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T: std::any::Any> std::cmp::Eq for WeakID<T> {}
impl<T: std::any::Any> std::hash::Hash for WeakID<T> {
    /// A note on hashes - this relies on the internal representation of TypeID,
    /// which is unstable between compilations. Do NOT serialize or otherwise rely on
    /// comparisons between hashes from different executions of the program.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::TypeId::of::<T>().hash(state);
        state.write_u64(self.id);
    }
}

impl<T: std::any::Any> FuzzID<T> {
    pub fn id(&self) -> u64 {
        self.id
    }
    /// Construct a weak clone of this ID.
    pub fn weak(&self) -> WeakID<T> {
        WeakID {
            id: self.id,
            _phantom: self._phantom,
        }
    }
    /// Get a dummy ID. Useful for testing, but can be used for evil.
    pub unsafe fn dummy() -> Self {
        FuzzID {
            id: 0,
            _phantom: Default::default(),
        }
    }
}
impl<T: std::any::Any> Into<WeakID<T>> for &FuzzID<T> {
    fn into(self) -> WeakID<T> {
        self.weak()
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

        // Incredibly unrealistic - At one brush stroke per second, 24/7/365, it will take
        // half a trillion years to overflow. This assert is debug-only, to catch exhaustion from some
        // programming error.
        debug_assert!(id != 0, "{} ID overflow!", std::any::type_name::<T>());

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
impl<T: std::any::Any> std::fmt::Display for WeakID<T> {
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

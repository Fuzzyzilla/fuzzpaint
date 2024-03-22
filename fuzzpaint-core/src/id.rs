//! # IDs
//! For many purposes, a unique ID is needed. This is implemented in this module via the `FuzzID<T>` type,
//! which generates unique IDs namespaced by the type T. Order of IDs is not guaranteed.
//!
//! To get a process unique ID, simply use `FuzzID<YourNamespaceTy>`'s `Default` impl. To eargerly acquire many ids,
//! use `FuzzID::many`.

// Collection of pending IDs by type.
// Type name mess, but a RWLock'd BTreeMap from typeID to next available FuzzID
static ID_SERVER: parking_lot::RwLock<
    std::collections::BTreeMap<std::any::TypeId, std::sync::atomic::AtomicU64>,
> = parking_lot::const_rwlock(std::collections::BTreeMap::new());

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

// Safety - it's literally just a u64 lol
// We need these because if T is !Send or !Sync that is carried
// over to the ID, even though we don't actually store a T and thus
// shouldn't be bound by this.
unsafe impl<T: std::any::Any> Send for FuzzID<T> {}
unsafe impl<T: std::any::Any> Sync for FuzzID<T> {}

impl<T: std::any::Any> std::hash::Hash for FuzzID<T> {
    /// A note on hashes - this relies on the internal representation of `TypeID`,
    /// which is unstable between compilations. Do NOT serialize or otherwise rely on
    /// comparisons between hashes from different executions of the program.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::TypeId::of::<T>().hash(state);
        self.id.hash(state);
    }
}

impl<T: std::any::Any> FuzzID<T> {
    /// Get the raw numeric value of this ID.
    /// IDs from differing namespaces may share the same numeric ID!
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id.get()
    }
    /// Allocate many IDs at once. Much much faster than doing them one at a time for bulk operations and doesn't allocate.
    ///
    /// It is important to limit count to a reasonable amount - it is on the caller to ensure this.
    /// Attempt to allocate after exhaustion of all `u64::MAX - 1` IDs leads to unclean program termination,
    /// which can be reached in a two calls of this fn! Note that IDs are assigned eagerly - dropping the returned
    /// iterator early does *not* recycle the unused IDs.
    ///
    /// *The order of IDs is undefined.* All that's guarunteed is that they're unique.
    pub fn many(count: usize) -> impl ExactSizeIterator<Item = Self> {
        // Count of 0 is not a logic error. it is handled gracefully :3
        // Usize is always <= 64bits
        let count_u64 = count as u64;

        // ID of zero will be invalid, start at one and go up.
        let start_id = {
            let read = ID_SERVER.upgradable_read();
            let ty = std::any::TypeId::of::<T>();
            if let Some(atomic) = read.get(&ty) {
                //We don't really care about the order things happen in, it just needs
                //to be unique.
                atomic.fetch_add(count_u64, std::sync::atomic::Ordering::Relaxed)
            } else {
                // We need to insert into the map - transition to exclusive access
                // This is a hugely uncommon operation, so we optimize for the other path
                // (will only happen a dozen or so times throughout program's entire life)
                let mut write = parking_lot::RwLockUpgradableReadGuard::upgrade(read);
                // Initialize at count+1, return start ID of 1
                // Wrapping add. It's not a logic error to request u64::MAX-1, but it will immediately crash
                // on next alloc. Not a good idea, but not incorrect.
                write.insert(ty, (count_u64.wrapping_add(1)).into());
                1
            }
        };

        // Overflow occured! Todo: next alloc will succeed, should I latch to failure? Current impl
        // results in a situation where the next thread *could* alloc non-unique
        // IDs during the delay from detecting and logging the error.
        #[allow(clippy::manual_assert)]
        if (start_id.wrapping_add(count_u64)) <= count_u64 {
            // In builds, terminate. In testing, panic, so that tests for overflow may be implemented.
            #[cfg(not(test))]
            {
                log::error!("{} ID overflow! Aborting!", std::any::type_name::<T>());
                log::logger().flush();
                // Panic is not enough - we cannot allow any threads to continue, global state is unfixably borked!
                std::process::abort();
            }
            #[cfg(test)]
            {
                panic!("{} ID overflow! Aborting!", std::any::type_name::<T>())
            }
        }

        // Must use `usize` indices for ExactSizeIterator, as absolute values of the IDs would
        // overflow a 32-bit system's usize
        (0..count).map(move |idx| {
            let id = idx as u64 + start_id;
            FuzzID {
                // Non-zero-ness checked by overflow catching logic.
                id: std::num::NonZeroU64::new(id).unwrap(),
                _phantom: std::marker::PhantomData,
            }
        })
    }
}
impl<T: std::any::Any> Default for FuzzID<T> {
    fn default() -> Self {
        Self::many(1).next().unwrap()
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
#[cfg(test)]
mod test {
    use super::FuzzID;
    // These tests are full of anti-patterns used to test this specific impl!!
    // Do not rely on the properties shown here!!!! (other than allocating 0 ids being valid,
    // and all IDs being unique.)

    // Tests modify global shared state, as they're running in one process. :V
    // Thus they must all have their own ID namespace.

    #[test]
    fn none_ids() {
        // Local namespace for testing.
        struct Namespace;
        type TestID = FuzzID<Namespace>;

        // Allocating none should be valid.
        let _ = TestID::many(0);
        let _ = TestID::many(0);
        let _ = TestID::many(0);
        let _ = TestID::many(0);
        let _ = TestID::many(0);

        let id = TestID::default();
        // Not a stable guarantee! Dont use this!!
        assert_eq!(id.id(), 1);
    }
    #[test]
    fn many_ids_unique() {
        // Local namespace for testing.
        struct Namespace;
        type TestID = FuzzID<Namespace>;

        let count = 1024;
        let mut v: Vec<_> = TestID::many(count).collect();

        // Don't do this!!
        v.sort_unstable_by_key(FuzzID::id);
        let length_before = v.len();
        v.dedup();
        let length_after = v.len();

        assert_eq!(length_before, length_after, "had duplicate ids");
    }
    // Test only makes sense if we can fit u64::MAX in a usize
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn near_overflow() {
        // Local namespace for testing.
        struct Namespace;
        type TestID = FuzzID<Namespace>;

        // None of these should panic. Alloc'ing one more should!
        let _ = TestID::many(0);
        // Minus one, as they're NonZeroU64 which has one fewer possible values.
        let _ = TestID::many((u64::MAX - 1) as usize);
        let _ = TestID::many(0);
    }
    // Test only makes sense if we can fit u64::MAX in a usize
    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "ID overflow")]
    fn overflow() {
        // Local namespace for testing.
        struct Namespace;
        type TestID = FuzzID<Namespace>;

        // Does NOT panic. Tested by [near_overflow]
        // (thus near_overflow failing leads to this one spuriously succeeding lol)
        let _ = TestID::many(0);
        // Minus one, as they're NonZeroU64 which has one fewer possible values.
        let _ = TestID::many((u64::MAX - 1) as usize);
        let _ = TestID::many(0);
        // Should panic!
        let _ = TestID::many(1);
    }
}

//! # Points
//!
//! Points have the largest size footprint of all resources, due to how numerous they are.
//! Thus, it makes sense that their repository implementation should recieve the most care.
//! For now, the collection just grows unboundedly and no eviction is done -
//! however, the API is constructed to allow for smart in-memory compression or dumping old
//! data to disk in the future.

/// Get the shared global instance of the point repository.
pub fn globa() -> &'static PointRepository {
    static REPO: std::sync::OnceLock<PointRepository> = std::sync::OnceLock::new();
    REPO.get_or_init(PointRepository::new)
}

pub struct PointCollectionIDMarker;
pub type PointCollectionID = crate::FuzzID<PointCollectionIDMarker>;
pub type WeakPointCollectionID = crate::WeakID<PointCollectionIDMarker>;

/// A handle for reading a collection of points. Can be cloned and shared between threads,
/// however take care not to allow it to become leaked - it will not allow the resources
/// to be reclaimed by the repository for the duration of the lock's lifetime.
#[derive(Clone)]
pub struct PointCollectionReadLock {
    points: &'static [crate::StrokePoint],
}
impl AsRef<[crate::StrokePoint]> for PointCollectionReadLock {
    // seal the detail that this is secretly 'static (shhhh...)
    fn as_ref<'a>(&'a self) -> &'a [crate::StrokePoint] {
        self.points
    }
}
impl std::ops::Deref for PointCollectionReadLock {
    type Target = [crate::StrokePoint];
    // seal the detail that this is secretly 'static (shhhh...)
    fn deref<'a>(&'a self) -> &'a Self::Target {
        self.points
    }
}

#[derive(Clone)]
struct PointCollectionAllocInfo {
    /// Which PointPack is it in?
    /// (currently an index)
    pack_id: usize,
    /// What point index into that pack does it start?
    start: usize,
    /// How many points long?
    len: usize,
}
pub struct PointRepository {
    packs: parking_lot::RwLock<Vec<PointPack>>,
    allocs:
        parking_lot::RwLock<hashbrown::HashMap<WeakPointCollectionID, PointCollectionAllocInfo>>,
}
impl PointRepository {
    fn new() -> Self {
        // Self doesn't impl Default as we don't want any ctors to be public.
        Self {
            packs: Default::default(),
            allocs: Default::default(),
        }
    }
    /// Insert the collection into the repository, yielding a unique ID.
    /// Fails if the length of the collection is > 0x10_00_00
    pub fn insert(&self, collection: &[crate::StrokePoint]) -> Option<PointCollectionID> {
        if collection.len() <= PACK_SIZE {
            // Find a pack where `try_bump` succeeds.
            let pack_reads = self.packs.upgradable_read();
            if let Some((pack_id, start)) = pack_reads
                .iter()
                .enumerate()
                .find_map(|(idx, pack)| Some((idx, pack.try_bump_write(collection)?)))
            {
                // We don't need this lock anymore!
                drop(pack_reads);

                // populate info
                let info = PointCollectionAllocInfo {
                    len: collection.len(),
                    pack_id,
                    start,
                };
                // generate a new id and write metadata
                let id = PointCollectionID::default();
                self.allocs.write().insert(id.weak(), info);
                Some(id)
            } else {
                // No packs were found with space to bump. Make a new one
                let new_pack = PointPack::new();
                // Unwrap is infallible - we checked the size requirement, so there's certainly room!
                let start = new_pack.try_bump_write(collection).unwrap();
                // put the pack into self, getting it's index
                let pack_id = {
                    let mut write = parking_lot::RwLockUpgradableReadGuard::upgrade(pack_reads);
                    write.push(new_pack);
                    write.len()
                };
                // populate info
                let info = PointCollectionAllocInfo {
                    len: collection.len(),
                    pack_id,
                    start,
                };
                // generate a new id and write metadata
                let id = PointCollectionID::default();
                self.allocs.write().insert(id.weak(), info);
                Some(id)
            }
        } else {
            None
        }
    }
    /// Get the number of points in the given point collection, or None if the ID is not known
    /// to this repository.
    ///
    /// Provides the ability to fetch the number of points in a collection even if the data is not resident.
    /// If you need the data afterwards, prefer try_get() followed by PointCollectionReadLock::len()
    pub fn len_of(&self, id: impl Into<WeakPointCollectionID>) -> Option<usize> {
        self.allocs.read().get(&id.into()).map(|info| info.len)
    }
    pub fn try_get(
        &self,
        id: impl Into<WeakPointCollectionID>,
    ) -> Result<PointCollectionReadLock, super::TryRepositoryError> {
        let id = id.into();
        let alloc = self
            .allocs
            .read()
            .get(&id)
            .ok_or(super::TryRepositoryError::NotFound)?
            .clone();
        let packs_read = self.packs.read();
        let Some(pack) = packs_read.get(alloc.pack_id) else {
            // Implementation bug!
            log::debug!("{id} allocation found, but pack doesn't exist!");
            return Err(super::TryRepositoryError::NotFound);
        };
        // Check the alloc range is reasonable
        debug_assert!(alloc
            .start
            .checked_add(alloc.len)
            .is_some_and(|last| last <= PACK_SIZE));

        let Some(slice) = pack.try_read(alloc.start, alloc.len) else {
            // Implementation bug!
            log::debug!("{id} allocation found, but out of bounds within it's pack!");
            return Err(super::TryRepositoryError::NotFound);
        };
        Ok(PointCollectionReadLock { points: slice })
    }
}
// A large collection of continguous points on the heap
struct PointPack {
    /// a non-null pointer to array of PACK_SIZE points.
    points: *mut crate::StrokePoint,
    /// Current past-the-end index for the allocator.
    /// Indices before this are considered immutable, after are considered mutable.
    bump_free_idx: parking_lot::Mutex<usize>,
}
const PACK_SIZE: usize = 1024 * 1024;
impl PointPack {
    /// Try to allocate and write a contiguous section of points, returning the start idx of where it was written.
    /// If not enough space, the `self` is left unchanged and None is returned.
    fn try_bump_write(&self, data: &[crate::StrokePoint]) -> Option<usize> {
        if data.len() <= PACK_SIZE {
            let mut free_idx = self.bump_free_idx.lock();
            let old_idx = *free_idx;
            let new_idx = old_idx.checked_add(data.len())?;
            if new_idx > PACK_SIZE {
                None
            } else {
                // Safety - No shared mutable or immutable access can occur here,
                // since we own the mutex. Todo: could cause much pointless waiting for before the idx!
                let slice: &'static mut [crate::StrokePoint] =
                    unsafe { std::slice::from_raw_parts_mut(self.points.add(old_idx), data.len()) };
                slice
                    .iter_mut()
                    .zip(data.iter())
                    .for_each(|(into, from)| *into = *from);
                *free_idx = new_idx;
                Some(old_idx)
            }
        } else {
            None
        }
    }
    /// Try to read some continuous section of strokes. returns None if the region is outside the span
    /// of the currently allocated memory.
    ///
    /// Performs no check that the given start and length correspond to a single allocation!
    fn try_read(&self, start: usize, len: usize) -> Option<&'static [crate::StrokePoint]> {
        // Check if this whole region is within the allocated, read-only section.
        if start
            .checked_add(len)
            .is_some_and(|past_end| past_end <= *self.bump_free_idx.lock())
        {
            // Safety: no shared mutable access, as mutation never happens before the bump idx
            Some(unsafe { std::slice::from_raw_parts(self.points.add(start), len) })
        } else {
            None
        }
    }
    fn new() -> Self {
        let size = std::mem::size_of::<crate::StrokePoint>() * PACK_SIZE;
        let align = std::mem::align_of::<crate::StrokePoint>();
        debug_assert!(size != 0);
        debug_assert!(align != 0 && align.is_power_of_two());

        // Safety: Size and align constraints ensured by debug asserts and unwraps.
        // (is there a better way to get a large zeroed heap array?)
        let points = unsafe {
            std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align(size, align).unwrap())
                .cast::<crate::StrokePoint>()
        };
        assert!(!points.is_null());
        // We do not dealloc points at any point.
        // The packs will be re-used for the lifetime of the program.
        Self {
            points,
            bump_free_idx: 0.into(),
        }
    }
}
// Safety - the pointer refers to heap mem, and can be transferred.
unsafe impl Send for PointPack {}

// Safety - The mutex prevents similtaneous mutable and immutable access.
unsafe impl Sync for PointPack {}

//! # Points
//!
//! Points have the largest size footprint of all resources, due to how numerous they are.
//! Thus, it makes sense that their repository implementation should recieve the most care.
//! For now, the collection just grows unboundedly and no eviction is done -
//! however, the API is constructed to allow for smart in-memory compression or dumping old
//! data to disk in the future.

pub mod archetype;
pub use archetype::PointArchetype;
pub mod io;

mod slab;
use slab::Slab;

/// Get the shared global instance of the point repository.
pub fn global() -> &'static PointRepository {
    static REPO: std::sync::OnceLock<PointRepository> = std::sync::OnceLock::new();
    REPO.get_or_init(PointRepository::new)
}

#[derive(Copy, Clone)]
pub struct CollectionSummary {
    /// The archetype of the points of the collection.
    pub archetype: PointArchetype,
    /// Count of points within the collection
    pub len: usize,
    /// final arc length of the collected points, available if the archetype includes PointArchetype::ARC_LENGTH bit.
    pub arc_length: Option<f32>,
}
impl CollectionSummary {
    /// Gets the number of elements represented by this summary.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.len.saturating_mul(self.archetype.elements())
    }
}

impl From<&[crate::stroke::Point]> for CollectionSummary {
    fn from(value: &[crate::stroke::Point]) -> Self {
        CollectionSummary {
            archetype: crate::stroke::Point::archetype(),
            len: value.len(),
            arc_length: value.last().map_or(0.0, |point| point.dist).into(),
        }
    }
}

pub struct PointCollectionIDMarker;
pub type PointCollectionID = crate::FuzzID<PointCollectionIDMarker>;

/// A handle for reading a collection of points. Can be cloned and shared between threads,
/// however take care not to allow it to become leaked - it will not allow the resources
/// to be reclaimed by the repository for the duration of the lock's lifetime.
#[derive(Clone)]
pub struct PointCollectionReadLock {
    points: &'static [f32],
}
impl AsRef<[f32]> for PointCollectionReadLock {
    // seal the detail that this is secretly 'static (shhhh...)
    fn as_ref(&'_ self) -> &'_ [f32] {
        self.points
    }
}
impl std::ops::Deref for PointCollectionReadLock {
    type Target = [f32];
    // seal the detail that this is secretly 'static (shhhh...)
    fn deref(&'_ self) -> &'_ Self::Target {
        self.points
    }
}

#[derive(thiserror::Error, Debug)]
pub enum WriteError {
    #[error("point collection {} is unknown", .0)]
    UnknownID(PointCollectionID),
    #[error("too much data")]
    TooLong,
    #[error("too many entries")]
    TooManyEntries,
    #[error(transparent)]
    IOError(#[from] std::io::Error),
}
#[derive(Copy, Clone)]
struct PointCollectionAllocInfo {
    /// Which PointSlab is it in?
    /// (currently an index)
    slab_id: usize,
    /// What *element* index into that slab does it start?
    ///
    /// Note that summary.len is in units of points, not elements.
    start: usize,
    /// A summary of the data within, that can be queried even if the bulk
    /// data is non-resident.
    summary: CollectionSummary,
}
// 4MiB of floats
pub const SLAB_ELEMENT_COUNT: usize = 1024 * 1024;
type ElementSlab = slab::Slab<f32, SLAB_ELEMENT_COUNT>;

pub struct PointRepository {
    slabs: parking_lot::RwLock<Vec<ElementSlab>>,
    allocs: parking_lot::RwLock<hashbrown::HashMap<PointCollectionID, PointCollectionAllocInfo>>,
}
impl PointRepository {
    fn new() -> Self {
        // Self doesn't impl Default as we don't want any ctors to be public.
        Self {
            slabs: parking_lot::RwLock::default(),
            allocs: parking_lot::RwLock::default(),
        }
    }
    /// Get the memory usage of resident data (uncompressed in RAM), in bytes, and the capacity.
    #[must_use]
    pub fn resident_usage(&self) -> (usize, usize) {
        let read = self.slabs.read();
        let num_slabs = read.len();
        let capacity = num_slabs.saturating_mul(ElementSlab::size_bytes());
        let usage = read
            .iter()
            .map(Slab::hint_usage_bytes)
            .fold(0, usize::saturating_add);
        (usage, capacity)
    }
    /// Insert the collection into the repository, yielding a unique ID.
    /// Fails if the length of the collection caintains > [`SLAB_ELEMENT_COUNT`] f32 elements
    #[must_use = "the returned ID is needed to fetch the data in the future"]
    pub fn insert(&self, collection: &[crate::stroke::Point]) -> Option<PointCollectionID> {
        let elements = bytemuck::cast_slice(collection);
        if elements.len() <= SLAB_ELEMENT_COUNT {
            let slab_reads = self.slabs.upgradable_read();
            // Find a slab where `try_bump_write` succeeds.
            if let Some((slab_id, start)) = slab_reads
                .iter()
                .enumerate()
                .find_map(|(idx, slab)| Some((idx, slab.shared_bump_write(elements)?)))
            {
                // We don't need this lock anymore!
                drop(slab_reads);

                // populate info
                let info = PointCollectionAllocInfo {
                    summary: collection.into(),
                    slab_id,
                    start,
                };
                // generate a new id and write metadata
                let id = PointCollectionID::default();
                self.allocs.write().insert(id, info);
                Some(id)
            } else {
                // No slabs were found with space to bump. Make a new one
                let new_slab = ElementSlab::new();
                // Unwrap is infallible - we checked the size requirement, so there's certainly room!
                let start = new_slab.shared_bump_write(elements).unwrap();
                // put the slab into self, getting it's index
                let slab_id = {
                    let mut write = parking_lot::RwLockUpgradableReadGuard::upgrade(slab_reads);
                    write.push(new_slab);
                    write.len() - 1
                };
                // populate info
                let info = PointCollectionAllocInfo {
                    summary: collection.into(),
                    slab_id,
                    start,
                };
                // generate a new id and write metadata
                let id = PointCollectionID::default();
                self.allocs.write().insert(id, info);
                Some(id)
            }
        } else {
            None
        }
    }

    /// Get a [`CollectionSummary`] for the given collection, reporting certain key aspects of a stroke without
    /// it needing to be loaded into resident memory. None if the ID is not known
    /// to this repository.
    pub fn summary_of(&self, id: PointCollectionID) -> Option<CollectionSummary> {
        self.alloc_of(id).map(|alloc| alloc.summary)
    }
    fn alloc_of(&self, id: PointCollectionID) -> Option<PointCollectionAllocInfo> {
        self.allocs.read().get(&id).copied()
    }
    pub fn try_get(
        &self,
        id: PointCollectionID,
    ) -> Result<PointCollectionReadLock, super::TryRepositoryError> {
        let alloc = self
            .alloc_of(id)
            .ok_or(super::TryRepositoryError::NotFound)?;
        let slabs_read = self.slabs.read();
        let Some(slab) = slabs_read.get(alloc.slab_id) else {
            // Implementation bug!
            log::debug!("{id} allocation found, but slab doesn't exist!");
            return Err(super::TryRepositoryError::NotFound);
        };
        // Check the alloc range is reasonable
        assert!(alloc
            .summary
            .len
            .checked_mul(alloc.summary.archetype.elements())
            .and_then(|elem_len| elem_len.checked_add(alloc.start))
            .is_some_and(|last| last <= SLAB_ELEMENT_COUNT));

        let Some(slice) = slab.try_read(
            alloc.start,
            // won't overflow, already checked!
            alloc.summary.len * alloc.summary.archetype.elements(),
        ) else {
            // Implementation bug!
            log::debug!("{id} allocation found, but out of bounds within it's slab!");
            return Err(super::TryRepositoryError::NotFound);
        };
        Ok(PointCollectionReadLock {
            points: bytemuck::cast_slice(slice),
        })
    }
}

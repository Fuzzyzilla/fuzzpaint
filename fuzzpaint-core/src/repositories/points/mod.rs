//! # Points
//!
//! Points have the largest size footprint of all resources, due to how numerous they are.
//! Thus, it makes sense that their repository implementation should recieve the most care.
//! For now, the collection just grows unboundedly and no eviction is done -
//! however, the API is constructed to allow for smart in-memory compression or dumping old
//! data to disk in the future.

pub mod io;
mod slab;
use slab::Slab;

use crate::stroke::{Archetype, StrokeSlice};

fn summarize(stroke: StrokeSlice) -> CollectionSummary {
    // Funny `try`
    // Calc arc length by observing arc length at end minus start.
    let arc_length = || -> Option<f32> {
        let last = stroke.last()?.arc_length()?;
        // Unwraps ok since first succeeded
        Some(last - stroke.first().unwrap().arc_length().unwrap())
    }();

    CollectionSummary {
        archetype: stroke.archetype(),
        len: stroke.len(),
        arc_length,
    }
}

#[derive(Copy, Clone)]
pub struct CollectionSummary {
    /// The archetype of the points of the collection.
    pub archetype: Archetype,
    /// Count of points within the collection
    pub len: usize,
    /// final arc length of the collected points, available if the archetype includes Archetype::ARC_LENGTH bit.
    pub arc_length: Option<f32>,
}
impl CollectionSummary {
    /// Gets the number of elements represented by this summary.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.len * self.archetype.elements()
    }
}

pub struct PointCollectionIDMarker;
pub type PointCollectionID = crate::FuzzID<PointCollectionIDMarker>;

/// A handle for reading a collection of points. Can be cloned and shared between threads,
/// however take care not to allow it to become leaked - it will not allow the resources
/// to be reclaimed by the repository for the duration of the lock's lifetime.
#[derive(Clone)]
pub struct BorrowedStrokeReadLock {
    stroke: StrokeSlice<'static>,
}
impl BorrowedStrokeReadLock {
    // we want to seal the fact that this is 'static. Can't be done with deref!
    #[must_use]
    pub fn get<'a>(&'a self) -> StrokeSlice<'a> {
        self.stroke
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
type ElementSlab = slab::Slab<u32, SLAB_ELEMENT_COUNT>;

#[derive(Default)]
pub struct Points {
    slabs: parking_lot::RwLock<Vec<ElementSlab>>,
    allocs: parking_lot::RwLock<hashbrown::HashMap<PointCollectionID, PointCollectionAllocInfo>>,
}
impl Points {
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
    pub fn insert(&self, collection: StrokeSlice) -> Option<PointCollectionID> {
        let elements = collection.elements();
        if elements.len() > SLAB_ELEMENT_COUNT {
            // Too long to ever fit!
            return None;
        }

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
                summary: summarize(collection),
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
                summary: summarize(collection),
                slab_id,
                start,
            };
            // generate a new id and write metadata
            let id = PointCollectionID::default();
            self.allocs.write().insert(id, info);
            Some(id)
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
    ) -> Result<BorrowedStrokeReadLock, super::TryRepositoryError> {
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
        Ok(BorrowedStrokeReadLock {
            stroke: StrokeSlice::new(slice, alloc.summary.archetype).unwrap(),
        })
    }
}

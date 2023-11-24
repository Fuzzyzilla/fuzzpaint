//! # Points
//!
//! Points have the largest size footprint of all resources, due to how numerous they are.
//! Thus, it makes sense that their repository implementation should recieve the most care.
//! For now, the collection just grows unboundedly and no eviction is done -
//! however, the API is constructed to allow for smart in-memory compression or dumping old
//! data to disk in the future.

use crate::FuzzID;

/// Get the shared global instance of the point repository.
pub fn global() -> &'static PointRepository {
    static REPO: std::sync::OnceLock<PointRepository> = std::sync::OnceLock::new();
    REPO.get_or_init(PointRepository::new)
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Eq, PartialEq, Hash, bytemuck::Pod, bytemuck::Zeroable, Debug)]
    /// Description of a point's data fields. Organized such that devices that have later flags are
    /// also likely to have prior flags.
    ///
    /// Note that position nor arc_length are required fields. Arc_length is derived data,
    /// and position may be ignored for strokes which trace a predefined path.
    ///
    /// Selected (loosely) from 2D-drawing-relavent packets defined in the windows ink API:
    /// https://learn.microsoft.com/en-us/windows/win32/tablet/packetpropertyguids-constants
    #[rustfmt::skip]
    #[repr(transparent)]
    pub struct PointArchetype : u8 {
        /// The point stream reports an (X: f32, Y: f32) position.
        const POSITION =   0b0000_0001;
        /// The point stream reports an f32 timestamp, in seconds from an arbitrary start moment.
        const TIME =       0b0000_0010;
        /// The point stream reports an f32, representing the cumulative length of the path from the start.
        const ARC_LENGTH = 0b0000_0100;
        /// The point stream reports a normalized, non-saturated pressure value.
        const PRESSURE =   0b0000_1000;
        /// The point stream reports a signed noramlized (X: f32, Y: f32) tilt, where positive X is to the right,
        /// positive Y is towards the user.
        const TILT =       0b0001_0000;
        /// Alias for bits that contain two fields
        const TWO_FIELDS = 0b0001_0001;
        /// The point stream reports a normalized f32 distance, in arbitrary units.
        const DISTANCE =   0b0010_0000;
        /// The point stream reports stylus roll (rotation along it's axis). Units and sign unknown!
        ///
        /// FIXME: Someone with such hardware, please let me know what it's units are :3
        const ROLL =       0b0100_0000;
        /// The point stream reports wheel values in signed, unnormalized, non-wrapping degrees, f32.
        ///
        /// Wheels are a general-purpose value which the user can use in their brushes. It may
        /// correspond to a physical wheel on the pen or pad, a touch slider, ect. which may be interacted
        /// with during the stroke for expressive effects.
        const WHEEL =      0b1000_0000;
    }
}
impl PointArchetype {
    /// How many elements (f32) does a point of this archetype occupy?
    pub fn len(self) -> usize {
        // Formerly Self::iter based but the codegen was un-scrumptious

        // Every field specifies one element, count them all
        self.bits().count_ones() as usize
        // These fields specify two elements, count them again
            + (self & Self::TWO_FIELDS).bits().count_ones() as usize
    }
    pub fn len_bytes(self) -> usize {
        self.len() * std::mem::size_of::<f32>()
    }
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

impl From<&[crate::StrokePoint]> for CollectionSummary {
    fn from(value: &[crate::StrokePoint]) -> Self {
        CollectionSummary {
            archetype: PointArchetype::POSITION
                | PointArchetype::ARC_LENGTH
                | PointArchetype::PRESSURE,
            len: value.len(),
            arc_length: Some(value.last().map(|point| point.dist).unwrap_or(0.0)),
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
    /// What point index into that slab does it start?
    start: usize,
    /// A summary of the data within, that can be queried even if the bulk
    /// data is non-resident.
    summary: CollectionSummary,
}
pub struct PointRepository {
    slabs: parking_lot::RwLock<Vec<PointSlab>>,
    allocs: parking_lot::RwLock<hashbrown::HashMap<PointCollectionID, PointCollectionAllocInfo>>,
}
impl PointRepository {
    fn new() -> Self {
        // Self doesn't impl Default as we don't want any ctors to be public.
        Self {
            slabs: Default::default(),
            allocs: Default::default(),
        }
    }
    /// Get the memory usage of resident data (uncompressed in RAM), in bytes, and the capacity.
    pub fn resident_usage(&self) -> (usize, usize) {
        let read = self.slabs.read();
        let num_slabs = read.len();
        let capacity = num_slabs.saturating_mul(SLAB_SIZE_BYTES);
        let usage = read
            .iter()
            .map(|slab| slab.usage())
            .fold(0, usize::saturating_add)
            .saturating_mul(std::mem::size_of::<crate::StrokePoint>());
        (usage, capacity)
    }
    /// Insert the collection into the repository, yielding a unique ID.
    /// Fails if the length of the collection is > 0x10_00_00
    pub fn insert(&self, collection: &[crate::StrokePoint]) -> Option<PointCollectionID> {
        if collection.len() <= SLAB_SIZE_POINTS {
            // Find a slab where `try_bump` succeeds.
            let slab_reads = self.slabs.upgradable_read();
            if let Some((slab_id, start)) = slab_reads
                .iter()
                .enumerate()
                .find_map(|(idx, slab)| Some((idx, slab.try_bump_write(collection)?)))
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
                let new_slab = PointSlab::new();
                // Unwrap is infallible - we checked the size requirement, so there's certainly room!
                let start = new_slab.try_bump_write(collection).unwrap();
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
    /// Given an iterator of collection IDs, encodes them directly (in order) into the given Write stream in a `DICT ptls` chunk.
    /// On success, returns a map between PointCollectionID and file local id as written.
    pub fn write_dict_into(
        &self,
        ids: impl Iterator<Item = PointCollectionID>,
        writer: impl std::io::Write,
    ) -> Result<crate::io::id::FileLocalInterner<PointCollectionIDMarker>, WriteError> {
        use crate::io::{
            riff::{encode::SizedBinaryChunkWriter, ChunkID},
            OrphanMode, Version,
        };
        use az::CheckedAs;
        use std::io::{IoSlice, Write};

        const PTLS_WRITE_VERSION: Version = Version(0, 0, 0);

        let mut file_ids = crate::io::id::FileLocalInterner::new();
        // Collect all uniqe entries and allocs.
        let allocation_entries: Result<Vec<_>, WriteError> = ids
            .filter_map(|id| match file_ids.insert(id) {
                // New entry, collect it's alloc or short-circuit if not found
                Ok(true) => Some(self.alloc_of(id).ok_or(WriteError::UnknownID(id))),
                // Already collected
                Ok(false) => None,
                // Short circuit collection on err
                Err(crate::io::id::InternError::TooManyEntries) => {
                    Some(Err(WriteError::TooManyEntries))
                }
            })
            .collect();
        let allocation_entries = allocation_entries?;

        #[derive(Clone, Copy, bytemuck::NoUninit)]
        #[repr(C, packed)]
        struct DictMetadata {
            offset: u32,
            len: u32,
            arch: PointArchetype,
        }

        let mut total_data_len = 0u32;
        let meta_entries: Result<Vec<DictMetadata>, WriteError> = allocation_entries
            .iter()
            .map(|alloc| {
                let summary = alloc.summary;
                // Len in bytes must fit in u32
                let len = summary
                    .len
                    .checked_mul(summary.archetype.len_bytes())
                    .and_then(|len| len.checked_as())
                    .ok_or(WriteError::TooLong)?;
                let meta = DictMetadata {
                    offset: total_data_len,
                    len,
                    arch: summary.archetype,
                };

                // Data length must not overrun u32
                total_data_len = total_data_len.checked_add(len).ok_or(WriteError::TooLong)?;
                Ok(meta)
            })
            .collect();
        let meta_entries = meta_entries?;
        let num_meta_entries: u32 = meta_entries
            .len()
            .checked_as()
            .ok_or(WriteError::TooManyEntries)?;
        let meta_size: u32 = std::mem::size_of::<DictMetadata>()
            .checked_as()
            .ok_or(WriteError::TooLong)?;

        // Num metas times meta size
        let chunk_size = num_meta_entries
            .checked_mul(meta_size)
            // Header size
            .and_then(|total| total.checked_add(12))
            // Unstructure data size
            .and_then(|total| total.checked_add(total_data_len))
            .ok_or(WriteError::TooLong)?;

        // Allocate the chunk.
        // I should make a helper for DICT...
        let mut chunk = SizedBinaryChunkWriter::new_subtype(
            writer,
            ChunkID::DICT,
            ChunkID::PTLS,
            chunk_size as usize,
        )?;
        // Write header, metas
        {
            let meta_info = [num_meta_entries, meta_size];
            let mut header_and_meta = [
                IoSlice::new(bytemuck::bytes_of(&PTLS_WRITE_VERSION)),
                IoSlice::new(&[OrphanMode::Deny as u8]),
                IoSlice::new(bytemuck::cast_slice(&meta_info)),
                IoSlice::new(bytemuck::cast_slice(&meta_entries)),
            ];
            chunk.write_all_vectored(&mut header_and_meta)?;
        };

        // TODO: native -> little endian conversion.
        // Expensive to do! Would be cheaper if we know we're about to consume and invalidate the lists,
        // as we could convert in-place.
        #[cfg(not(target_endian = "little"))]
        compile_error!("FIXME!");

        // Collect and write bulk points
        let data_slices: Result<Vec<IoSlice<'_>>, ()> = {
            let slabs = self.slabs.read();
            allocation_entries
                .iter()
                .map(|entry| {
                    let Some(slab) = slabs.get(entry.slab_id) else {
                        // Implementation bug!
                        return Err(());
                    };
                    // Check the alloc range is reasonable
                    debug_assert!(entry
                        .start
                        .checked_add(entry.summary.len)
                        .is_some_and(|last| last <= SLAB_SIZE_POINTS));

                    let Some(slice) = slab.try_read(entry.start, entry.summary.len) else {
                        // Implementation bug!
                        return Err(());
                    };
                    Ok(IoSlice::new(bytemuck::cast_slice(slice)))
                })
                .collect()
        };
        let mut data_slices = data_slices.map_err(|_| {
            WriteError::IOError(std::io::Error::other(anyhow::anyhow!("internal error :(")))
        })?;
        chunk.write_all_vectored(&mut data_slices)?;
        // Pad, if needed (shouldn't be)
        chunk.pad_slow()?;

        Ok(file_ids)
    }
    /// Intern all the data from the given `DICT ptls`, returning a map of the newly allocated
    /// IDs.
    pub fn read_dict<R>(
        &self,
        mut dict: crate::io::riff::decode::DictReader<R>,
    ) -> std::io::Result<crate::io::id::ProcessLocalInterner<PointCollectionIDMarker>>
    where
        R: std::io::Read + std::io::Seek,
    {
        use crate::io::{id::ProcessLocalInterner, Version};
        use az::CheckedAs;
        use std::io::{Error as IOError, Read};
        if dict.version() != Version::CURRENT {
            // TODO lol
            return Err(IOError::other(anyhow::anyhow!("bad ver")));
        }
        // There's metas, but they're not the right size.
        // (this allows arbitrary size when there are zero entries - this is fine)
        if dict
            .meta_len_unsanitized()
            .is_some_and(|val| val.get() != std::mem::size_of::<DictMetadata>())
        {
            return Err(IOError::other(anyhow::anyhow!("bad metadata len")));
        }

        #[derive(Clone, Copy, bytemuck::AnyBitPattern, Debug)]
        #[repr(C, packed)]
        struct DictMetadata {
            offset: u32,
            len: u32,
            arch: PointArchetype,
        }
        let mut metas = Vec::<(Option<PointCollectionID>, DictMetadata)>::new();
        let mut unstructured = dict.try_for_each(|mut meta_read| {
            let mut bytes = [0; std::mem::size_of::<DictMetadata>()];
            meta_read.read_exact(&mut bytes)?;
            let meta: DictMetadata = bytemuck::pod_read_unaligned(&bytes);
            metas.push((None, meta));

            Ok(())
        })?;
        let reported_len = unstructured.data_len_unsanitized();
        // Make sure none surpass the end of the data chunk
        // AND make sure none surpass the limit of allocatable points, `SLAB_SIZE`
        if !metas.iter().all(|m| {
            m.1.len
                .checked_add(m.1.offset)
                .is_some_and(|end| end <= reported_len as u32)
                && m.1.len as usize <= SLAB_SIZE_BYTES
        }) {
            return Err(IOError::other(anyhow::anyhow!("point list data too long")));
        }
        // Allocate many ids, and disperse them
        let ids = {
            let count = metas
                .len()
                .checked_as::<u32>()
                .ok_or_else(|| IOError::other(anyhow::anyhow!("Too many elements")))?;
            let mut ids = ProcessLocalInterner::many_sequential(count as usize).unwrap();
            for (file_id, meta) in (0..count).into_iter().zip(metas.iter_mut()) {
                meta.0 = Some(ids.get_or_insert(file_id.into()))
            }
            ids
        };
        // Sort pointlists by start position
        metas.sort_unstable_by_key(|meta| meta.1.offset);

        // Strategy: because we cannot trust the length of `unstructured` nor the reported size of the metas
        // we cannot simply read all the data blindly. Instead:
        // * Pop the first meta
        // * Find an existing slab that'll fit it, or allocate if None (limit 16MiB)
        // * Pop all remaining metas that can *also* fit into that chunk
        // * Bulk read.
        // * Repeat until all metas are loaded.
        // * If we allocated and it failed, we can de-allocate.
        // Limitations: Can't yet de-allocate from existing slabs so failures leak mem,
        // + concurrent loading will over-commit on new blocks.

        // Blocks that were newly allocated for reading. May be freed if an error occurs.
        let mut new_blocks = smallvec::SmallVec::<[PointSlab; 2]>::new();
        // We can trust metas.len, as we were successfully able to read that many.
        let mut infos = Vec::<PointCollectionAllocInfo>::with_capacity(metas.len());

        let mut try_read_points = || -> Result<(), IOError> {
            // Get next - or finished!
            let Some(next) = metas.pop() else {
                return Ok(());
            };
            // Find a block that fits it
            let blocks = self.slabs.read();
            todo!();

            // Collect all subsequent ones that will also fit
            // (allows overlapping blocks) (todo: alignment issues?)

            // Read into

            // Create infos

            Ok(())
        };

        // Failed to read. Free any blocks we allocated for this task.
        if let Err(e) = try_read_points() {
            for block in new_blocks.into_iter() {
                // Safety - we took no references to this data.
                unsafe { block.free() }
            }
            return Err(e);
        }

        Ok(ids)
    }
    /// Get a [CollectionSummary] for the given collection, reporting certain key aspects of a stroke without
    /// it needing to be loaded into resident memory. None if the ID is not known
    /// to this repository.
    pub fn summary_of(&self, id: PointCollectionID) -> Option<CollectionSummary> {
        self.alloc_of(id).map(|alloc| alloc.summary)
    }
    fn alloc_of(&self, id: PointCollectionID) -> Option<PointCollectionAllocInfo> {
        self.allocs.read().get(&id).cloned()
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
        debug_assert!(alloc
            .start
            .checked_add(alloc.summary.len)
            .is_some_and(|last| last <= SLAB_SIZE_POINTS));

        let Some(slice) = slab.try_read(alloc.start, alloc.summary.len) else {
            // Implementation bug!
            log::debug!("{id} allocation found, but out of bounds within it's slab!");
            return Err(super::TryRepositoryError::NotFound);
        };
        Ok(PointCollectionReadLock { points: slice })
    }
}
// A large collection of continguous points on the heap
struct PointSlab {
    /// a non-null pointer to array of slab_SIZE points.
    points: *mut crate::StrokePoint,
    /// Write access guard.
    write_access: parking_lot::Mutex<()>,
    /// Current past-the-end index for the allocator.
    /// Indices < this are considered immutable, >= considered mutable.
    ///
    /// ***It is a logic error to write to this without holding a lock!***
    bump_position: std::sync::atomic::AtomicUsize,
}
const SLAB_SIZE_POINTS: usize = 1024 * 1024;
const SLAB_SIZE_BYTES: usize = SLAB_SIZE_POINTS * std::mem::size_of::<crate::StrokePoint>();

#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error("bump exceeds capacity")]
struct BumpTooLargeError;
struct SlabGuard<'a> {
    /// a non-null pointer to array of slab_SIZE points.
    points: *mut crate::StrokePoint,
    _write_access_lock: parking_lot::MutexGuard<'a, ()>,
    bump_position: &'a std::sync::atomic::AtomicUsize,
}
impl<'a> SlabGuard<'a> {
    /// How many more points can fit?
    fn remaining(&self) -> usize {
        SLAB_SIZE_POINTS.saturating_sub(self.position())
    }
    /// Get a mutable reference to the unfilled portion of the slab.
    /// Be sure to mark any consumed space as used with `self::bump`!
    fn unfilled<'s>(&'s mut self) -> &'s mut [crate::StrokePoint] {
        let position = self.position();
        let remaining = SLAB_SIZE_POINTS.saturating_sub(position);
        // Invariant should be upheld by everone else.
        assert!(position <= SLAB_SIZE_POINTS);
        debug_assert!(SLAB_SIZE_POINTS != 0); // -w- completeness

        // Safety - must remain inside the alloc'd object
        // We checked with assert!
        let ptr_start = unsafe { self.points.add(position) };

        // Safety - self has exclusive access to the Slab's writable portion.
        // We then hold self mutably while accessing.
        unsafe { std::slice::from_raw_parts_mut::<'s, crate::StrokePoint>(ptr_start, remaining) }
    }
    /// Bump the inner slab by this number of points. The points become frozen
    /// and cannot be modified if this call returns Ok().
    ///
    /// Returns Err and makes no changes if `num_points > remaining`.
    fn bump(&mut self, num_points: usize) -> Result<(), BumpTooLargeError> {
        // We perform a relaxed load here. This is fine, no races can occur - invariant is
        // that we have exclusive store access to this atomic.
        let position = self.position();
        if position
            .checked_add(num_points)
            .is_some_and(|end| end <= SLAB_SIZE_POINTS)
        {
            // Release - we must finish all prior ops (could be writes!) before the store occurs.
            self.bump_position
                .store(position + num_points, std::sync::atomic::Ordering::Release);
            Ok(())
        } else {
            Err(BumpTooLargeError)
        }
    }
    /// Returns the position of the bump index.
    /// This is the first index in the slab that is available for mutation, everything
    /// before this index is frozen and immutable.
    fn position(&self) -> usize {
        // Relaxed is fine. This thread is the only writer.
        self.bump_position
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
impl PointSlab {
    /// Lock the slab for exclusive low-level writing access.
    ///
    /// *Reads are still free to occur to any point before the bump index even while this lock is held.*
    fn lock<'a>(&'a self) -> SlabGuard<'a> {
        SlabGuard {
            points: self.points,
            bump_position: &self.bump_position,
            _write_access_lock: self.write_access.lock(),
        }
    }
    /// Try to allocate and write a contiguous section of points, returning the start idx of where it was written.
    ///
    /// If `Some(idx)` is returned, the data can be retrieved via `self::try_read(idx, data.len())`
    ///
    /// If not enough space, the `self` is left unchanged and None is returned.
    fn try_bump_write(&self, data: &[crate::StrokePoint]) -> Option<usize> {
        if data.len() <= SLAB_SIZE_POINTS {
            // Eager check for space before waiting on the lock.
            // Could still fail afterwards!
            if data.len()
                > SLAB_SIZE_POINTS.saturating_sub(
                    self.bump_position
                        .load(std::sync::atomic::Ordering::Relaxed),
                )
            {
                return None;
            }
            // *might* fit. Try to lock and write.
            let mut lock = self.lock();
            let start = lock.position();

            let unfilled = lock.unfilled();
            // Not enough space
            if unfilled.len() < data.len() {
                return None;
            }
            // Copy data into start region of unfilled range
            // Indexing ok - we checked the precondition manually.
            unfilled[..data.len()].copy_from_slice(data);
            // Bump the new data into immutable range
            // Unwrap ok - we checked the precondition manually.
            lock.bump(data.len()).unwrap();

            Some(start)
        } else {
            None
        }
    }
    /// Try to read some continuous section of strokes. returns None if the region is outside the span
    /// of the currently allocated memory.
    ///
    /// Performs no check that the given start and length correspond to a single suballocation!
    fn try_read(&self, start: usize, len: usize) -> Option<&'static [crate::StrokePoint]> {
        // Check if this whole region is within the allocated, read-only section.
        if start
            .checked_add(len)
            // Check it is within the readable range.
            // Acquire, since operations after this rely on the mem guarded by this load.
            .is_some_and(|past_end| {
                past_end
                    <= self
                        .bump_position
                        .load(std::sync::atomic::Ordering::Acquire)
            })
        {
            // Safety: no shared mutable access, as mutation never happens before the bump idx
            Some(unsafe { std::slice::from_raw_parts(self.points.add(start), len) })
        } else {
            None
        }
    }
    /// Get the number of points currently in use.
    /// This is a hint, and not intended to be used for safety!
    fn usage(&self) -> usize {
        self.bump_position
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Allocate a new slab of SLAB_SIZE points.
    ///
    /// `Self::try_new`, except terminates on allocation failure.
    fn new() -> Self {
        match Self::try_new() {
            Some(s) => s,
            None => std::alloc::handle_alloc_error(Self::layout()),
        }
    }
    /// Allocate a new slab of SLAB_SIZE points.
    ///
    /// Returns None if the allocation failed. To fail on this condition,
    /// prefer [std::alloc::handle_alloc_error] over a panic.
    ///
    /// There is no guaruntee that this won't terminate the process instead of returning None.
    fn try_new() -> Option<Self> {
        let layout = Self::layout();
        // (is there a better way to get a large, arbitrarily-initialized heap array?)
        let points = unsafe { std::alloc::alloc_zeroed(layout).cast::<crate::StrokePoint>() };

        if points.is_null() {
            None
        } else {
            Some(Self {
                points,
                write_access: Default::default(),
                bump_position: 0.into(),
            })
        }
    }
    /// Free the memory of this slab. By default, memory is leaked on drop as the references to this slab's
    /// data live arbitrarily long.
    ///
    /// Safety: There must not be any outstanding references to this slab's memory (acquired by `try_read`).
    unsafe fn free(self) {
        // Safety - using same layout as used to create it.
        // Use-after-free forwarded to this fn's safety contract.
        unsafe { std::alloc::dealloc(self.points as *mut _, Self::layout()) }
    }
    fn layout() -> std::alloc::Layout {
        const SIZE: usize = SLAB_SIZE_BYTES;
        const ALIGN: usize = std::mem::align_of::<crate::StrokePoint>();
        debug_assert!(SIZE != 0);
        debug_assert!(ALIGN != 0 && ALIGN.is_power_of_two());
        debug_assert!(SIZE % ALIGN == 0);

        // Unwrap - if this fails, it'll be *VERY* obvious lol
        // Does not depend on runtime info, so it'll always fail or succeed on the first allocation attempt.
        std::alloc::Layout::from_size_align(SIZE, ALIGN).unwrap()
    }
}
// Safety - the pointer refers to heap mem, and can be transferred.
unsafe impl Send for PointSlab {}

// Safety - The mutex prevents similtaneous mutable and immutable access.
unsafe impl Sync for PointSlab {}

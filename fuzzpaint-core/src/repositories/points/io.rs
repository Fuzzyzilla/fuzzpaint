use std::collections::VecDeque;

// More of an #include situation than a module situation lol
#[allow(clippy::wildcard_imports)]
use super::*;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, packed)]
struct DictMetadata {
    offset: u32,
    len: u32,
    arch: PointArchetype,
}

// Collect all subsequent ones that will also fit
// (allows overlapping blocks)

/// Collect a slice while `pred` returns true.
///
/// Inverse of the unstable `[T]::split_once` but it's unstable anyway.
/// Similar to `group_by`, `split`, `split_once` ect...... hmmst
fn take_while<T, Pred>(slice: &[T], mut pred: Pred) -> &[T]
where
    Pred: FnMut(&T) -> bool,
{
    if let Some(pos) = slice.iter().position(|t| !pred(t)) {
        &slice[..pos]
    } else {
        slice
    }
}

/// For a slab that's either locked or owned
enum SlabSrc<'a, T: bytemuck::Pod, const N: usize> {
    Shared {
        idx: usize,
        lock: slab::Guard<'a, T, N>,
    },
    Owned(slab::Slab<T, N>),
}
impl<'a, T: bytemuck::Pod, const N: usize> SlabSrc<'a, T, N> {
    fn position(&mut self) -> usize {
        match self {
            Self::Shared { lock, .. } => lock.position(),
            Self::Owned(slab) => slab.position(),
        }
    }
    fn remaining(&mut self) -> usize {
        match self {
            Self::Shared { lock, .. } => lock.remaining(),
            Self::Owned(slab) => slab.remaining(),
        }
    }
    fn parts_mut(&mut self) -> (&[T], &mut [T]) {
        match self {
            Self::Shared { lock, .. } => lock.parts_mut(),
            Self::Owned(slab) => slab.parts_mut(),
        }
    }
    fn bump(&mut self, amount: usize) -> Result<(), slab::BumpTooLargeError> {
        match self {
            Self::Shared { lock, .. } => lock.bump(amount),
            Self::Owned(slab) => slab.bump(amount),
        }
    }
}

#[derive(Copy, Clone)]
enum LazyID {
    /// ID refers to a concrete idx in the existing shared slabs.
    Shared(usize),
    /// ID refers to this index in the local slab stack, basis must be shifted at upload time.
    Local(usize),
}
#[derive(Copy, Clone)]
struct LazyPointCollectionAllocInfo {
    /// Which PointSlab is it in?
    /// (currently an index)
    slab_id: LazyID,
    /// What *element* index into that slab does it start?
    ///
    /// Note that summary.len is in units of points, not elements.
    start: usize,
    /// A summary of the data within, that can be queried even if the bulk
    /// data is non-resident.
    summary: CollectionSummary,
}

impl super::Points {
    /// Given an iterator of collection IDs, encodes them directly (in order) into the given Write stream in a `DICT ptls` chunk.
    /// On success, returns a map between `PointCollectionID` and file local id as written.
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

        let mut file_ids = crate::io::id::FileLocalInterner::default();
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

        let mut total_data_bytes = 0u32;
        let meta_entries: Result<Vec<DictMetadata>, WriteError> = allocation_entries
            .iter()
            .map(|alloc| {
                let summary = alloc.summary;
                // Len in bytes must fit in u32
                let len = summary
                    .len
                    .checked_mul(summary.archetype.len_bytes())
                    .and_then(usize::checked_as)
                    .ok_or(WriteError::TooLong)?;
                let meta = DictMetadata {
                    offset: total_data_bytes,
                    len,
                    arch: summary.archetype,
                };

                // Data length must not overrun u32
                total_data_bytes = total_data_bytes
                    .checked_add(len)
                    .ok_or(WriteError::TooLong)?;
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
            .and_then(|total| total.checked_add(total_data_bytes))
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

                    let Some(slice) = slab.try_read(
                        entry.start,
                        entry.summary.len * entry.summary.archetype.elements(),
                    ) else {
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
        dict: crate::io::riff::decode::DictReader<R>,
    ) -> std::io::Result<crate::io::id::ProcessLocalInterner<PointCollectionIDMarker>>
    where
        R: std::io::Read + crate::io::common::SoftSeek,
    {
        use crate::io::{common::SoftSeek, id::ProcessLocalInterner, Version};
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

        let mut metas = VecDeque::<(Option<PointCollectionID>, DictMetadata)>::new();
        let mut unstructured = dict.try_for_each(|mut meta_read| {
            let mut bytes = [0; std::mem::size_of::<DictMetadata>()];
            meta_read.read_exact(&mut bytes)?;
            let meta: DictMetadata = bytemuck::pod_read_unaligned(&bytes);
            metas.push_back((None, meta));

            Ok(())
        })?;
        let reported_len = unstructured.data_len_unsanitized();
        // Make sure none surpass the end of the data chunk
        // AND make sure none surpass the limit of allocatable points, `SLAB_SIZE`
        if !metas.iter().all(|m| {
            m.1.len
                .checked_add(m.1.offset)
                .is_some_and(|end| end <= reported_len as u32)
                // Check small enough to even fit in a slab
                && m.1.len as usize <= SLAB_ELEMENT_COUNT * std::mem::size_of::<f32>()
                // Check is StrokePoint (relax this when other types available)
                && m.1.len as usize % std::mem::size_of::<crate::stroke::Point>() == 0
        }) {
            return Err(IOError::other(anyhow::anyhow!("point list data too long")));
        }
        // Allocate many ids, and disperse them
        let file_ids = {
            let count = metas
                .len()
                .checked_as::<u32>()
                .ok_or_else(|| IOError::other(anyhow::anyhow!("Too many elements")))?;
            let mut ids = ProcessLocalInterner::many_sequential(count as usize).unwrap();
            for (file_id, meta) in (0..count).zip(metas.iter_mut()) {
                meta.0 = Some(ids.get_or_insert(file_id.into()));
            }
            ids
        };
        // Sort pointlists by start position
        metas
            .make_contiguous()
            .sort_unstable_by_key(|meta| meta.1.offset);

        // Strategy: because we cannot trust the length of `unstructured` nor the reported size of the metas
        // we cannot simply read all the data blindly. Instead:
        //   * Pop the first meta
        //   * Find an existing slab that'll fit it, or allocate if None (limit 16MiB)
        //   * Pop all remaining metas that can *also* fit into that chunk
        //   * Bulk read.
        //  * Repeat until all metas are loaded.
        // * If we allocated and it failed, we can de-allocate.
        // Limitations: Can't yet de-allocate from existing slabs so failures leak mem,
        // + concurrent loading will over-commit on new blocks.

        // Blocks that were newly allocated for reading. May be freed if an error occurs.
        let mut new_slabs: smallvec::SmallVec<[ElementSlab; 2]> = smallvec::SmallVec::new();
        // We can trust the length of metas now, since we were successfully able to read that many.
        let mut allocs =
            Vec::<(PointCollectionID, LazyPointCollectionAllocInfo)>::with_capacity(metas.len());

        // This is an absolute disaster, readability and perf wise.
        // Any attempt to simplify it results in inscrutable lifetime errors D:
        // Will revisit after i've gone and done something else for a while to refresh my
        // withered soul

        // Todos: existing new slabs are not considered when searching for a slab to write in
        //   resulting in  a bunch of extra slabs being made
        let mut try_read_points = || -> Result<(), IOError> {
            while let Some(first_meta) = metas.pop_front() {
                // Find a block that fits it
                let slabs = self.slabs.read();
                let mut slab = {
                    let slab_info = slabs.iter().enumerate().find_map(|(idx, slab)| {
                        // Check if it *might* fit (can still fail)
                        // bytes -> elements
                        if slab.hint_remaining() >= first_meta.1.len as usize / 4 {
                            let lock = slab.lock();
                            // Check if it actually fits
                            // bytes -> elements
                            if lock.remaining() >= first_meta.1.len as usize / 4 {
                                Some((idx, lock))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });
                    // If no block has enough space, make a new one and lock that one instead.
                    match slab_info {
                        Some((idx, lock)) => SlabSrc::Shared { idx, lock },
                        None => SlabSrc::Owned(Slab::new()),
                    }
                };

                // Immutable - they're sorted, so the start point never needs to move.
                let range_start_bytes = first_meta.1.offset;
                let mut range_past_end_bytes = range_start_bytes + first_meta.1.len;

                // Keep track of free space
                let mut remaining_elements = slab.remaining() - first_meta.1.len as usize / 4;
                // First element we're writing into
                let start_element = slab.position();

                // We know metas only has a first slice because we never push!
                let also_fit = take_while(metas.as_slices().0, |meta| {
                    // Discontiguous!
                    // (dont need to check < start, they're sorted!)
                    if meta.1.offset > range_past_end_bytes {
                        return false;
                    }
                    // Unaligned!
                    if (meta.1.offset - range_start_bytes) % 4 != 0 {
                        return false;
                    }
                    // Fits?
                    if meta.1.len as usize / 4 <= remaining_elements {
                        remaining_elements -= meta.1.len as usize / 4;
                        // Push forward, if needed
                        range_past_end_bytes = range_past_end_bytes.max(meta.1.offset + meta.1.len);
                        true
                    } else {
                        false
                    }
                });

                let count_bytes = (range_past_end_bytes - range_start_bytes) as usize;
                // Should be divisible exactly. We check that everything is aligned to fours.
                assert!(count_bytes % 4 == 0);
                let count_elements = count_bytes / 4;

                // Seek, if necessary
                // We only ever need to move forwards
                // In theory we shouldn't even need to seek, but we cannot assume that.
                let cur = unstructured.soft_position()?;
                let forward_dist = u64::from(range_start_bytes)
                    .checked_sub(cur)
                    // debug assert lol
                    .expect("seek back");
                // we don't care if underlying reader is seeked
                unstructured.soft_seek(forward_dist as i64)?;

                // Read!
                let (_, unfilled) = slab.parts_mut();
                unstructured
                    .read_exact(bytemuck::cast_slice_mut(&mut unfilled[..count_elements]))?;

                // todo: postprocess (endian swap + interleaving derived attribs)
                slab.bump(count_elements).unwrap();
                // Summarize (we could lower to read-only access on the slab but lifetimes are nightmare)
                let summaries: Vec<CollectionSummary> = std::iter::once(&first_meta)
                    .chain(also_fit.iter())
                    .map(|(_, meta)| {
                        let last_point = if meta.len == 0 {
                            None
                        } else {
                            let (immutable, _) = slab.parts_mut();
                            // Find where the first element is mapped to in the slab
                            let start_idx =
                                start_element + (meta.offset - range_start_bytes) as usize / 4;
                            // Find where the past-the-end point in the slab
                            let past_the_end = start_idx + meta.len as usize / 4;
                            // Step back one point
                            let last_point_start = past_the_end - meta.arch.elements();
                            // Make sure we didn't step back past the start
                            if last_point_start >= start_idx {
                                // Try to read elements+cast to point
                                let opt_last_point = immutable
                                    .get(last_point_start..last_point_start + meta.arch.elements())
                                    .and_then(|slice| {
                                        bytemuck::try_cast_slice::<_, crate::stroke::Point>(slice)
                                            .ok()
                                    })
                                    .and_then(|slice| slice.get(0));
                                opt_last_point
                            } else {
                                None
                            }
                        };
                        CollectionSummary {
                            archetype: meta.arch,
                            len: meta.len as usize / std::mem::size_of::<crate::stroke::Point>(),
                            // Option into Optional panics on none? silly dubious
                            // Some if available and non-nan, None otherwise.
                            arc_length: last_point.map(|l| l.dist),
                        }
                    })
                    .collect();

                // Finished with the slab. Now we need to write alloc infos, so collect the idx
                // of the slab we just wrote.
                let slab_id = match slab {
                    SlabSrc::Shared { idx, .. } => LazyID::Shared(idx),
                    SlabSrc::Owned(o) => {
                        new_slabs.push(o);
                        LazyID::Local(new_slabs.len() - 1)
                    }
                };

                // Create alloc infos
                std::iter::once(&first_meta)
                    .chain(also_fit.iter())
                    .zip(summaries.into_iter())
                    .for_each(|((id, meta), summary)| {
                        let alloc_info = LazyPointCollectionAllocInfo {
                            slab_id,
                            // Exact div. We checked they were aligned to fours!
                            start: start_element + (meta.offset - range_start_bytes) as usize / 4,
                            // Exact div. We checked length was OK in pre-sort.
                            summary,
                        };
                        // unwrap ok - we assigned ids earlier.
                        allocs.push((id.unwrap(), alloc_info));
                    });

                // Pop all that we handled this iteration (first was already popped)
                metas.drain(..also_fit.len());
            }

            // Fellthrough - we successfully read all points!
            Ok(())
        };

        // Failed to read. Free any blocks we allocated for this task and diverge.
        if let Err(e) = try_read_points() {
            for block in new_slabs {
                // Safety - we only took short-lived references to this data for
                // generating the summaries, and they've since been dropped.
                unsafe { block.free() }
            }
            return Err(e);
        }
        // Read success
        // We have slabs to share!
        if !new_slabs.is_empty() {
            let start_idx = {
                let mut write = self.slabs.write();
                let start_idx = write.len();
                write.extend(new_slabs);
                start_idx
            };
            // We now have a mapping of New -> Shared ids
            for alloc in &mut allocs {
                if let LazyID::Local(local) = alloc.1.slab_id {
                    alloc.1.slab_id = LazyID::Shared(local + start_idx);
                }
            }
        }
        // At this point ever alloc should be in Shared state.
        {
            let mut write = self.allocs.write();
            for (id, alloc) in allocs {
                let slab_id = match alloc.slab_id {
                    LazyID::Shared(id) => id,
                    // Impl error!
                    LazyID::Local(_) => unimplemented!(),
                };
                write.insert(
                    id,
                    PointCollectionAllocInfo {
                        slab_id,
                        start: alloc.start,
                        summary: alloc.summary,
                    },
                );
            }
        }
        // Report back the FileID->FuzzID mapping
        Ok(file_ids)
    }
}

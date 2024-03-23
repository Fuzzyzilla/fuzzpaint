use crate::vulkano_prelude::*;
use std::sync::Arc;

use fuzzpaint_core::{repositories::points, state::stroke_collection::ImmutableStroke};

pub enum SyncOutput<InnerFuture: vk::sync::GpuFuture> {
    /// No sync needed.
    Immediate,
    /// The `elements` and `residual` buffers are used for read/write access until this fence.
    Fence(vk::FenceSignalFuture<InnerFuture>),
}
pub struct StrokeAlloc {
    /// Offset into the `elements` buffer, in f32 elements
    pub offset: usize,
    /// Summary of the collection.
    pub summary: points::CollectionSummary,
    /// which stroke does this data come from?
    pub src: ImmutableStroke,
}
pub struct StrokeBatch {
    /// Filled portion of the buffer.
    ///
    /// All read/write usage must be finished by the time the `SyncOutput` is signaled.
    pub elements: vk::Subbuffer<[f32]>,
    /// Unfilled portion of the same buffer, for arbitrary use.
    /// Contents are undefined - guarunteed to be aligned and sized as `f32`, however.
    ///
    /// Beware of `NonCoherentAtomSize` - this is not pre-aligned to that!
    ///
    /// All read/write usage must be finished by the time the `SyncOutput` is signaled.
    pub residual: Option<vk::Subbuffer<[f32]>>,
    pub allocs: Vec<StrokeAlloc>,
}

#[derive(thiserror::Error, Debug)]
pub enum BatchError<Inner: std::fmt::Debug> {
    /// The inner closure reported an error.
    // This Debug trait bound is icky and pervasive when it really should be `Error`.
    // Unfortunately `anyhow::Error` cannot be `Error`, unsure of a better bound for this.
    #[error("{:?}", .0)]
    Inner(Inner),
    /// The staging buffer's access contract was broken.
    #[error(transparent)]
    AccessError(vulkano::sync::HostAccessError),
    /// Waiting on `SyncOutput::Fence` failed.
    #[error(transparent)]
    FenceError(vulkano::Validated<vulkano::VulkanError>),
}

/// A buffer wrapper that uploads and dispatches lists of strokes.
/// Eagerly allocates space for a staging buffer.
pub struct StrokeBatcher {
    buffer: Arc<vk::Buffer>,
}
impl StrokeBatcher {
    /// Create a new `StrokeDispatcher` with space to hold `capacity` f32 elements.
    pub fn new(
        allocator: Arc<dyn vulkano::memory::allocator::MemoryAllocator>,
        capacity: usize,
        usage: vk::BufferUsage,
        sharing: vk::Sharing<smallvec::SmallVec<[u32; 4]>>,
    ) -> anyhow::Result<Self> {
        if capacity == 0 {
            anyhow::bail!("cannot create zero-size buffer");
        }
        let buffer = vk::Buffer::new(
            allocator,
            vk::BufferCreateInfo {
                sharing,
                usage,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            // Unwrap ok - we checked that capacity != 0 (and `array` catches arithmetic errors)
            vulkano::memory::allocator::DeviceLayout::from_layout(
                std::alloc::Layout::array::<f32>(capacity)?,
            )
            .unwrap(),
        )?;

        Ok(Self { buffer })
    }
    /// Batches strokes from the given iterator, calling `F` on each batch.
    /// Duplicates in the input iterator are *not* deduplicated!
    ///
    /// The `StrokeBatch` given to the function is guaranteed to describe regions
    /// sorted by `offset` and all regions will be non-overlapping.
    ///
    /// # Errors
    /// Returns the inner error of the closure, or if a write into the staging buffer failed.
    /// In the case that an error is reported, the closure *must immediately release read/write access to the buffer*,
    /// or future uses of this batcher may fail.
    // Takes &mut self, because there is no internal sync!
    pub fn batch<Strokes, Future, F, InnerError>(
        &mut self,
        strokes: Strokes,
        mut consume: F,
    ) -> Result<(), BatchError<InnerError>>
    where
        Strokes: Iterator<Item = ImmutableStroke>,
        Future: vk::sync::GpuFuture,
        F: FnMut(&mut StrokeBatch) -> Result<SyncOutput<Future>, InnerError>,
        InnerError: std::fmt::Debug,
    {
        let mut peek = strokes.peekable();
        let mut batch = StrokeBatch {
            allocs: Vec::new(),
            // Dummy, immediately overwritten :V
            elements: vk::Subbuffer::new(self.buffer.clone()).cast_aligned(),
            residual: None,
        };
        while peek.peek().is_some() {
            self.batch_one(&mut batch, &mut peek, &mut consume)?;
            batch.allocs.clear();
        }
        Ok(())
    }
    /// Perform a single batch.
    fn batch_one<Strokes, Future, F, InnerError>(
        &mut self,
        // Hoist allocs out into main loop
        batch: &mut StrokeBatch,
        strokes: &mut std::iter::Peekable<Strokes>,
        mut consume: F,
    ) -> Result<(), BatchError<InnerError>>
    where
        Strokes: Iterator<Item = ImmutableStroke>,
        Future: vk::sync::GpuFuture,
        F: FnMut(&mut StrokeBatch) -> Result<SyncOutput<Future>, InnerError>,
        InnerError: std::fmt::Debug,
    {
        // Alignment is a NOP - we required align to f32 on alloc
        let subbuffer = vk::Subbuffer::new(self.buffer.clone()).cast_aligned::<f32>();
        let after_end_idx = {
            let mut write = subbuffer.write().map_err(BatchError::AccessError)?;
            // As OK - length of buf is never larger than u64.
            Self::fill(strokes, &mut batch.allocs, write.as_mut()) as u64
        };
        if after_end_idx == 0 {
            // No work to do!
            return Ok(());
        }
        // Vulkano does not allow zero-sized buffer slices.
        // Instead, use `None` to represent that state.
        if after_end_idx < subbuffer.len() {
            // First is *exclusive* of idx, second is inclusive.
            let (elements, residual) = subbuffer.split_at(after_end_idx);
            batch.elements = elements;
            batch.residual = Some(residual);
        } else {
            batch.elements = subbuffer;
            batch.residual = None;
        }
        let sync = consume(batch).map_err(BatchError::Inner)?;
        match sync {
            SyncOutput::Fence(f) => f.wait(None).map_err(BatchError::FenceError),
            SyncOutput::Immediate => Ok(()),
        }
    }
    /// Fill in the given buffer and allocs with sequential stroke data.
    /// Returns past-the-end index.
    fn fill<Strokes>(
        strokes: &mut std::iter::Peekable<Strokes>,
        into_allocs: &mut Vec<StrokeAlloc>,
        // Name is a hint that we have preferential access to *sequential* elements, not random access!
        into_sequential_elems: &mut [f32],
    ) -> usize
    where
        Strokes: Iterator<Item = ImmutableStroke>,
    {
        // This impl leaves a lot to be desired, with several global locks per stroke...
        // It should be moved into the points repository, where it has more info and
        // can deduplicate the IO logic as well.
        let mut next_pos = 0;
        while let Some(&next) = strokes.peek() {
            // not found o.O
            // fixme!
            let Some(info) = points::global().summary_of(next.point_collection) else {
                panic!("bad id!")
            };

            if into_sequential_elems.len() - next_pos < info.elements() {
                // Can't fit more
                break;
            }

            // not found o.O
            // fixme!
            let Ok(read) = points::global().try_get(next.point_collection) else {
                panic!("bad id!")
            };
            assert_eq!(read.len(), info.elements());

            // Copy over
            into_sequential_elems[next_pos..(next_pos + read.len())].clone_from_slice(&read);
            // Describe allocation
            into_allocs.push(StrokeAlloc {
                offset: next_pos,
                summary: info,
                src: next,
            });
            // Advance to next
            next_pos += read.len();
            let _ = strokes.next();
        }

        // Fellthrough - ran out of strokes to read!
        next_pos
    }
}

/// A large collection of continguous items on the heap, where concurrent immutable and mutable access are
/// allowed on opposite sides of the partition.
///
/// T::drop will *never* be run for items in this collection.
pub struct Slab<T: bytemuck::Pod, const N: usize> {
    /// a non-null pointer to array of slab_SIZE points.
    array: *mut T,
    /// Write access guard.
    write_access: parking_lot::Mutex<()>,
    /// Current past-the-end index for the allocator.
    /// Indices < this are considered immutable, >= considered mutable.
    ///
    /// ***It is a logic error to write to this without holding a lock!***
    bump_position: std::sync::atomic::AtomicUsize,
}

#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error("bump exceeds capacity")]
pub struct BumpTooLargeError;
pub struct SlabGuard<'a, T: bytemuck::Pod, const N: usize> {
    /// a non-null pointer to [T; N].
    array: *mut T,
    _write_access_lock: parking_lot::MutexGuard<'a, ()>,
    bump_position: &'a std::sync::atomic::AtomicUsize,
}
// This is the exact same impl on Slab itself, but using atomic ops. Code duplication icky!
impl<'a, T: bytemuck::Pod, const N: usize> SlabGuard<'a, T, N> {
    /// How many more items can fit? Note that, unlike `Slab::remaining`,
    /// this is the true capacity rather than just a hint.
    pub fn remaining(&self) -> usize {
        N.saturating_sub(self.position())
    }
    /// Get a references to the immutable and mutable parts of the allocator.
    /// Be sure to mark any consumed space as used with `self::bump`!
    ///
    /// Note that even while this exclusive lock is held, there may still be readers accessing
    /// the immutable section at any time!
    ///
    /// Items written here will *never* be dropped!
    pub fn parts_mut<'s>(&'s mut self) -> (&'s [T], &'s mut [T]) {
        let position = self.position();
        let mutable_size = N.saturating_sub(position);
        // Invariant should be upheld by everone else.
        assert!(position <= N);

        // Safety - must remain inside or one past-the-end of the alloc'd object
        // We checked with assert guard!
        let mutable_start = unsafe { self.array.add(position) };

        unsafe {
            (
                // immutable: position is guarded to be <= allocated length. Ok even if position == 0
                std::slice::from_raw_parts::<'s, T>(self.array, position),
                // mutable: self has exclusive access to the Slab's writable portion.
                //       We then hold self mutably while accessing.
                std::slice::from_raw_parts_mut::<'s, T>(mutable_start, mutable_size),
            )
        }
    }
    /// Bump the inner slab by this number of items. The items become frozen
    /// and cannot be modified if this call returns Ok().
    ///
    /// Returns Err and makes no changes if `num_elements > remaining`.
    pub fn bump(&mut self, num_elements: usize) -> Result<(), BumpTooLargeError> {
        // We perform a relaxed load here. This is fine, no races can occur - invariant is
        // that we have exclusive store access to this atomic.
        let position = self.position();
        if position
            .checked_add(num_elements)
            .is_some_and(|end| end <= N)
        {
            // Release - we must finish all prior ops (could be writes!) before the store occurs.
            self.bump_position.store(
                position + num_elements,
                std::sync::atomic::Ordering::Release,
            );
            Ok(())
        } else {
            Err(BumpTooLargeError)
        }
    }
    /// Returns the position of the bump index.
    /// This is the first index in the slab that is available for mutation, everything
    /// before this index is frozen and immutable.
    pub fn position(&self) -> usize {
        // Relaxed is fine. This thread is the only writer.
        self.bump_position
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
impl<T: bytemuck::Pod, const N: usize> Slab<T, N> {
    /// Lock the slab for exclusive low-level writing access.
    ///
    /// *Reads are still free to occur to any point before the bump index even while this lock is held.*
    pub fn lock<'a>(&'a self) -> SlabGuard<'a, T, N> {
        SlabGuard {
            array: self.array,
            bump_position: &self.bump_position,
            _write_access_lock: self.write_access.lock(),
        }
    }
    /// Locklessly get the current bump position. Indices < this are immutable,
    /// >= are mutable and unallocated.
    ///
    /// Requires `&mut self` and thus exclusive access. `Use self::lock` for shared access
    pub fn position(&mut self) -> usize {
        *self.bump_position.get_mut()
    }
    /// Locklessly get the number of available indices.
    ///
    /// Requires `&mut self` and thus exclusive access. `Use self::lock` for shared access
    pub fn remaining(&mut self) -> usize {
        N.saturating_sub(self.position())
    }
    /// Locklessly bump the allocator position forward some number of elements.
    /// On success, `num_elements` that used to be in the mutable section are now in the immutable section.
    ///
    /// Requires `&mut self` and thus exclusive access. `Use self::lock` for shared access
    pub fn bump(&mut self, num_elements: usize) -> Result<(), BumpTooLargeError> {
        let position = self.position();
        let new_position = position
            .checked_add(num_elements)
            .ok_or(BumpTooLargeError)?;
        if new_position > N {
            Err(BumpTooLargeError)
        } else {
            *self.bump_position.get_mut() = new_position;
            Ok(())
        }
    }
    /// Locklessly access immutable and mutable sections of the allocator.
    /// Note that even while exclusive access is held, readers may still have access to the immutable
    /// section!
    ///
    /// After writing, be sure to call `self::bump` to freeze the written data
    ///
    /// Requires `&mut self` and thus exclusive access. `Use self::lock` for shared access
    pub fn parts_mut<'s>(&'s mut self) -> (&'s [T], &'s mut [T]) {
        let position = self.position();
        let mutable_size = N.saturating_sub(position);
        // Invariant should be upheld by everone else.
        assert!(position <= N);

        // Safety - must remain inside or one past-the-end the alloc'd object
        // We checked with guard!
        let mutable_start = unsafe { self.array.add(position) };

        unsafe {
            (
                // immutable: position is guarded to be <= allocated length. Ok even if position == 0
                std::slice::from_raw_parts::<'s, T>(self.array, position),
                // mutable: self has exclusive access to the Slab's writable portion.
                //       We then hold self mutably while accessing.
                std::slice::from_raw_parts_mut::<'s, T>(mutable_start, mutable_size),
            )
        }
    }
    /// Try to allocate and write a contiguous slice, returning the start idx of where it was written.
    /// Items written here will *never* be dropped!
    ///
    /// If `Some(idx)` is returned, the data can be retrieved via `self::try_read(idx, data.len())`
    ///
    /// If not enough space, the `self` is left unchanged and None is returned.
    pub fn shared_bump_write(&self, data: &[T]) -> Option<usize> {
        if data.len() <= N {
            // Eager check for space before waiting on the lock.
            // Could still fail afterwards!
            if data.len()
                > N.saturating_sub(
                    self.bump_position
                        .load(std::sync::atomic::Ordering::Relaxed),
                )
            {
                return None;
            }
            // *might* fit. Try to lock and write.
            let mut lock = self.lock();
            let start = lock.position();

            let (_, unfilled) = lock.parts_mut();
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
    /// Try to read some continuous slice of data. returns None if the region is outside the span
    /// of the currently allocated memory.
    ///
    /// Performs no check that the given start and length correspond to a single suballocation.
    pub fn try_read(&self, start: usize, len: usize) -> Option<&'static [T]> {
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
            Some(unsafe { std::slice::from_raw_parts(self.array.add(start), len) })
        } else {
            None
        }
    }
    /// Get the number of indices currently in use.
    /// This is a hint - it may become immediately out-of-date and is not suitable for use in safety preconditions!
    pub fn hint_usage(&self) -> usize {
        self.bump_position
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Get the number of bytes currently in use.
    /// This is a hint - it may become immediately out-of-date and is not suitable for use in safety preconditions!
    pub fn hint_usage_bytes(&self) -> usize {
        self.bump_position
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_mul(std::mem::size_of::<T>())
    }
    /// Get the number of indices available for writing.
    /// This is a hint - it may become immediately out-of-date and is not suitable for use in safety preconditions!
    ///
    /// Use `self.lock().remaining()` for a true result.
    pub fn hint_remaining(&self) -> usize {
        N.saturating_sub(
            self.bump_position
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }
    /// Get the heap size in bytes.
    pub const fn size_bytes() -> usize {
        std::mem::size_of::<T>().saturating_mul(N)
    }
    /// Allocate a new slab of [T; N].
    ///
    /// `Self::try_new`, except terminates on allocation failure.
    pub fn new() -> Self {
        match Self::try_new() {
            Some(s) => s,
            None => std::alloc::handle_alloc_error(Self::layout()),
        }
    }
    /// Allocate a new slab of [T; N].
    ///
    /// Returns None if the allocation failed. To fail on this condition,
    /// prefer [std::alloc::handle_alloc_error] over a panic.
    ///
    /// There is no guaruntee that this won't terminate the process on failure instead of returning None.
    pub fn try_new() -> Option<Self> {
        let layout = Self::layout();
        assert_ne!(layout.size(), 0);
        // (is there a better way to get a large, arbitrarily-initialized heap array?)
        // Safety: Layout is statically generated and known to be OK,
        // T: Pod so can be initialized with zeros safely.
        let mem = unsafe { std::alloc::alloc_zeroed(layout).cast::<T>() };

        if mem.is_null() {
            None
        } else {
            Some(Self {
                array: mem,
                write_access: parking_lot::const_mutex(()),
                bump_position: 0.into(),
            })
        }
    }
    /// Free the memory of this slab. By default, memory is leaked on drop as the references to this slab's
    /// data live arbitrarily long.
    ///
    /// Destructors of the values are *not* run.
    ///
    /// Safety: There must not be any outstanding references to this slab's memory (acquired by `try_read`).
    pub unsafe fn free(self) {
        // Safety - using same layout as used to create it.
        // Use-after-free forwarded to this fn's safety contract.
        unsafe { std::alloc::dealloc(self.array as *mut _, Self::layout()) }
    }
    const fn layout() -> std::alloc::Layout {
        std::alloc::Layout::new::<[T; N]>()
    }
}
// Unsure of how necessary the bounds on T are here,
// I don't fully understand so just be as strict as possible.
// Safety - the pointer refers to heap mem, and can be transferred.
unsafe impl<T: Send + Sync + bytemuck::Pod, const N: usize> Send for Slab<T, N> {}
// Safety - The mutex prevents similtaneous mutable and immutable access.
unsafe impl<T: Sync + Sync + bytemuck::Pod, const N: usize> Sync for Slab<T, N> {}

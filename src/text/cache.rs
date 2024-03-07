use super::{interface, SizeClass, UnsizedGlyphKey};
use crate::vulkano_prelude::*;
use vulkano::{
    buffer::Subbuffer,
    memory::{
        allocator::{
            suballocator::{BuddyAllocator, Suballocation, Suballocator, SuballocatorError},
            AllocationType, DeviceLayout,
        },
        DeviceAlignment,
    },
};

#[must_use = "leaks mem on drop - must be explicitly dealloc'd!"]
struct VertexAllocation {
    // offsets of zero represents the start of the subbuffer,
    // length is in elements, not bytes
    vertices: Suballocation,
    indices: Suballocation,
}
struct VertexAllocator<Vertex, Allocator> {
    vertices: vk::Subbuffer<[Vertex]>,
    // u16 - if there are more than 65k indicies in a *single character* that would be a problem lmao
    // (>22k triangles) Lets just assume that's not gonna need to happen.
    // Arranged such that 0 means "first vertex of this allocation", so the draw call's base-vertex must be set accordingly.
    indices: vk::Subbuffer<[u16]>,
    vertex_allocator: Allocator,
    index_allocator: Allocator,
}
#[derive(thiserror::Error, Debug)]
pub enum VertexAllocationError {
    #[error("vertex alloc failed")]
    Index(#[source] SuballocatorError),
    #[error("vertex alloc failed")]
    Vertex(#[source] SuballocatorError),
    #[error("zero-size vertex or index buffer")]
    ZeroSize,
}
impl<Vertex, Allocator> VertexAllocator<Vertex, Allocator>
where
    Allocator: Suballocator,
{
    fn new(vertices: Subbuffer<[Vertex]>, indices: Subbuffer<[u16]>) -> Self {
        use vulkano::memory::allocator::suballocator::Region;
        Self {
            // Unwrap OK - we already have subbuffers to these regions, thus their lengths must be valid.
            vertex_allocator: Allocator::new(Region::new(0, vertices.len()).unwrap()),
            index_allocator: Allocator::new(Region::new(0, indices.len()).unwrap()),
            vertices,
            indices,
        }
    }
    /// Access the subbuffers referred to by this allocation.
    /// Vulkano makes this a safe op, but it is not checked whether this allocation is valid.
    fn get(&self, allocation: &VertexAllocation) -> (Subbuffer<[Vertex]>, Subbuffer<[u16]>) {
        (
            self.vertices.clone().slice(
                allocation.vertices.offset..(allocation.vertices.offset + allocation.vertices.size),
            ),
            self.indices.clone().slice(
                allocation.indices.offset..(allocation.indices.offset + allocation.indices.size),
            ),
        )
    }
    /// Allocate for the given arrays. *Does not write the slices* - they are merely for added type safety.
    fn allocate(
        &self,
        vertices: &[Vertex],
        indices: &[u16],
    ) -> Result<VertexAllocation, VertexAllocationError> {
        use vulkano::NonZeroDeviceSize;
        let indices = self
            .index_allocator
            .allocate(
                DeviceLayout::new(
                    NonZeroDeviceSize::new(indices.len() as u64)
                        .ok_or(VertexAllocationError::ZeroSize)?,
                    // Alignment, in elements, always 1.
                    DeviceAlignment::MIN,
                )
                // Infallible, alignment is one.
                .unwrap(),
                AllocationType::Linear,
                // suballocating a linear buffer - no alignment is needed.
                DeviceAlignment::MIN,
            )
            .map_err(VertexAllocationError::Index)?;
        let vertices = self
            .vertex_allocator
            .allocate(
                DeviceLayout::new(
                    NonZeroDeviceSize::new(vertices.len() as u64)
                        .ok_or(VertexAllocationError::ZeroSize)?,
                    // Alignment, in elements, always 1.
                    DeviceAlignment::MIN,
                )
                // Infallible, alignment is one.
                .unwrap(),
                AllocationType::Linear,
                // suballocating a linear buffer - no alignment is needed.
                DeviceAlignment::MIN,
            )
            .map_err(VertexAllocationError::Vertex)?;

        Ok(VertexAllocation { vertices, indices })
    }
    /// # Safety
    /// See Vulkano `suballocator::deallocate`
    unsafe fn deallocate(&self, alloc: VertexAllocation) {
        // Safety contract forwarded to this fn.
        unsafe {
            self.index_allocator.deallocate(alloc.indices);
            self.vertex_allocator.deallocate(alloc.vertices);
        }
    }
}
/// Holds all the tessellated allocations for this glyph, at potentially various sizes.
///
/// # Safety
/// This type's unsafe functions are never immediate UB, as this struct does no unsafe ops on its own.
/// However, upholding these invariants makes life a whole lot easier when it comes to the *actual* unsafe
/// operations that they are used for later. That is, *all allocations present in this collection are valid.*
#[derive(Default)]
struct MonotoneGlyphInfo {
    /// List of every available tesselated form, by size class.
    sizes: std::collections::BTreeMap<SizeClass, VertexAllocation>,
}
impl MonotoneGlyphInfo {
    /// Retrieve the allocation for the size class greater or equal to the given class.
    #[must_use]
    fn get_ceil(&self, class: SizeClass) -> Option<(SizeClass, &VertexAllocation)> {
        // Accept any class equal or larger to this one
        self.sizes.range(class..).next().map(|(s, a)| (*s, a))
    }
    /// Take all of the allocations that are not the largest.
    /// *it is up to the caller to free the returned allocations!*
    #[must_use = "leaks the vertex allocations if not consumed"]
    fn take_not_max(&mut self) -> std::collections::BTreeMap<SizeClass, VertexAllocation> {
        if let Some((&largest, _)) = self.sizes.last_key_value() {
            // Mutates in place, only leaving smaller in self.
            let only_largest = self.sizes.split_off(&largest);
            // Swap them out - take the smaller ones, leaving the largest.
            std::mem::replace(&mut self.sizes, only_largest)
        } else {
            // None found => None lower
            std::collections::BTreeMap::new()
        }
    }
    /// Get the entry of this class or a greater class.
    /// If not available, returns an empty entry to the given class.
    /// # Safety
    /// The allocation left in the entry, if any:
    /// * must be alive for the duration of its inclusion in this collection.
    #[must_use]
    unsafe fn ceil_entry(
        &mut self,
        class: SizeClass,
    ) -> (
        SizeClass,
        std::collections::btree_map::Entry<'_, SizeClass, VertexAllocation>,
    ) {
        if let Some((ceil_size, _)) = self.get_ceil(class) {
            // This is a double walk, i don't know how to do it any better tho :V
            // We have a size! Get it's entry:
            (ceil_size, self.sizes.entry(ceil_size))
        } else {
            (class, self.sizes.entry(class))
        }
    }
    /// Insert or replace the given size class, returning the old value allocation if any.
    /// # Safety
    /// The inserted allocation:
    /// * must be alive for the duration of its inclusion in this collection.
    unsafe fn insert(
        &mut self,
        class: SizeClass,
        allocation: VertexAllocation,
    ) -> Option<VertexAllocation> {
        self.sizes.insert(class, allocation)
    }
}
/// A color glyph simply delegates rendering to several other glyphs.
struct ColorGlyphInfo {
    /// Layers of this glyph. To render this color glyph at any given size class,
    /// place all of these glyphs at that same (or higher) class atop each other in order.
    layers: Box<[(super::ttf_parser::GlyphId, super::GlyphColorMode)]>,
}
/// One, the other, or both.
enum EitherBoth<A, B> {
    Left(A),
    Right(B),
    Both(A, B),
}
/// Glyphs can have COLR data, but at the same time can also have a contour!
/// If a given data is not present, it merely means it has not been requested and is thus not cached,
/// not that it's not available in the face itself.
struct GlyphInfo(
    // This is ALWAYS init at rest.
    std::mem::MaybeUninit<EitherBoth<MonotoneGlyphInfo, ColorGlyphInfo>>,
);
impl GlyphInfo {
    fn new_monotone() -> Self {
        Self(std::mem::MaybeUninit::new(EitherBoth::Left(
            MonotoneGlyphInfo::default(),
        )))
    }
    fn monotone(&self) -> Option<&MonotoneGlyphInfo> {
        // Safety: always init at rest.
        match unsafe { self.0.assume_init_ref() } {
            EitherBoth::Left(mono) | EitherBoth::Both(mono, _) => Some(mono),
            EitherBoth::Right(_) => None,
        }
    }
    fn monotone_mut(&mut self) -> Option<&mut MonotoneGlyphInfo> {
        // Safety: always init at rest.
        match unsafe { self.0.assume_init_mut() } {
            EitherBoth::Left(mono) | EitherBoth::Both(mono, _) => Some(mono),
            EitherBoth::Right(_) => None,
        }
    }
    fn monotone_mut_or_default(&mut self) -> &mut MonotoneGlyphInfo {
        // Take the inner value from self
        // Safety; always init at rest. here, we're in flux, so we take it just to
        // put it back again.
        let mut inner = unsafe { self.0.assume_init_read() };
        // Insert mono if not available.
        if let EitherBoth::Right(r) = inner {
            inner = EitherBoth::Both(MonotoneGlyphInfo::default(), r);
        };
        // Won't leak, we moved out of it.
        match self.0.write(inner) {
            EitherBoth::Left(mono) | EitherBoth::Both(mono, _) => mono,
            // Just made sure this isn't the case.
            EitherBoth::Right(_) => unreachable!(),
        }
    }
}
/// Stores generated glyph data, indexed by `(face, variation, index, size class)`.
pub struct GlyphCache {
    allocators: VertexAllocator<interface::Vertex, BuddyAllocator>,
    // Map between a varied face glyph, and it's color + potentially many tesselated forms.
    glyphs: hashbrown::HashMap<UnsizedGlyphKey, GlyphInfo>,
}
#[derive(thiserror::Error, Debug)]
pub enum CacheInsertError {
    #[error(transparent)]
    Alloc(#[from] VertexAllocationError),
    #[error(transparent)]
    AccessError(#[from] vk::HostAccessError),
    #[error("a mesh of this key already exists")]
    AlreadyExists,
}
#[derive(thiserror::Error, Debug)]
pub enum ColorInsertError {
    #[error("color subglyphs must not be have color themselves")]
    WouldRecurse,
}
impl GlyphCache {
    /// Create a new cache from subbuffers. Buffers must be host accessible.
    pub fn new(vertices: Subbuffer<[interface::Vertex]>, indices: Subbuffer<[u16]>) -> Self {
        Self {
            allocators: VertexAllocator::new(vertices, indices),
            glyphs: hashbrown::HashMap::new(),
        }
    }
    /// Access the buffers as-is. Use `get_mesh` to get the ranges into these buffers
    /// needed to draw a glyph.
    pub fn buffers(&self) -> (&Subbuffer<[interface::Vertex]>, &Subbuffer<[u16]>) {
        (&self.allocators.vertices, &self.allocators.indices)
    }
    /// Remove redundant small glyph allocations.
    pub fn clean(&mut self) {
        for (_, info) in &mut self.glyphs {
            // Color chars have no cleanup.
            if let Some(monotone_info) = info.monotone_mut() {
                // Collect allocs of small chars.
                let cleaned = monotone_info.take_not_max();
                for (_, alloc) in cleaned {
                    // Safety: We encapsulate this alloc, known not to be dealloc'd except where manually taken
                    unsafe {
                        self.allocators.vertex_allocator.deallocate(alloc.vertices);
                        self.allocators.index_allocator.deallocate(alloc.indices);
                    }
                }
            }
        }
    }
    /// Try to get the mesh for the given glyph and sizeclass, or None.
    ///
    /// The returned mesh indices are guaranteed to be valid for the range of vertices.
    // FIXME: terrible return type
    pub fn get_mesh(
        &self,
        key: &UnsizedGlyphKey,
        class: SizeClass,
    ) -> Option<([u32; 2], [u32; 2])> {
        self.glyphs
            .get(key)
            .and_then(GlyphInfo::monotone)
            .and_then(|mono| mono.get_ceil(class))
            .map(|(_, alloc)| {
                (
                    // Unwraps ok - checked at insertion time that they fit in u32
                    [
                        alloc.vertices.offset.try_into().unwrap(),
                        alloc.vertices.size.try_into().unwrap(),
                    ],
                    [
                        alloc.indices.offset.try_into().unwrap(),
                        alloc.indices.size.try_into().unwrap(),
                    ],
                )
            })
    }
    pub fn insert_color_info(
        &mut self,
        key: &UnsizedGlyphKey,
        color_info: &[()],
    ) -> Result<(), ColorInsertError> {
        todo!()
    }
    /// Insert or replace the mesh for a given glyph.
    /// # Safety
    /// The the every element of `indices` must be smaller than the length of `vertices`.
    pub unsafe fn insert_mesh(
        &mut self,
        key: UnsizedGlyphKey,
        class: SizeClass,
        vertices: &[super::interface::Vertex],
        indices: &[u16],
    ) -> Result<(), CacheInsertError> {
        // fixme!!
        assert!(u32::try_from(vertices.len()).is_ok() && u32::try_from(indices.len()).is_ok());
        crate::render_device::debug_assert_indices_safe(vertices, indices);
        // Insert into the glyph's data
        let monotone = self
            .glyphs
            .entry(key)
            .or_insert(GlyphInfo::new_monotone())
            .monotone_mut_or_default();

        // Make sure we don't overwrite anything (`self::insert` should be a strictly additive operation)
        if monotone
            .get_ceil(class)
            .is_some_and(|(found_class, _)| found_class == class)
        {
            return Err(CacheInsertError::AlreadyExists);
        }

        // Allocate and try to copy data in
        let allocation = self.allocators.allocate(vertices, indices)?;

        let try_memcpy = || -> Result<(), vk::HostAccessError> {
            let (vertex_buffer, index_buffer) = self.allocators.get(&allocation);
            vertex_buffer.write()?.copy_from_slice(vertices);
            index_buffer.write()?.copy_from_slice(indices);

            Ok(())
        };

        if let Err(e) = try_memcpy() {
            // Failed to write, dealloc.
            // Safety: We just made it of course it came from this allocator!
            unsafe {
                self.allocators.deallocate(allocation);
            }
            return Err(e.into());
        }

        // Safety: We encapsulate this alloc, known never to be
        // dealloc'd except where explicitly taken out of the set beforehand
        let old = unsafe { monotone.insert(class, allocation) };
        // Checked that this was a new alloc at the top.
        debug_assert!(old.is_none());

        Ok(())
    }
}

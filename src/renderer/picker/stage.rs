use crate::vulkano_prelude::*;

use std::sync::Arc;

#[derive(Clone, Copy)]
/// Represents the transform that must happen when sampling
/// when a blit has been elided (would otherwise always be non-mirrored)
struct BufferAddressing {
    h_mirror: bool,
    v_mirror: bool,
}
impl BufferAddressing {
    // iunno how any of this will work but it is encapsulated logic so whatver i figure out will work x3
    /// From a top-left and bottom-right, derive an addressing mode such that
    /// `[0,0]` is mapped to `top_left` and `extent` is mapped to `bottom_right`.
    fn from_points(top_left: [u32; 2], bottom_right: [u32; 2]) -> Self {
        let h_mirror = top_left[0] > bottom_right[0];
        let v_mirror = top_left[1] > bottom_right[1];

        Self { h_mirror, v_mirror }
    }
    /// Transform the coord based on this addressing scheme
    fn index(self, [x, y]: [u32; 2], [width, height]: [u32; 2]) -> Option<u64> {
        if x >= width || y >= height {
            None
        } else {
            let x = if self.h_mirror { width - x - 1 } else { x };
            let y = if self.v_mirror {
                height - y - 1
            } else {
                height
            };

            Some(u64::from(x) + u64::from(y) * u64::from(width))
        }
    }
}
#[derive(Clone, Copy)]
enum BufferState {
    /// There is no data.
    Uninit,
    /// Default packed.
    Packed,
    /// Sized with arbitrary addressing
    Sized {
        extent: [u32; 2],
        addressing: BufferAddressing,
    },
}
impl BufferState {
    /// Given a coord and full size, calculate the index of a given point.
    /// None if out-of-bounds.
    fn index(self, coord: [u32; 2], packed_extent: [u32; 2]) -> Option<u64> {
        match self {
            BufferState::Uninit => None,
            BufferState::Packed => {
                if coord[0] >= packed_extent[0] || coord[1] >= packed_extent[1] {
                    None
                } else {
                    Some(u64::from(coord[0]) + u64::from(coord[1]) * u64::from(packed_extent[0]))
                }
            }
            BufferState::Sized { extent, addressing } => addressing.index(coord, extent),
        }
    }
}
/// Holds the stage mutably while writing.
/// On drop, the fence is synchronized, and the mutability is held for that time.
pub struct ImageStageFuture<'stage, Future: GpuFuture> {
    stage: &'stage mut ImageStage,
    fence: vk::FenceSignalFuture<Future>,
}
impl<Future: GpuFuture> ImageStageFuture<'_, Future> {
    /// Detach the lifetime of the stage and take the future.
    ///
    /// This will never result in UB, however it may result in runtime errors on sampling if not properly synchronized!
    pub fn detach(self) -> vk::FenceSignalFuture<Future> {
        self.fence
    }
}
// todo: defer impl GpuFuture for ImageStageFuture

/// Stores a small staging image, and a buffer which it is decoded into.
/// Used for downloading and sampling images from the host.
///
/// Intended to be constructed once and re-used.
// Using vulkano's RawImage and manually-allocated mem, (and device capabilities allowing), we could
// safely interpret the same memory alloc as any of the needed image formats without needing to
// alloc for each one. *BONK* no premature optimization!!
pub struct ImageStage {
    image: Arc<vk::Image>,
    buffer: Arc<vk::Buffer>,
    /// Filled portion of the buffer, meaningful if BufferState is NOT `Uninit`
    init_slice: vk::Subbuffer<[u8]>,
    buffer_state: BufferState,
}
impl ImageStage {
    pub fn new(
        allocator: Arc<dyn vulkano::memory::allocator::MemoryAllocator>,
        format: vk::Format,
        staged_size: [u32; 2],
    ) -> anyhow::Result<Self> {
        super::check_valid_binary_format(format)?;
        let total_bytes = (u64::from(staged_size[0]))
            .checked_mul(u64::from(staged_size[1]))
            // This "block_size" is one texel (checked above)
            .and_then(|texels| texels.checked_mul(format.block_size()))
            .and_then(std::num::NonZeroU64::new)
            .ok_or_else(|| anyhow::anyhow!("bad stage image dimensions: {staged_size:?}"))?;
        // todo: check for blit capability, fallback for that.
        let image = vk::Image::new(
            allocator.clone(),
            vk::ImageCreateInfo {
                format,
                // Blit is a `transfer_dst` usage,,, weird
                usage: vk::ImageUsage::TRANSFER_SRC | vk::ImageUsage::TRANSFER_DST,
                image_type: vulkano::image::ImageType::Dim2d,
                extent: [staged_size[0], staged_size[1], 1],
                // We will use Graphics for both the blit and subsequent copy for simplicity.
                sharing: vk::Sharing::Exclusive,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                // Only device needs to access this.
                // Prefer a device local with no host access.
                memory_type_filter: vk::MemoryTypeFilter {
                    not_preferred_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
                    ..vk::MemoryTypeFilter::PREFER_DEVICE
                },
                ..Default::default()
            },
        )?;

        let buffer = vk::Buffer::new(
            allocator,
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::TRANSFER_DST,
                // Only the graphics queue will touch this.
                sharing: vk::Sharing::Exclusive,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                // "Download" buffer
                // "Random Access" may still fetch over the PCI bus, prefer to not have this happen!
                memory_type_filter: vk::MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | vk::MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            // Unwrap OK - an align of 1 will have no overflow possible.
            vulkano::memory::allocator::DeviceLayout::new(
                total_bytes,
                // vkCmdCopyImageToBuffer has no requirements nor performance
                // advisories regarding align, just use 1! :3
                vulkano::memory::DeviceAlignment::MIN,
            )
            .unwrap(),
        )?;

        Ok(Self {
            image,
            // Dummy, weh.
            init_slice: vk::Subbuffer::new(buffer.clone()),
            buffer,
            buffer_state: BufferState::Uninit,
        })
    }
    /// Execute a transfer from the given input image and texel range, downsampled into host memory.
    /// Note that the names here are a tad misleading - they could specify other corners,
    /// and `bottom_right` is not necessarily > `top_left`. This corresponds to a mirroring operation.
    ///
    /// `other` and `self` is in use until the returned future.
    // There isn't much logical reason to break this up, and any further line
    // compression will serve to make it *less* readable.
    #[allow(clippy::too_many_lines)]
    pub fn download(
        &mut self,
        ctx: &crate::render_device::RenderContext,
        other: Arc<vk::Image>,
        subresource: vk::ImageSubresourceLayers,
        top_left: [i32; 2],
        bottom_right: [i32; 2],
    ) -> anyhow::Result<ImageStageFuture<impl GpuFuture>> {
        use az::SaturatingAs;
        // Don't allow "compatible format" or "mutable format" features.
        if other.format() != self.image.format() {
            anyhow::bail!("image formats must match")
        }
        // Vulkan specifies that blit coordinates are wrapped via `CLAMP_TO_EDGE`
        // Perform that transform ourselves so we can see the true coordinates used:
        // Saturate to 0,0, further clamp to image extent
        let image_extent = other.extent();

        let top_left: [u32; 2] = [top_left[0].saturating_as(), top_left[1].saturating_as()];
        let top_left = [
            top_left[0].min(image_extent[0]),
            top_left[1].min(image_extent[1]),
        ];
        let bottom_right: [u32; 2] = [
            bottom_right[0].saturating_as(),
            bottom_right[1].saturating_as(),
        ];
        let bottom_right = [
            bottom_right[0].min(image_extent[0]),
            bottom_right[1].min(image_extent[1]),
        ];

        // Absolute diff gives us the effective size of the transfer op.
        let src_transfer_extent = [
            top_left[0].abs_diff(bottom_right[0]),
            top_left[1].abs_diff(bottom_right[1]),
        ];
        // Number of bytes to transfer this many texels.
        // This is ok, since we checked that self.image is valid bitwise format and other is same format.
        let src_transfer_size = u64::from(src_transfer_extent[0])
            * u64::from(src_transfer_extent[1])
            * other.format().block_size();

        let (opt_blit, copy, new_state) = if src_transfer_size <= self.buffer.size() {
            // We can skip the blit and just do a transfer from `other` to `self.buffer`!
            // A blit would only serve to *inflate* the image, no good!
            // We can handle negative coordinate mapping by mirroring sampled coords on the host side.
            let image_offset = [
                top_left[0].min(bottom_right[0]),
                top_left[1].min(bottom_right[1]),
                0,
            ];
            let copy = vk::BufferImageCopy {
                image_offset,
                image_extent: [src_transfer_extent[0], src_transfer_extent[1], 0],
                image_subresource: subresource,
                // buffer layout spec left at zeros -
                // automatically packs the texels tightly.
                ..Default::default()
            };
            let copy = vk::CopyImageToBufferInfo {
                regions: smallvec::smallvec![copy],
                ..vk::CopyImageToBufferInfo::image_buffer(
                    other,
                    vk::Subbuffer::new(self.buffer.clone()),
                )
            };
            (
                None,
                copy,
                BufferState::Sized {
                    extent: src_transfer_extent,
                    addressing: BufferAddressing::from_points(top_left, bottom_right),
                },
            )
        } else {
            // Too large, gotta blit!
            let self_extent = self.image.extent();
            let self_subresource = vk::ImageSubresourceLayers {
                mip_level: 0,
                array_layers: 0..1,
                aspects: vk::ImageAspects::COLOR,
            };
            // Map the entire source image onto self, even if that transform is anisotropic.
            let blit = vk::ImageBlit {
                src_offsets: [
                    [top_left[0], top_left[1], 0],
                    [bottom_right[0], bottom_right[1], 0],
                ],
                src_subresource: subresource,
                // Entire self.image:
                dst_offsets: [[0; 3], [self_extent[0], self_extent[1], 0]],
                dst_subresource: self_subresource.clone(),
                ..Default::default()
            };
            let blit = vk::BlitImageInfo {
                filter: vk::Filter::Nearest,
                regions: smallvec::smallvec![blit],
                ..vk::BlitImageInfo::images(other, self.image.clone())
            };
            // Then copy the blitted image into our buffer!
            let copy = vk::BufferImageCopy {
                image_offset: [0; 3],
                image_extent: [self_extent[0], self_extent[1], 0],
                image_subresource: self_subresource,
                // buffer specification left at zeros -
                // automatically packs the texels tightly.
                ..Default::default()
            };
            let copy = vk::CopyImageToBufferInfo {
                regions: smallvec::smallvec![copy],
                ..vk::CopyImageToBufferInfo::image_buffer(
                    self.image.clone(),
                    vk::Subbuffer::new(self.buffer.clone()),
                )
            };
            (Some(blit), copy, BufferState::Packed)
        };

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            ctx.allocators().command_buffer(),
            ctx.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        if let Some(blit) = opt_blit {
            command_buffer.blit_image(blit)?;
        }
        command_buffer.copy_image_to_buffer(copy)?;
        let command_buffer = command_buffer.build()?;

        // Execute and synchronize.
        let fence = vk::sync::now(ctx.device().clone())
            .then_execute(ctx.queues().graphics().queue().clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        self.buffer_state = new_state;
        // Won't panic - we just set self to not be uninit.
        self.update_self_slice();

        // Return the future and bind our lifetime to it.
        Ok(ImageStageFuture { fence, stage: self })
    }
    /// Extent of the internal buffer, derived from `self.buffer_state`. None if uninit.
    pub fn extent(&self) -> Option<[u32; 2]> {
        match self.buffer_state {
            BufferState::Packed => {
                let extent = self.image.extent();
                Some([extent[0], extent[1]])
            }
            BufferState::Sized { extent, .. } => Some(extent),
            BufferState::Uninit => None,
        }
    }
    /// Number of init bytes of internal buffer, derived from `self.buffer_state`. None if uninit.
    pub fn init_len_bytes(&self) -> Option<u64> {
        let extent = self.extent()?;
        // checked at create time that a single block == a single texel
        let texel_size = self.image.format().block_size();
        Some(u64::from(extent[0]) * u64::from(extent[1]) * texel_size)
    }
    /// Update `self.init_slice`. Panics if `buffer_state` is uninit or reports an extent larger than can fit.
    fn update_self_slice(&mut self) {
        // Logic error for self to be uninit.
        let len = self.init_len_bytes().unwrap();
        self.init_slice = vk::Subbuffer::new(self.buffer.clone()).slice(0..len);
    }
    /// Fetch a single texel from the local buffer.
    /// Image bits at the given location are interpreted bitwise as `Texel`.
    ///
    /// `local` is in local space, where `top_left` is `[0, 0]` increasing towards `self.extent() - [1, 1]`
    /// at `bottom_right` (this may be a mirrored representation of the image, if the coordinates passed to
    /// `self::download` were specified as such)
    ///
    /// For many texel fetches, it is much more efficient to take a sampler using `Self::{owned_sampler, sampler}`.
    pub fn fetch<Texel>(&self, local: [u32; 2]) -> Result<Texel, SamplingError>
    where
        Texel: bytemuck::Pod,
    {
        let texel_size = std::mem::size_of::<Texel>() as u64;
        // Can't think of a way to enforce this better than a runtime err :V
        // Could instead have typed wrappers that check once at creation time?
        if texel_size == self.image.format().block_size() {
            if matches!(self.buffer_state, BufferState::Uninit) {
                return Err(SamplingError::Uninit);
            }
            let extent = self.image.extent();
            let idx = self
                .buffer_state
                .index(local, [extent[0], extent[1]])
                .ok_or(SamplingError::OutOfBounds)?;
            // Slice before accessing, so that only the bit we care about
            // actually gets memmapped.
            let byte_range = (idx * texel_size)..((idx + 1) * texel_size);
            // Check slice range (`.slice` would panic)
            if byte_range.end > self.init_slice.len() {
                return Err(SamplingError::OutOfBounds);
            }
            let sliced = self.init_slice.clone().slice(byte_range);
            let read = sliced.read().map_err(SamplingError::AccessError)?;

            // Mapped buffers are aligned to the gcd of the
            // physical device's `minMemoryMapAlignment` and `Texel`'s size - which could be less than `Texel`'s align.

            // On most systems it is aligned to a very large amount, but to check would
            // prolly be more expensive than it's worth.
            Ok(bytemuck::pod_read_unaligned(&read))
        } else {
            Err(SamplingError::BadSize)
        }
    }
    /// Take a heap copy of the image, to be used for sampling.
    /// For a single texel fetch, it is more efficient to use `Self::fetch`.
    pub fn owned_sampler<Texel: bytemuck::Pod>(
        &self,
    ) -> Result<OwnedSampler<Texel>, SamplingError> {
        self.sampler().map(BorrowedSampler::to_owned)
    }
    /// Take a reference to the image, to be used for sampling.
    /// For a single texel fetch, it is more efficient to use `Self::fetch`.
    pub fn sampler<Texel: bytemuck::Pod>(
        &self,
    ) -> Result<BorrowedSampler<'_, Texel>, SamplingError> {
        let texel_size = std::mem::size_of::<Texel>() as u64;
        if texel_size == self.image.format().block_size() {
            let (extent, addressing) = match self.buffer_state {
                BufferState::Uninit => return Err(SamplingError::Uninit),
                BufferState::Packed => {
                    let extent = self.image.extent();
                    (
                        [extent[0], extent[1]],
                        BufferAddressing {
                            h_mirror: false,
                            v_mirror: false,
                        },
                    )
                }
                BufferState::Sized { extent, addressing } => (extent, addressing),
            };

            let mapped_bytes = self.init_slice.read().map_err(SamplingError::AccessError)?;

            let expected_len = u64::from(extent[0]) * u64::from(extent[1]) * texel_size;
            // BorrowedSampler assumes this invariant, so just make sure we didn't bork anything.
            assert_eq!(expected_len, mapped_bytes.len() as u64);
            Ok(BorrowedSampler {
                extent,
                addressing,
                mapped_bytes,
                _phantom: std::marker::PhantomData,
            })
        } else {
            Err(SamplingError::BadSize)
        }
    }
}
#[derive(thiserror::Error, Debug)]
pub enum SamplingError {
    /// The sample falls outside the staged image region.
    #[error("sample coordinate out-of-bounds")]
    OutOfBounds,
    /// The buffer is uninit..
    #[error("staging buffer has not been filed yet")]
    Uninit,
    /// Access error occured. The device is still writing this buffer, hints
    /// at a failure to await the fence.
    #[error(transparent)]
    AccessError(vulkano::sync::HostAccessError),
    /// The size of the provided `Texel` is not correct.
    #[error("texel size does not match image format")]
    BadSize,
}
pub struct BorrowedSampler<'stage, Texel> {
    extent: [u32; 2],
    /// Buffer mirroring mode
    addressing: BufferAddressing,
    /// Bytes that represent the texels. NOT GUARANTEED TO BE ALIGNED TO `TEXEL`!
    /// Must have precisely the right length according the `Texel size * extent`
    mapped_bytes: vulkano::buffer::subbuffer::BufferReadGuard<'stage, [u8]>,
    /// Pretend we hold slice of `Texel`s.
    _phantom: std::marker::PhantomData<&'stage [Texel]>,
}
impl<Texel: bytemuck::Pod> BorrowedSampler<'_, Texel> {
    /// Make a heap copy of this sampler, ending the borrow of the `ImageStage`.
    // Can't use `ToOwned::to_owned`, as `Self` is not `Borrow<OwningSampler>` (and cannot be)
    pub fn to_owned(self) -> OwnedSampler<Texel> {
        let bytes: &[u8] = &self.mapped_bytes;
        // checked at construction time!
        debug_assert_eq!(bytes.len() % std::mem::size_of::<Texel>(), 0);
        // SAFETY: `bytemuck::Pod` implies `bytemuck::Zeroable`, thus zeros are a correctly initialized `Texel` instance.
        let mut texels = unsafe {
            // May be able to elide this zeroing by writing bytes immediately using `slice_as_bytes`,
            // but things get more dangerous!
            Box::<[Texel]>::new_zeroed_slice(bytes.len() / std::mem::size_of::<Texel>())
                .assume_init()
        };

        // memcpy `bytes` into `texels`.
        // No panic - of course it's aligned to ones, and we checked at creation time the lens match exactly.
        // Doing it this way allows efficient copying of data regardless of align, yippee!
        bytemuck::cast_slice_mut(&mut texels).copy_from_slice(bytes);

        OwnedSampler {
            extent: self.extent,
            addressing: self.addressing,
            data: texels,
        }
    }
}
impl<Texel: bytemuck::Pod> Sampler for BorrowedSampler<'_, Texel> {
    type Texel = Texel;
    fn fetch(&self, local: [u32; 2]) -> Option<Texel> {
        use az::CheckedAs;
        let texel_size = std::mem::size_of::<Texel>() as u64;
        let elem_idx = self.addressing.index(local, self.extent)?;

        let byte_range = (elem_idx * texel_size)..((elem_idx + 1) * texel_size);
        // Only fallible on 32bit
        let byte_range = byte_range.start.checked_as()?..byte_range.end.checked_as()?;
        // None if bytes out-of-bounds (shouldn't be possible due to self's preconditions! oh well, handle it)
        let bytes = self.mapped_bytes.get(byte_range)?;

        // No guaruntee that memmapped `[u8]` is aligned to `[Texel]`.
        // On most systems it is aligned to a very large amount, but to check would
        // prolly be more expensive than it's worth.
        Some(bytemuck::pod_read_unaligned(bytes))
    }
    fn extent(&self) -> [u32; 2] {
        self.extent
    }
}

/// A sample which holds a heap copy of the image data.
pub struct OwnedSampler<Texel> {
    extent: [u32; 2],
    /// Buffer mirroring mode
    addressing: BufferAddressing,
    data: Box<[Texel]>,
}
impl<Texel: bytemuck::Pod> Sampler for OwnedSampler<Texel> {
    type Texel = Texel;
    fn fetch(&self, local: [u32; 2]) -> Option<Texel> {
        use az::CheckedAs;
        let idx: usize = self
            .addressing
            .index(self.extent, local)?
            // Only fallible on 32bit
            .checked_as()?;
        self.data.get(idx).copied()
    }
    fn extent(&self) -> [u32; 2] {
        self.extent
    }
}
pub trait Sampler {
    type Texel;
    fn extent(&self) -> [u32; 2];
    /// Fetch a texel from the image, in local space.
    ///
    /// `None` if out-of-bounds.
    fn fetch(&self, local: [u32; 2]) -> Option<Self::Texel>;
    /// Fetch a texel from normalized UV.
    /// `None` if out-of-bounds, `NaN`, or `inf`.
    // These lossy casts are intentional and/or checked.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn fetch_nearest_normalized(&self, normalized: [f32; 2]) -> Option<Self::Texel> {
        use std::num::FpCategory;
        // Not nan nor inf
        match (normalized[0].classify(), normalized[1].classify()) {
            (_, FpCategory::Infinite | FpCategory::Nan)
            | (FpCategory::Infinite | FpCategory::Nan, _) => return None,
            _ => (),
        };
        // Not out-of-range
        if normalized[0] < 0.0 || normalized[0] > 1.0 || normalized[1] < 0.0 || normalized[1] > 1.0
        {
            return None;
        }
        let extent = self.extent();
        // map 0..=1 (inclusive) to 0..extent (exclusive)
        // why is everything i wanna use unstable :cry:
        // we don't expect these `as` casts to be lossy as a stage that large (16,777,216) is absurd!
        // these `as` rounds are not a soundness issue regardless.
        let coord = [
            normalized[0] * (extent[0] as f32).next_down(),
            normalized[1] * (extent[1] as f32).next_down(),
        ];
        // Truncate to integer coordinate.
        // As ok - truncation is intentional, and it's known non-negative.
        let coord = [coord[0] as u32, coord[1] as u32];

        self.fetch(coord)
    }
}

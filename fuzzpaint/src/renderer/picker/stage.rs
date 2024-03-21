use crate::vulkano_prelude::*;

use std::sync::Arc;

/// Metadata about the picked region, allowing remapping of texture space to sample space.
/// `dst_extent` gives the size of the buffer data, while `src_extent` tells the region of the image it is sourced from.
#[derive(Copy, Clone)]
struct ResizedImageExtent {
    /// Origin of the samples.
    origin: [u32; 2],
    /// Extent of the src image that the internal data represents.
    src_extent: [u32; 2],
    /// Internal size of the data. May be lesser or greater than the stage size!
    dst_extent: [u32; 2],
}
impl ResizedImageExtent {
    /// Given a texel coordinate in *src space*, convert it to a texel coordinate in internal buffer space.
    /// None if out of bounds!
    fn remap(self, [mut x, mut y]: [u32; 2]) -> Option<[u32; 2]> {
        // Checked - if goes below zero, we're out-of-bounds
        x = x.checked_sub(self.origin[0])?;
        y = y.checked_sub(self.origin[1])?;
        // Make sure not past the edge
        if x >= self.src_extent[0] || y >= self.src_extent[1] {
            return None;
        }
        // dst extent will never be larger than src_extent.
        // just handle it here so that math is more well-defined later on :P
        x = if self.dst_extent[0] >= self.src_extent[0] {
            x
        } else {
            Self::int_remap(x, self.src_extent[0], self.dst_extent[0])
        };
        y = if self.dst_extent[1] >= self.src_extent[1] {
            y
        } else {
            Self::int_remap(y, self.src_extent[1], self.dst_extent[1])
        };

        Some([x, y])
    }
    /// Given a texel coordinate in *src space*, convert it to an texel index into the packed buffer.
    fn index(self, pos: [u32; 2]) -> Option<u64> {
        let [x, y] = self.remap(pos)?;
        debug_assert!(x < self.dst_extent[0]);
        debug_assert!(y < self.dst_extent[1]);
        // math infallible.
        Some(u64::from(x) + u64::from(y) * u64::from(self.dst_extent[0]))
    }
    /// Map an int from `0..old_max` to `0..new_max`, ***assuming the `new_max` is smaller***
    /// Clamps to range.
    fn int_remap(value: u32, old_max_exclusive: u32, new_max_exclusive: u32) -> u32 {
        // Handle badness.
        if value >= old_max_exclusive {
            return new_max_exclusive;
        }
        // won't overflow:   u32::MAX * u32::MAX < u64::MAX
        let numerator = u64::from(value) * u64::from(new_max_exclusive);
        // Not zero - checked above.
        let value = numerator / u64::from(old_max_exclusive);
        debug_assert!(u32::try_from(value).is_ok());
        let value = u32::try_from(value).unwrap();

        // Ensure we didn't overshoot
        value.min(new_max_exclusive)
    }
}
/// Holds the stage mutably while writing.
/// On drop, the fence is synchronized, and the mutability is held for that time.
pub struct ImageStageFuture<'stage, Future: GpuFuture> {
    _stage: &'stage mut ImageStage,
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
    buffer_state: Option<ResizedImageExtent>,
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
            buffer_state: None,
        })
    }
    /// Execute a transfer from the given input image and texel range, downsampled into host memory.
    ///
    /// `other` and `self` is in use until the returned future signals.
    ///
    /// # Errors
    /// Returns any vulkan errors that happen in the course of transferring, or if the image region falls outside of
    /// the available region
    // There isn't a lot of separation of concerns to do here..
    // Would make it worse to split it apart instead of having one long fn, imho.
    #[allow(clippy::too_many_lines)]
    pub fn download(
        &mut self,
        ctx: &crate::render_device::RenderContext,
        other: Arc<vk::Image>,
        subresource: vk::ImageSubresourceLayers,
        origin: [u32; 2],
        extent: [u32; 2],
    ) -> anyhow::Result<ImageStageFuture<impl GpuFuture>> {
        // Don't allow "compatible format" or "mutable format" features.
        if other.format() != self.image.format() {
            anyhow::bail!("image formats must match")
        }
        if subresource.mip_level != 0 {
            // Realized this after writing all the logic. :V
            // Not going to use this functionality anytime soon though.
            anyhow::bail!("mip levels unsupported, FIXME!")
        }
        // Vulkan specifies that blit coordinates are wrapped via `CLAMP_TO_EDGE`
        // Perform that transform ourselves so we can see the true coordinates used:
        let max_coordinate = other.extent();
        let clamped_max_coordinate = [
            origin[0].saturating_add(extent[0]).min(max_coordinate[0]),
            origin[1].saturating_add(extent[1]).min(max_coordinate[1]),
        ];

        let clamped_extent = [
            clamped_max_coordinate[0].saturating_sub(origin[0]),
            clamped_max_coordinate[1].saturating_sub(origin[1]),
        ];
        // Transparently handles origin > max.
        if clamped_extent[0] == 0 || clamped_extent[1] == 0 {
            // Not striclty an error, we could return an empty sampler. However it would muddy the return type.
            anyhow::bail!("can't create zero-size sampler")
        }
        // Number of bytes to transfer this many texels.
        // It is ok to interpret block_size as texel size, since we checked that self.image is
        // valid bitwise format and other is same format.
        let src_transfer_size = u64::from(clamped_extent[0])
            * u64::from(clamped_extent[1])
            * other.format().block_size();

        let (opt_blit, copy, new_state) = if src_transfer_size <= self.buffer.size() {
            // We can skip the blit and just do a transfer from `other` to `self.buffer`!
            // A blit would only serve to *inflate* the image, no good!
            // We can handle negative coordinate mapping by mirroring sampled coords on the host side.
            let image_offset = [origin[0], origin[1], 0];
            let copy = vk::BufferImageCopy {
                image_offset,
                image_extent: [clamped_extent[0], clamped_extent[1], 0],
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
                ResizedImageExtent {
                    origin,
                    // Same extent for both - no resize was needed, and none should be performed at sampling time.
                    src_extent: clamped_extent,
                    dst_extent: clamped_extent,
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
                    [origin[0], origin[1], 0],
                    [clamped_max_coordinate[0], clamped_max_coordinate[1], 0],
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
            (
                Some(blit),
                copy,
                ResizedImageExtent {
                    origin,
                    // Resizing was necessary, let the sampler know the extents differ.
                    src_extent: clamped_extent,
                    dst_extent: [self_extent[0], self_extent[1]],
                },
            )
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

        self.buffer_state = Some(new_state);
        // Won't panic - we just set self to not be uninit.
        self.update_self_slice();

        // Return the future and bind our lifetime to it.
        Ok(ImageStageFuture {
            fence,
            _stage: self,
        })
    }
    /// Extent of the internal buffer, derived from `self.buffer_state`. None if uninit.
    pub fn extent(&self) -> Option<[u32; 2]> {
        self.buffer_state.map(|s| s.dst_extent)
    }
    /// Number of init bytes of internal buffer, derived from `self.buffer_state`. None if uninit.
    pub fn init_len_bytes(&self) -> Option<u64> {
        let extent = self.extent()?;
        // checked at create time that a single block == a single texel
        let texel_size = self.image.format().block_size();
        Some(u64::from(extent[0]) * u64::from(extent[1]) * texel_size)
    }
    /// Update `self.init_slice`. Panics if `buffer_state` is None or reports an extent larger than can fit.
    fn update_self_slice(&mut self) {
        // Logic error for self to be uninit.
        let len = self.init_len_bytes().unwrap();
        self.init_slice = vk::Subbuffer::new(self.buffer.clone()).slice(0..len);
    }
    /// Take a heap copy of the image, to be used for sampling.
    /// For a single texel fetch, it is more efficient to use `Self::fetch`.
    pub fn owned_sampler<Texel: bytemuck::Pod>(
        &self,
    ) -> Result<OwnedSampler<Texel>, SamplingError> {
        Ok(self.sampler()?.to_owned())
    }
    /// Take a reference to the image, to be used for sampling.
    /// For a single texel fetch, it is more efficient to use `Self::fetch`.
    pub fn sampler<Texel: bytemuck::Pod>(
        &self,
    ) -> Result<BorrowedSampler<'_, Texel>, SamplingError> {
        let texel_size = std::mem::size_of::<Texel>() as u64;
        if texel_size == self.image.format().block_size() {
            let Some(extents) = self.buffer_state else {
                return Err(SamplingError::Uninit);
            };

            let mapped_bytes = self.init_slice.read().map_err(SamplingError::AccessError)?;

            let expected_len =
                u64::from(extents.dst_extent[0]) * u64::from(extents.dst_extent[1]) * texel_size;
            // `BorrowedSampler` assumes this invariant, so just make sure we didn't bork anything.
            assert_eq!(expected_len, mapped_bytes.len() as u64);
            Ok(BorrowedSampler {
                extents,
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
    /// The buffer is uninit..
    #[error("staging buffer has not been filled yet")]
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
    extents: ResizedImageExtent,
    /// Bytes that represent the texels. NOT GUARANTEED TO BE ALIGNED TO `TEXEL`!
    /// Must have precisely the right length according the `Texel size * extent`
    mapped_bytes: vulkano::buffer::subbuffer::BufferReadGuard<'stage, [u8]>,
    /// Pretend we hold slice of `Texel`s.
    _phantom: std::marker::PhantomData<&'stage [Texel]>,
}
impl<Texel: bytemuck::Pod> BorrowedSampler<'_, Texel> {
    /// Make a heap copy of this sampler, ending the borrow of the `ImageStage`.
    // Can't use `ToOwned::to_owned`, as `Self` is not `Borrow<OwningSampler>` (and cannot be)
    pub fn to_owned(&self) -> OwnedSampler<Texel> {
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
            extents: self.extents,
            data: texels,
        }
    }
}
impl<Texel: bytemuck::Pod> Sampler for BorrowedSampler<'_, Texel> {
    type Texel = Texel;
    fn fetch(&self, coord: [u32; 2]) -> Option<Texel> {
        use az::CheckedAs;
        let elem_idx = self.extents.index(coord)?;
        let texel_size = std::mem::size_of::<Texel>() as u64;

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
}

/// A sample which holds a heap copy of the image data.
pub struct OwnedSampler<Texel> {
    extents: ResizedImageExtent,
    data: Box<[Texel]>,
}
impl<Texel: bytemuck::Pod> Sampler for OwnedSampler<Texel> {
    type Texel = Texel;
    fn fetch(&self, coord: [u32; 2]) -> Option<Texel> {
        use az::CheckedAs;
        let idx: usize = self
            .extents
            .index(coord)?
            // Only fallible on 32bit
            .checked_as()?;
        self.data.get(idx).copied()
    }
}
pub trait Sampler {
    type Texel;
    /// Fetch a texel from the image, in it's local space.
    ///
    /// `None` if out-of-bounds of this sampler.
    fn fetch(&self, coord: [u32; 2]) -> Option<Self::Texel>;
    /*/// Fetch a texel from normalized UV.
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
    }*/
}

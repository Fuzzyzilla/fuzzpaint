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
    fn address(self, [x, y]: [u32; 2], [width, height]: [u32; 2]) -> Option<u64> {
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
                if coord[0] >= packed_extent[0] || coord[1] > packed_extent[1] {
                    None
                } else {
                    Some(u64::from(coord[0]) + u64::from(coord[1]) * u64::from(packed_extent[0]))
                }
            }
            BufferState::Sized { extent, addressing } => addressing.address(coord, extent),
        }
    }
}

/// Stores a small staging image, and a buffer which it is decoded into.
/// Used for downloading and sampling images from the host.
///
/// Intended to be constructed once and re-used.
// Using vulkano's RawImage and manually-allocated mem, (and device capabilities allowing), we could
// safely interpret the same memory alloc as any of the needed image formats without needing to
// alloc for each one. *BONK* no premature optimization!!
struct ImageStage {
    image: Arc<vk::Image>,
    buffer: Arc<vk::Buffer>,
    buffer_state: BufferState,
}
impl ImageStage {
    fn new(
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
                memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
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
                memory_type_filter: vk::MemoryTypeFilter::HOST_RANDOM_ACCESS,
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
            buffer,
            buffer_state: BufferState::Uninit,
        })
    }
    /// Execute a transfer from the given input image and texel range, downsampled into host memory.
    /// Note that the names here are a tad misleading - they could specify other corners,
    /// and `bottom_right` is not necessarily > `top_left`. This corresponds to a mirroring operation.
    // There isn't much logical reason to break this up, and any further line
    // compression will serve to make it *less* readable.
    #[allow(clippy::too_many_lines)]
    fn download(
        &mut self,
        ctx: &crate::render_device::RenderContext,
        other: Arc<vk::Image>,
        subresource: vk::ImageSubresourceLayers,
        top_left: [i32; 2],
        bottom_right: [i32; 2],
    ) -> anyhow::Result<vk::FenceSignalFuture<impl GpuFuture>> {
        use az::SaturatingAs;
        // Don't allow "compatible formats" or "mutable format" features.
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

        Ok(fence)
    }
    /// Fetch a single texel from the local buffer.
    /// Image bits at the given location are interpreted bitwise as `Texel`.
    ///
    /// `coord`: `top_left` is `[0, 0]` increasing towards `bottom_right` (this may be a
    /// mirrored representation of the image, if the coordinates passed to `self::transfer`
    /// were specified as such)
    fn fetch<Texel>(&self, coord: [u32; 2]) -> Result<Texel, SamplingError>
    where
        Texel: bytemuck::Pod,
    {
        let texel_size = std::mem::size_of::<Texel>() as u64;
        // Can't think of a way to enforce this better than a runtime err :V
        // Could instead have typed wrappers that check once at creation time?
        if texel_size == self.image.format().block_size() {
            let extent = self.image.extent();
            let idx = self
                .buffer_state
                .index(coord, [extent[0], extent[1]])
                .ok_or(SamplingError::OutOfBounds)?;
            let buf = vk::Subbuffer::new(self.buffer.clone());
            // Slice before accessing, so that only the bit we care about
            // actually gets memmapped.
            let byte_range = (idx * texel_size)..((idx + 1) * texel_size);
            // Check slice range (`.slice` would panic)
            if byte_range.end > buf.len() {
                return Err(SamplingError::OutOfBounds);
            }
            let sliced = buf.slice(byte_range);
            let read = sliced.read().map_err(SamplingError::AccessError)?;

            // This works, but I am not sure of the alignment of mapped buffers.
            Ok(bytemuck::pod_read_unaligned(&read))
        } else {
            Err(SamplingError::BadSize)
        }
    }
}
#[derive(thiserror::Error, Debug)]
enum SamplingError {
    /// The sample falls outside the staged image region.
    #[error("sample coordinate out-of-bounds")]
    OutOfBounds,
    /// Access error occured. The device is still writing this buffer, hints
    /// at a failure to await the fence.
    #[error(transparent)]
    AccessError(vulkano::sync::HostAccessError),
    /// The size of the provided `Texel` is not correct.
    #[error("texel size does not match image format")]
    BadSize,
}

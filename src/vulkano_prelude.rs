//Traits
pub use vulkano::{
    pipeline::{graphics::vertex_input::Vertex, Pipeline},
    sync::GpuFuture,
};

//Types and such
pub mod vk {
    // import the correct coordinate space from ultraviolet
    pub use ultraviolet::projection::lh_ydown as projection;

    pub use vulkano::{
        DeviceSize,
        buffer::{
            view::{BufferView, BufferViewCreateInfo},
            Buffer, BufferContents, BufferCreateInfo, BufferUsage,
            Subbuffer,
        },
        command_buffer::{
            allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
            AutoCommandBufferBuilder,
            BufferImageCopy,
            CommandBufferUsage,
            //Image ops
            CopyBufferToImageInfo,
            PrimaryAutoCommandBuffer,
            //Renderpass
            RenderPassBeginInfo,
            SecondaryAutoCommandBuffer,
            SubpassContents,
            ClearColorImageInfo,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
            PersistentDescriptorSet, WriteDescriptorSet,
        },
        device::{
            physical::{PhysicalDevice, PhysicalDeviceType},
            Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo,
            QueueFlags,
        },
        format::{ClearValue, Format},
        image::{
            view::{ImageView, ImageViewCreateInfo},
            AttachmentImage, ImageCreateFlags, ImageDimensions, ImageUsage, ImmutableImage,
            StorageImage, SwapchainImage,
            ImageSubresourceRange, ImageAspects, ImageAccess,
            SampleCount,
        },
        instance::{Instance, InstanceCreateInfo},
        library::VulkanLibrary,
        memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
        pipeline::{
            compute::ComputePipeline,
            graphics::{
                color_blend::{AttachmentBlend, ColorBlendState},
                input_assembly::{InputAssemblyState, PrimitiveTopology},
                rasterization::{CullMode, RasterizationState},
                vertex_input::Vertex,
                viewport::{Scissor, Viewport, ViewportState},
                GraphicsPipeline, GraphicsPipelineBuilder,
                multisample::MultisampleState,
            },
            layout::PipelineLayout,
            PartialStateMode, StateMode,
        },
        render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
        sampler::{ComponentMapping, ComponentSwizzle, Filter, Sampler, SamplerCreateInfo},
        swapchain::{
            acquire_next_image, PresentInfo, PresentMode, Surface, SurfaceInfo, Swapchain,
            SwapchainCreateInfo, SwapchainPresentInfo,
        },
        sync,
        sync::{future::NowFuture, Sharing},
        Version,
    };
}

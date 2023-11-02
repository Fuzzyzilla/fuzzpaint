//Traits
pub use vulkano::{
    pipeline::graphics::vertex_input::VertexDefinition,
    pipeline::{graphics::vertex_input::Vertex, Pipeline},
    sync::GpuFuture,
};

//Types and such
pub mod vk {
    // import the correct coordinate space from ultraviolet
    pub use ultraviolet::projection::lh_ydown as projection;

    pub use vulkano::{
        buffer::{
            view::{BufferView, BufferViewCreateInfo},
            Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
        },
        command_buffer::{
            allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
            AutoCommandBufferBuilder,
            BufferImageCopy,
            ClearColorImageInfo,
            CommandBufferUsage,
            //Image ops
            CopyBufferToImageInfo,
            PrimaryAutoCommandBuffer,
            //Renderpass
            RenderPassBeginInfo,
            SecondaryAutoCommandBuffer,
            SubpassBeginInfo,
            SubpassContents,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator,
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            PersistentDescriptorSet, WriteDescriptorSet,
        },
        device::{
            physical::{PhysicalDevice, PhysicalDeviceType},
            Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo,
            QueueFlags,
        },
        format::{ClearValue, Format},
        image::{
            sampler::{ComponentMapping, ComponentSwizzle, Filter, Sampler, SamplerCreateInfo},
            view::{ImageView, ImageViewCreateInfo},
            Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageSubresourceLayers,
            ImageSubresourceRange, ImageUsage,
        },
        instance::{Instance, InstanceCreateInfo},
        library::VulkanLibrary,
        memory::{
            allocator::{
                AllocationCreateInfo, MemoryAllocatePreference, MemoryTypeFilter,
                StandardMemoryAllocator,
            },
            MemoryPropertyFlags,
        },
        pipeline::{
            compute::{ComputePipeline, ComputePipelineCreateInfo},
            graphics::{
                color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
                input_assembly::{InputAssemblyState, PrimitiveTopology},
                multisample::MultisampleState,
                rasterization::{CullMode, RasterizationState},
                vertex_input::{Vertex, VertexInputState},
                viewport::{Scissor, Viewport, ViewportState},
                GraphicsPipeline, GraphicsPipelineCreateInfo,
            },
            layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
            DynamicState, PipelineShaderStageCreateInfo,
        },
        render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
        shader::{ShaderStages, SpecializationConstant},
        swapchain::{
            acquire_next_image, PresentInfo, PresentMode, Surface, SurfaceInfo, Swapchain,
            SwapchainCreateInfo, SwapchainPresentInfo,
        },
        sync,
        sync::{future::NowFuture, Sharing},
        DeviceSize, Version,
    };
}

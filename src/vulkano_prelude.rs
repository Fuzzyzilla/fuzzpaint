//Traits
pub use vulkano::{
    pipeline::{
        graphics::vertex_input::{Vertex, VertexDefinition},
        Pipeline,
    },
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
            //Image ops
            BlitImageInfo,
            BufferImageCopy,
            ClearColorImageInfo,
            CommandBufferUsage,
            CopyBufferToImageInfo,
            CopyImageToBufferInfo,
            ImageBlit,
            PrimaryAutoCommandBuffer,
            RenderPassBeginInfo,
            RenderingAttachmentInfo,
            //Renderpass
            RenderingInfo,
            SecondaryAutoCommandBuffer,
            SubpassBeginInfo,
            SubpassContents,
            SubpassEndInfo,
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
        format::{ClearValue, Format, FormatFeatures},
        image::{
            sampler::{ComponentMapping, ComponentSwizzle, Filter, Sampler, SamplerCreateInfo},
            view::{ImageView, ImageViewCreateInfo, ImageViewType},
            Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout,
            ImageSubresourceLayers, ImageSubresourceRange, ImageType, ImageUsage,
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
                color_blend::{
                    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState,
                    ColorBlendState,
                },
                input_assembly::{InputAssemblyState, PrimitiveTopology},
                multisample::MultisampleState,
                rasterization::{CullMode, RasterizationState},
                subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
                vertex_input::{Vertex, VertexInputState},
                viewport::{Scissor, Viewport, ViewportState},
                GraphicsPipeline, GraphicsPipelineCreateInfo,
            },
            layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
            DynamicState, PipelineBindPoint, PipelineShaderStageCreateInfo,
        },
        render_pass::{
            AttachmentLoadOp, AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass,
            Subpass,
        },
        shader::{ShaderStages, SpecializationConstant},
        swapchain::{
            acquire_next_image, PresentInfo, PresentMode, Surface, SurfaceInfo, Swapchain,
            SwapchainCreateInfo, SwapchainPresentInfo,
        },
        sync::{
            self,
            future::{FenceSignalFuture, NowFuture, SemaphoreSignalFuture},
            Sharing,
        },
        DeviceSize, Validated, Version, VulkanError,
    };
}

use crate::vulkano_prelude::*;
use std::{fmt::Debug, sync::Arc};

use fuzzpaint_core::blend::{Blend, BlendMode};
use vulkano::VulkanObject;

type BlendCompiler =
    fn(Arc<vk::Device>) -> Result<Arc<vk::ShaderModule>, vk::Validated<vk::VulkanError>>;

/// Different creation parameters of blend math, in order of preference.
/// E.g., a simple blend should be the preferable implementation over a compute, if possible.
///
/// Use to create a [`CompiledBlend`]
enum BlendLogic {
    /// The blend can be represented as a standard blend function. Hardware accelerated, pipelinable, and coherent. Nice :3
    Simple(vk::AttachmentBlend),
    /// Provide a Load function for a shader to compute `A (+) B` for arbitrary blend logic. Still pipeliend, but noncoherent 3:
    /// (except perhaps if the device has fragment interlock, todo!)
    Arbitrary(fn(Arc<vk::Device>) -> Result<Arc<vk::ShaderModule>, vk::Validated<vk::VulkanError>>),
    // other ideas for implementing, with various stages of slowness (still better than loopback fragment):
    // * we have blend constants and dual src blend at our disposal! These are free, but latter requires hardware support.
    // * Additionally, we could use the fragment to do arbitrary transforms to `Src` but still use regular blend eqs
    // not sure what modes would use these other techniques, yet.
}
impl BlendLogic {
    /// Get the logic needed to perform a blend.
    fn of(blend: BlendMode, clip: bool) -> BlendLogic {
        use vk::{AttachmentBlend, BlendFactor};

        // When writing new operations here, remember that:
        // * if the result alpha is zero, RGB must also be zero.
        // * if the dst alpha is zero and not clip, then use src directly. -- maybe?
        // * if the src alpha is zero, dst should be unchanged.
        // * if clip, then the dst alpha should be unchanged.
        // This serves as a decent litmus test for if something is horribly borked.

        // The logic for just the alpha channel, color channels must be overridden.
        let alpha_channel = if clip {
            AttachmentBlend {
                // Keep alpha from dest.
                src_alpha_blend_factor: BlendFactor::Zero,
                dst_alpha_blend_factor: BlendFactor::One,
                ..Default::default()
            }
        } else {
            AttachmentBlend {
                src_alpha_blend_factor: BlendFactor::One,
                dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..Default::default()
            }
        };

        match (blend, clip) {
            (BlendMode::Normal, false) => BlendLogic::Simple(AttachmentBlend {
                src_color_blend_factor: BlendFactor::One,
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..alpha_channel
            }),
            (BlendMode::Normal, true) => BlendLogic::Simple(AttachmentBlend {
                src_color_blend_factor: BlendFactor::DstAlpha,
                // Should this be (1 - (Sa * Da))? this can't be represented, if so.
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..alpha_channel
            }),
            (BlendMode::Add, false) => BlendLogic::Simple(AttachmentBlend {
                src_color_blend_factor: BlendFactor::One,
                dst_color_blend_factor: BlendFactor::One,
                ..alpha_channel
            }),
            (BlendMode::Add, true) => BlendLogic::Simple(AttachmentBlend {
                src_color_blend_factor: BlendFactor::DstAlpha,
                dst_color_blend_factor: BlendFactor::One,
                ..alpha_channel
            }),
            (BlendMode::Multiply, false) => BlendLogic::Simple(AttachmentBlend {
                // Not quite right, always results in black on a transparent background.
                // Funny! No mul op, so instead "Normal" op with the left side having a multiply factor.
                src_color_blend_factor: BlendFactor::DstColor,
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..alpha_channel
            }),
            (BlendMode::Multiply, true) => BlendLogic::Arbitrary(shaders::noncoherent_test::load),
            _ => unimplemented!(),
        }
    }
}

/// Different implementations of blending logic, from most optimal to least.
#[derive(Clone, PartialEq, Eq)]
enum CompiledBlend {
    /// A pipeline which takes an image and outputs it directly to the entire viewport.
    /// Inputs: attachment 0 set 0 binding 0 - RGBA input attachment.
    /// Outputs: location 0: RGBA
    SimpleCoherent(Arc<vk::GraphicsPipeline>),
    /// A pipeline which performs a non-coherent "loopback" operation
    Loopback(Arc<vk::GraphicsPipeline>),
}

mod shaders {
    #[derive(
        PartialEq,
        Copy,
        Clone,
        bytemuck::NoUninit,
        bytemuck::Zeroable,
        vulkano::buffer::BufferContents,
    )]
    #[repr(C)]
    pub struct Constants {
        pub solid_color: [f32; 4],
        // Safety: VkBool32, must be 0 or 1
        is_solid: u32,
    }
    impl Constants {
        pub fn new_solid(solid_color: fuzzpaint_core::color::Color) -> Self {
            Self {
                solid_color: solid_color.as_array(),
                is_solid: true.into(),
            }
        }
        pub fn new_image(alpha: f32) -> Self {
            Self {
                solid_color: [0.0, 0.0, 0.0, alpha],
                is_solid: false.into(),
            }
        }
    }

    /// Vertex shader that fills the viewport and provides UV at location 0.
    /// Call with three vertices.
    pub mod fullscreen_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                    #version 460

                    layout(location = 0) out vec2 uv;

                    void main() {
                        // fullscreen tri
                        vec2 pos = vec2(
                            float((gl_VertexIndex & 1) * 4 - 1),
                            float((gl_VertexIndex & 2) * 2 - 1)
                        );

                        uv = pos / 2.0 + 0.5;
                        gl_Position = vec4(
                            pos,
                            0.0,
                            1.0
                        );
                    }
                "
        }
    }
    /// A shader that does nothing special, filling the viewport with either image at set 0, binding 0, or solid color if specified.
    /// Use fixed function blend math to achieve blend effects. This is the most efficient blend method!
    pub mod coherent_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                    #version 460
                    layout(set = 0, binding = 0) uniform sampler2D src;
                    // reserved, undefined behavior to read from.
                    // layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput dont_use;

                    layout(push_constant) uniform Constants {
                        // Solid color constant, otherwise just the alpha is used as global multiplier.
                        vec4 solid_color;
                        // True if the shader should 'sample' from `solid_color` instead of the image.
                        // UB to read image if this is set.
                        bool is_solid;
                    };

                    layout(location = 0) in vec2 uv;
                    layout(location = 0) out vec4 color;

                    void main() {
                        color = is_solid ? solid_color : (texture(src, uv) * solid_color.a);
                    }
                "
        }
    }
    pub mod noncoherent_test {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                    #version 460
                    layout(set = 0, binding = 0) uniform sampler2D src;
                    layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput dst;

                    layout(push_constant) uniform Constants {
                        // Solid color constant, otherwise just the alpha is used as global multiplier.
                        vec4 solid_color;
                        // True if the shader should 'sample' from `solid_color` instead of the image.
                        // UB to read image if this is set.
                        bool is_solid;
                    };

                    layout(location = 0) in vec2 uv;
                    layout(location = 0) out vec4 color;

                    void main() {
                        color = is_solid ? solid_color.gbra : (texture(src, uv).gbra * solid_color.a);
                    }
                "
        }
    }
}

/// A handle to the result of a previous blend command, can be submitted to the device to
/// execute the work or added into another blend operation as an operand.
///
/// If this handle is dropped or forgotten, it is safe to assume the resources it describes
/// are never accessed.
pub struct BlendInvocationHandle {
    // First item = first operation
    operations: Vec<(BlendImageSource, Blend)>,
    clear_destination: bool,
    destination_image: Arc<vk::ImageView>,
}
/// Source for a blend operation.
pub enum BlendImageSource {
    /// An image usable without synchronization.
    /// Image must be immediately ready for use at the time of blend submission,
    /// and must not be written until the blend operation is complete.
    Immediate(Arc<vk::ImageView>),
    /*
    // For simplicity this is left unimplemented. In order to implement this, a lot of fighting with
    // vulkano sync might be necessary, or I can lower down with the Ash crate.

    /// The given image will be ready after this semaphore becomes signalled.
    /// Image must not be written until the blend operation is complete.
    AfterSemaphore {
        image: Arc<vk::ImageView>,
        semaphore: vk::SemaphoreSignalFuture<Box<dyn GpuFuture>>,
    },*/
    /// The image comes from a previous blend operation.
    /// Synchronization and submission will be handled automatically.
    BlendInvocation(BlendInvocationHandle),
    SolidColor(fuzzpaint_core::color::Color),
}
impl From<BlendInvocationHandle> for BlendImageSource {
    fn from(value: BlendInvocationHandle) -> Self {
        Self::BlendInvocation(value)
    }
}
impl BlendImageSource {
    fn view(&self) -> Option<&Arc<vk::ImageView>> {
        match self {
            BlendImageSource::Immediate(image)
            | BlendImageSource::BlendInvocation(BlendInvocationHandle {
                destination_image: image,
                ..
            }) => Some(image),
            Self::SolidColor(_) => None,
        }
    }
}

/// Checks if `src` requires access to the same subresource as `dest`.
fn does_alias(dest: &vk::ImageView, src: &BlendImageSource) -> bool {
    // The simplest programming problems make my brain overheat
    /// Checks if two ranges contain any elements in common.
    fn ranges_intersect<T: Ord>(a: std::ops::Range<T>, b: std::ops::Range<T>) -> bool {
        // these ranges are provided by vulkano, so we hope it upholds this invariant.
        debug_assert!(a.start <= a.end);
        debug_assert!(b.start <= b.end);
        // Either range empty == never intersects
        !(a.is_empty() || b.is_empty()) &&
                // Check a inside b
                    (a.start >= b.start && a.start < b.end
                    || a.end > b.start && a.end <= b.end
                // Or, check b inside a
                    || b.start >= a.start && b.start < a.end
                    || b.end > a.start && b.end <= a.end)
    }

    let Some(src) = src.view() else {
        // No memory usage, no aliasing!
        return false;
    };
    if dest.image() == src.image() {
        // Compare subresources
        let dest = dest.subresource_range();
        let src = src.subresource_range();

        // Array overlaps, mips overlap, AND aspect overlaps? then aliases!
        ranges_intersect(dest.array_layers.clone(), src.array_layers.clone())
            && ranges_intersect(dest.mip_levels.clone(), src.mip_levels.clone())
            && dest.aspects.intersects(src.aspects)
    } else {
        false
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ImageSourceError {
    #[error("source image aliases the background image")]
    AlisesBackground,
}
pub struct BlendInvocationBuilder {
    clear_destination: bool,
    destination_image: Arc<vk::ImageView>,
    // Top of list = first operation.
    operations: Vec<(BlendImageSource, Blend)>,
}
impl BlendInvocationBuilder {
    /// Blend the given image onto the cumulative results of all previous blend operations.
    pub fn then_blend(
        &mut self,
        image: BlendImageSource,
        mode: Blend,
    ) -> Result<(), ImageSourceError> {
        if does_alias(&self.destination_image, &image) {
            Err(ImageSourceError::AlisesBackground)
        } else {
            self.operations.push((image, mode));
            Ok(())
        }
    }
    /// Reverse the order of the blend operations. The background will remain the background.
    /// Handles are treated as one blend unit, and their contents are not reversed in this operation.
    pub fn reverse(&mut self) {
        self.operations.reverse();
    }
    /// Build the invocation. This handle can be used in other blend invocations as a source,
    /// or it may be provided to [`BlendEngine::submit`] to begin device execution of the blend operation.
    ///
    /// If instead this result is discarded, it is safe to assume the resources described in this operation
    /// are never accessed.
    #[must_use = "blend handle must be submitted to the engine for blending to occur"]
    pub fn build(self) -> BlendInvocationHandle {
        BlendInvocationHandle {
            operations: self.operations,
            clear_destination: self.clear_destination,
            destination_image: self.destination_image,
        }
    }
}
use crate::vk;
pub struct BlendEngine {
    device: Arc<vk::Device>,
    workgroup_size: (u32, u32),
    // Based on chosen workgroup_size, device's max workgroup count, and device image size.
    max_image_size: (u32, u32),
    /// Layout for all blend operations.
    feedback_layout: Arc<vk::PipelineLayout>,
    /// Spawns fragments across the entire viewport. No outputs.
    fullscreen_vert: vk::EntryPoint,
    /// Passes fragments from input attachment 0, set 0, binding 0, unchanged.
    coherent_frag: vk::EntryPoint,
    /// Pipes need a renderpass to build against. We don't have the pass at the time of pipe compilation, tho!
    feedback_pass: Arc<vk::RenderPass>,
    /// (mode, clip) -> prepared blend pipe.
    mode_pipelines: hashbrown::HashMap<(BlendMode, bool), CompiledBlend>,
    /// Data needed to specify self-dependency barrier.
    self_dependency_flags: vk::SubpassDependency,
}
impl BlendEngine {
    /// Get or compile the blend logic for the given mode.
    pub fn lazy_blend_pipe(
        &mut self,
        mode: BlendMode,
        clip: bool,
    ) -> anyhow::Result<&CompiledBlend> {
        match self.mode_pipelines.entry((mode, clip)) {
            hashbrown::hash_map::Entry::Occupied(o) => Ok(o.into_mut()),
            hashbrown::hash_map::Entry::Vacant(v) => {
                // Fetch the equation
                let logic = BlendLogic::of(mode, clip);
                // Compile it!
                match logic {
                    BlendLogic::Simple(equation) => {
                        let pipe = vk::GraphicsPipeline::new(
                            self.device.clone(),
                            None,
                            vulkano::pipeline::graphics::GraphicsPipelineCreateInfo {
                                stages: smallvec::smallvec![
                                    vk::PipelineShaderStageCreateInfo::new(
                                        self.fullscreen_vert.clone()
                                    ),
                                    vk::PipelineShaderStageCreateInfo::new(
                                        self.coherent_frag.clone()
                                    )
                                ],
                                // Data generated by vertex iteself.
                                vertex_input_state: Some(vk::VertexInputState::new()),
                                input_assembly_state: Some(vk::InputAssemblyState::default()),
                                rasterization_state: Some(vk::RasterizationState::default()),
                                // Pass the requested equation directly!
                                color_blend_state: Some(vk::ColorBlendState {
                                    attachments: vec![vk::ColorBlendAttachmentState {
                                        blend: Some(equation),
                                        ..Default::default()
                                    }],
                                    ..Default::default()
                                }),
                                multisample_state: Some(vk::MultisampleState::default()),
                                // Viewport dynamic, scissor irrelevant.
                                viewport_state: Some(vk::ViewportState::default()),
                                dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                                subpass: Some(self.feedback_pass.clone().first_subpass().into()),
                                ..vulkano::pipeline::graphics::GraphicsPipelineCreateInfo::layout(
                                    self.feedback_layout.clone(),
                                )
                            },
                        )?;

                        Ok(v.insert(CompiledBlend::SimpleCoherent(pipe)))
                    }
                    BlendLogic::Arbitrary(load) => {
                        let shader = load(self.device.clone())?
                            .entry_point("main")
                            .ok_or_else(|| anyhow::anyhow!("entry point `main` not found"))?;

                        let pipe = vk::GraphicsPipeline::new(
                            self.device.clone(),
                            None,
                            vulkano::pipeline::graphics::GraphicsPipelineCreateInfo {
                                stages: smallvec::smallvec![
                                    vk::PipelineShaderStageCreateInfo::new(
                                        self.fullscreen_vert.clone()
                                    ),
                                    vk::PipelineShaderStageCreateInfo::new(shader)
                                ],
                                // Data generated by vertex iteself.
                                vertex_input_state: Some(vk::VertexInputState::new()),
                                input_assembly_state: Some(vk::InputAssemblyState::default()),
                                rasterization_state: Some(vk::RasterizationState::default()),
                                color_blend_state: Some(vk::ColorBlendState {
                                    // No blend equation, the shader is to do alllllll the work.
                                    attachments: vec![vk::ColorBlendAttachmentState::default()],
                                    ..Default::default()
                                }),
                                multisample_state: Some(vk::MultisampleState::default()),
                                // Viewport dynamic, scissor irrelevant.
                                viewport_state: Some(vk::ViewportState::default()),
                                dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                                subpass: Some(self.feedback_pass.clone().first_subpass().into()),
                                ..vulkano::pipeline::graphics::GraphicsPipelineCreateInfo::layout(
                                    self.feedback_layout.clone(),
                                )
                            },
                        )?;

                        Ok(v.insert(CompiledBlend::Loopback(pipe)))
                    }
                }
            }
        }
    }
    /// Get the pipeline for a blend mode, or `None` if it has not been compiled yet.
    pub fn blend_pipe(&self, mode: BlendMode, clip: bool) -> Option<&CompiledBlend> {
        self.mode_pipelines.get(&(mode, clip))
    }
    pub fn new(context: &crate::render_device::RenderContext) -> anyhow::Result<Self> {
        let device = context.device();
        // compute the workgroup size, specified as specialization constants
        let properties = device.physical_device().properties();
        let workgroup_size = {
            // Todo: better alg for this lol
            let largest_square = f64::from(properties.max_compute_work_group_invocations)
                .sqrt()
                .floor() as u32;
            let largest_square = largest_square
                .min(properties.max_compute_work_group_size[0])
                .min(properties.max_compute_work_group_size[1]);
            (largest_square, largest_square)
        };
        // Max image size, based on max num of workgroups and chosen workgroup size and device max storage image size
        let max_image_size = (
            (workgroup_size.0)
                .saturating_mul(properties.max_compute_work_group_count[0])
                .min(properties.max_image_dimension2_d),
            (workgroup_size.1)
                .saturating_mul(properties.max_compute_work_group_count[1])
                .min(properties.max_image_dimension2_d),
        );
        log::info!(
            "Blend workgroup size: {}x{}x1. Max image size {}x{}",
            workgroup_size.0,
            workgroup_size.1,
            max_image_size.0,
            max_image_size.1
        );

        // Build fixed layout for "simple coherent" blend processes.
        // Consists of a fixed sampler and one sampled image.
        let nearest_sampler = vk::Sampler::new(device.clone(), vk::SamplerCreateInfo::default())?;
        let mut sampler_bindings = std::collections::BTreeMap::new();
        sampler_bindings.insert(
            0,
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                immutable_samplers: vec![nearest_sampler],
                stages: vk::ShaderStages::FRAGMENT,
                ..vk::DescriptorSetLayoutBinding::descriptor_type(
                    vk::DescriptorType::CombinedImageSampler,
                )
            },
        );
        let mut input_attachment_bindings = std::collections::BTreeMap::new();
        input_attachment_bindings.insert(
            0,
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                immutable_samplers: vec![],
                stages: vk::ShaderStages::FRAGMENT,
                ..vk::DescriptorSetLayoutBinding::descriptor_type(
                    vk::DescriptorType::InputAttachment,
                )
            },
        );

        let sampler_layout = vk::DescriptorSetLayout::new(
            device.clone(),
            vk::DescriptorSetLayoutCreateInfo {
                bindings: sampler_bindings,
                ..Default::default()
            },
        )?;
        let input_attachment_layout = vk::DescriptorSetLayout::new(
            device.clone(),
            vk::DescriptorSetLayoutCreateInfo {
                bindings: input_attachment_bindings,
                ..Default::default()
            },
        )?;

        let alpha_push_constant = vk::PushConstantRange {
            offset: 0,
            // RGBA32F + Bool32
            size: 20,
            stages: vk::ShaderStages::FRAGMENT,
        };

        // These two layouts, for both kinds of pipe, are descriptor set compatible.
        let feedback_layout = vk::PipelineLayout::new(
            device.clone(),
            vk::PipelineLayoutCreateInfo {
                set_layouts: vec![sampler_layout, input_attachment_layout],
                push_constant_ranges: vec![alpha_push_constant],
                ..vk::PipelineLayoutCreateInfo::default()
            },
        )?;

        // Precompile these because we're gonna use em a lot!
        let fullscreen_vert = shaders::fullscreen_vert::load(device.clone())?
            .entry_point("main")
            .unwrap();
        // All simple coherent pipes share a fragment shader because the logic happens in
        // fixed-function blend hardware.
        let coherent_frag = shaders::coherent_frag::load(device.clone())?
            .entry_point("main")
            .unwrap();

        // allow self-dependencies (CmdPipelineBarrier) within subpass 0 so that we may use feedback loops.
        // ONLY barriers with these *exact* access/stages are allowed, and MUST be by-region :O
        let self_dependency_flags = vk::SubpassDependency {
            src_subpass: Some(0),
            src_stages: vk::sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            src_access: vk::sync::AccessFlags::COLOR_ATTACHMENT_WRITE,

            dst_subpass: Some(0),
            dst_stages: vk::sync::PipelineStages::FRAGMENT_SHADER,
            dst_access: vk::sync::AccessFlags::INPUT_ATTACHMENT_READ,

            dependency_flags: vk::sync::DependencyFlags::BY_REGION,
            ..Default::default()
        };

        let feedback_pass = {
            let feedback_attachment = vk::AttachmentDescription {
                // THIS SHOULD BE `ImageLayout::AttachmentFeedbackLoopOptimal`, vulkano doesn't support that yet.
                initial_layout: vk::ImageLayout::General,
                final_layout: vk::ImageLayout::General,
                format: crate::DOCUMENT_FORMAT,
                load_op: vk::AttachmentLoadOp::Load,
                // THIS SHOULD BE `StoreOp::None`, vulkano doesn't support that yet.
                // This is an input-only attachment and is not modified.
                store_op: vk::AttachmentStoreOp::Store,
                ..Default::default()
            };
            let input_reference = vk::AttachmentReference {
                aspects: vk::ImageAspects::COLOR,
                attachment: 0,
                layout: vk::ImageLayout::General,
                ..Default::default()
            };
            let output_reference = vk::AttachmentReference {
                aspects: vk::ImageAspects::empty(),
                attachment: 0,
                layout: vk::ImageLayout::General,
                ..Default::default()
            };
            let subpass = vk::SubpassDescription {
                color_attachments: vec![Some(output_reference)],
                input_attachments: vec![Some(input_reference)],
                ..Default::default()
            };

            let external_dependencies = vk::SubpassDependency {
                // We don't know what will be performed outside. I should come back and make this well-defined,
                // but use an extremely strong dependency instead :P
                src_subpass: None, // External
                src_stages: vk::sync::PipelineStages::BOTTOM_OF_PIPE,
                src_access: vk::sync::AccessFlags::MEMORY_WRITE
                    | vk::sync::AccessFlags::MEMORY_READ,

                dst_subpass: Some(0),
                // Before we read or write the images in any way within subpass.
                dst_stages: vk::sync::PipelineStages::FRAGMENT_SHADER
                    | vk::sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                dst_access: vk::sync::AccessFlags::INPUT_ATTACHMENT_READ
                    | vk::sync::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::sync::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::sync::AccessFlags::SHADER_READ,

                // Does this even make sense for an external dep? Oh well, it's allowed UwU
                dependency_flags: vk::sync::DependencyFlags::BY_REGION,
                ..Default::default()
            };

            vk::RenderPass::new(
                device.clone(),
                vk::RenderPassCreateInfo {
                    attachments: vec![feedback_attachment],
                    subpasses: vec![subpass],
                    dependencies: vec![external_dependencies, self_dependency_flags.clone()],
                    ..Default::default()
                },
            )?
        };

        let mut this = Self {
            device: device.clone(),
            max_image_size,
            workgroup_size,
            feedback_layout,
            fullscreen_vert,
            coherent_frag,
            feedback_pass,
            mode_pipelines: hashbrown::HashMap::new(),
            self_dependency_flags,
        };

        for (mode, clip) in [
            (BlendMode::Normal, false),
            (BlendMode::Normal, true),
            (BlendMode::Add, false),
            (BlendMode::Add, true),
            (BlendMode::Multiply, false),
            (BlendMode::Multiply, true),
        ] {
            this.lazy_blend_pipe(mode, clip)?;
        }

        Ok(this)
    }
    /// Begin a blend operation with the engine.
    /// The destination image must be available at the time of calling `submit`.
    #[must_use = "use the result to build an operation"]
    pub fn start(
        &self,
        destination_image: Arc<vk::ImageView>,
        clear_destination: bool,
    ) -> BlendInvocationBuilder {
        BlendInvocationBuilder {
            clear_destination,
            destination_image,
            operations: Vec::new(),
        }
    }
    /// Submits the work of one (or several composite) blend operations.
    /// This must be called for the work defined by the builders to actually take place.
    ///
    /// The returned `Fence` represents a signal after which the image is ready.
    ///
    /// # Safety
    /// * No images lifetimes are maintained.
    /// * All input images must be externally synchronized and retain shared access during the operation.
    /// * All output images must be externally synchronized and retain exclusive access during the operation.
    pub unsafe fn submit(
        &self,
        context: &crate::render_device::RenderContext,
        handle: BlendInvocationHandle,
    ) -> anyhow::Result<()> {
        use vulkano::command_buffer::sys::UnsafeCommandBuffer;
        // Per the vulkan specification, operations that result in binary semaphore signals
        // must be submitted prior to the operations that wait on those semaphores.
        // Thus, we must traverse the blend tree from the bottom-up.

        // Current implementation is non-optimal but that's not the goal at this point uwu

        // Array of Arrays of buffers. Each subsequent array is the next level of the tree, and will be
        // submitted from back to front with semaphores between.
        let mut layers: Vec<Vec<UnsafeCommandBuffer>> = Vec::new();
        // After the current walkthrough, which tasks to walk through next?
        let mut next_layer: Vec<BlendInvocationHandle> = vec![handle];

        while !next_layer.is_empty() {
            let mut this_layer: Vec<UnsafeCommandBuffer> = Vec::new();
            // Take all the next tasks, and build them. Any subtasks referenced will be added back to the queue.
            for task in std::mem::take(&mut next_layer) {
                // Build and push the commands for this operation
                // (commands can be build prior to images being ready)
                let cb = unsafe {
                    self.make_blend_commands(
                        context,
                        task.destination_image,
                        task.clear_destination,
                        &task.operations,
                    )?
                };
                this_layer.push(cb);

                // Append subtasks to the queue to be processed
                next_layer.extend(
                    // Move all subtasks
                    task.operations
                        .into_iter()
                        .filter_map(|(image, _)| match image {
                            BlendImageSource::BlendInvocation(inv) => Some(inv),
                            BlendImageSource::Immediate(_) => None,
                            BlendImageSource::SolidColor(_) => None,
                        }),
                );
            }

            layers.push(this_layer);
        }

        // now, layers should be filled, from back to front, with commands in the order they must be
        // executed for proper blending.
        for buffers in layers.into_iter().rev() {
            // `buffers` may all be executed in parallel without sync.
            // After all of these, the next iteration can proceed following a barrier.
            let graphics = context.queues().graphics().queue();
            let graphics_handle = graphics.handle();
            let pfn_submit = context.device().fns().v1_0.queue_submit;

            let raw_buffers = buffers
                .iter()
                .map(vulkano::VulkanObject::handle)
                .collect::<Vec<_>>();
            // No need to wait semaphores since each buffer contains an overkill barrier. Fixme!
            let submission = ash::vk::SubmitInfo::builder().command_buffers(&raw_buffers);

            // Synchronize queue externally!
            graphics
                .with(|_lock| unsafe {
                    (pfn_submit)(
                        graphics_handle,
                        1,
                        &submission.build(),
                        ash::vk::Fence::null(),
                    )
                })
                .result()?;
        }

        // Fixme: fence instead of flush.
        context
            .queues()
            .graphics()
            .queue()
            .with(|mut q| q.wait_idle())?;
        Ok(())
    }
    /// Layers will be blended, from front to back of the slice, into a mutable background.
    ///
    /// Any [`BlendImageSource::BlendInvocation`] items are assumed to have been rendered already, thus
    /// only their destination images are taken into account.
    ///
    /// # Safety
    /// * `background` must not be aliased by any image view of `layers`.
    /// * Returned command buffer assumes exclusive access to `destination_image` and shared
    ///   access to all layer image views after `BOTTOM_OF_PIPE` in the previous submission.
    /// * No lifetime checking is done.
    unsafe fn make_blend_commands(
        &self,
        context: &crate::render_device::RenderContext,
        destination_image: Arc<vk::ImageView>,
        clear_destination: bool,
        layers: &[(BlendImageSource, Blend)],
    ) -> anyhow::Result<vulkano::command_buffer::sys::UnsafeCommandBuffer> {
        // Unfortunately, we *need* to use unsafe command buffer here. There is currently
        // an error with Auto command buffer, where pipeline barriers are not inserted correctly
        // between `CompiledBlend::Loopback` pipes leading to race conditions, and there is no way to do it manually
        // other than to lower to `unsafe`!
        unsafe {
            let mut commands = vulkano::command_buffer::sys::UnsafeCommandBufferBuilder::new(
                context.allocators().command_buffer(),
                context.queues().graphics().idx(),
                vulkano::command_buffer::CommandBufferLevel::Primary,
                vulkano::command_buffer::sys::CommandBufferBeginInfo {
                    usage: vk::CommandBufferUsage::OneTimeSubmit,
                    inheritance_info: None,
                    ..Default::default()
                },
            )?;

            // Still honor the request to clear the image if no layers are provided.
            // In the not empty case, it's handled by a clear_attachment instead.
            if layers.is_empty() {
                if clear_destination {
                    // Wait for exclusive access...

                    // We need to make sure that the previous command buffer isn't accessing the images we're about to use.
                    let global_barrier = {
                        // We could make a barrier-per-image... Seems pricey for the driver to handle.
                        let barrier = vulkano::sync::MemoryBarrier {
                            // We can't know what happened before, so source is maximally strong
                            src_access: vulkano::sync::AccessFlags::MEMORY_WRITE
                                | vulkano::sync::AccessFlags::MEMORY_READ,
                            src_stages: vulkano::sync::PipelineStages::BOTTOM_OF_PIPE,

                            // We *do* know what we're gonna do, though!
                            // We can't clear until all red/writes from before complete.
                            dst_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                            // Only need CLEAR but that's extension...
                            dst_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,

                            ..Default::default()
                        };

                        vulkano::sync::DependencyInfo {
                            dependency_flags: vulkano::sync::DependencyFlags::BY_REGION,
                            memory_barriers: smallvec::smallvec![barrier],
                            ..Default::default()
                        }
                    };

                    commands.pipeline_barrier(&global_barrier)?;
                    commands.clear_color_image(&vk::ClearColorImageInfo {
                        clear_value: [0.0; 4].into(),
                        regions: smallvec::smallvec![destination_image.subresource_range().clone(),],
                        ..vk::ClearColorImageInfo::image(destination_image.image().clone())
                    })?;
                }
                return Ok(commands.build()?);
            }

            commands
                // Implicit external barrier here due to external dependency in the subpass!
                .begin_render_pass(
                    &vk::RenderPassBeginInfo {
                        render_pass: self.feedback_pass.clone(),
                        clear_values: vec![None], //vec![clear_destination.then_some([0.0; 4].into())],
                        // Todo: Cache. Framebuffers are not trivial to construct.
                        ..vk::RenderPassBeginInfo::framebuffer(vk::Framebuffer::new(
                            self.feedback_pass.clone(),
                            vk::FramebufferCreateInfo {
                                attachments: vec![destination_image.clone()],
                                extent: [
                                    destination_image.image().extent()[0],
                                    destination_image.image().extent()[1],
                                ],
                                ..Default::default()
                            },
                        )?)
                    },
                    &vk::SubpassBeginInfo::default(),
                )?
                .set_viewport(
                    0,
                    &[vk::Viewport {
                        depth_range: 0.0..=1.0,
                        offset: [0.0; 2],
                        extent: [
                            destination_image.image().extent()[0] as f32,
                            destination_image.image().extent()[1] as f32,
                        ],
                    }],
                )?;

            // Bind the input attachment at set 1
            let feedback_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                self.feedback_layout.set_layouts()[1].clone(),
                [vk::WriteDescriptorSet::image_view(
                    0,
                    destination_image.clone(),
                )],
                [],
            )?;
            commands.bind_descriptor_sets(
                vk::PipelineBindPoint::Graphics,
                &self.feedback_layout,
                1,
                &[feedback_set.into()],
            )?;

            // Self-dependency barrier. MUST be a subset of the pipeline dependency flags.
            let feedback_barrier = {
                let destination_image_barrier = vulkano::sync::ImageMemoryBarrier {
                    src_access: self.self_dependency_flags.src_access,
                    src_stages: self.self_dependency_flags.src_stages,

                    dst_access: self.self_dependency_flags.dst_access,
                    dst_stages: self.self_dependency_flags.dst_stages,

                    new_layout: vk::ImageLayout::General,
                    old_layout: vk::ImageLayout::General,

                    queue_family_ownership_transfer: None,
                    subresource_range: vk::ImageSubresourceRange {
                        array_layers: 0..1,
                        aspects: vk::ImageAspects::COLOR,
                        mip_levels: 0..1,
                    },
                    ..vulkano::sync::ImageMemoryBarrier::image(destination_image.image().clone())
                };
                vulkano::sync::DependencyInfo {
                    dependency_flags: self.self_dependency_flags.dependency_flags,
                    image_memory_barriers: smallvec::smallvec![destination_image_barrier],
                    ..Default::default()
                }
            };

            if clear_destination {
                // Clear before any blends occur. This is properly barrier'd
                // by setting write to `true` initially.
                commands.clear_attachments(
                    &[vulkano::command_buffer::ClearAttachment::Color {
                        color_attachment: 0,
                        clear_value: [0.0; 4].into(),
                    }],
                    &[vulkano::command_buffer::ClearRect {
                        offset: [0; 2],
                        extent: [
                            destination_image.image().extent()[0],
                            destination_image.image().extent()[1],
                        ],
                        array_layers: 0..1,
                    }],
                )?;
            }

            // Whether we just wrote to the destination image on the last loop.
            // Clear counts as a write!
            let mut had_write = clear_destination;
            // Whether the current pipe will read the destination image. It is UB
            // for a read to occur after a write without a barrier.
            let mut will_read = false;
            let mut last_mode = None;

            let mut last_constants = None;
            for (image_src, blend) in layers {
                let Blend {
                    mode,
                    alpha_clip,
                    opacity,
                } = *blend;
                // bind a new pipeline if changed from last iter
                if last_mode != Some((mode, alpha_clip)) {
                    let Some(pipe) = self.blend_pipe(mode, alpha_clip).cloned() else {
                        anyhow::bail!("Blend mode {:?} unsupported", mode)
                    };
                    match &pipe {
                        CompiledBlend::SimpleCoherent(pipe) => {
                            will_read = false;
                            commands.bind_pipeline_graphics(pipe)?;
                        }
                        CompiledBlend::Loopback(pipe) => {
                            will_read = true;
                            commands.bind_pipeline_graphics(pipe)?;
                        }
                    }
                    last_mode = Some((mode, alpha_clip));
                }

                // Set the image. The sampler is bundled magically by being baked into the layout itself.
                match image_src {
                    BlendImageSource::BlendInvocation(BlendInvocationHandle {
                        destination_image: view,
                        ..
                    })
                    | BlendImageSource::Immediate(view) => {
                        let sampler_set = vk::PersistentDescriptorSet::new(
                            context.allocators().descriptor_set(),
                            self.feedback_layout.set_layouts()[0].clone(),
                            [vk::WriteDescriptorSet::image_view(0, view.clone())],
                            [],
                        )?;
                        commands.bind_descriptor_sets(
                            vk::PipelineBindPoint::Graphics,
                            &self.feedback_layout,
                            0,
                            &[sampler_set.into()],
                        )?;

                        let constants = shaders::Constants::new_image(opacity);
                        if last_constants != Some(constants) {
                            commands.push_constants(&self.feedback_layout, 0, &constants)?;
                            last_constants = Some(constants);
                        }
                    }
                    &BlendImageSource::SolidColor(color) => {
                        let constants = shaders::Constants::new_solid(color.alpha_multipy(
                            fuzzpaint_core::util::FiniteF32::new(opacity).unwrap_or_default(),
                        ));

                        if last_constants != Some(constants) {
                            commands.push_constants(&self.feedback_layout, 0, &constants)?;
                            last_constants = Some(constants);
                        }
                    }
                }

                if had_write && will_read {
                    // Insert a barrier to ensure last loop's write completes before this loop's read.
                    commands.pipeline_barrier(&feedback_barrier)?;
                }

                commands.draw(3, 1, 0, 0)?;
                had_write = true;
            }
            commands.end_render_pass(&vk::SubpassEndInfo::default())?;
            Ok(commands.build()?)
        }
    }
}

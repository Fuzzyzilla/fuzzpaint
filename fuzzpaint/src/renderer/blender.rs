use crate::vulkano_prelude::*;
use std::{fmt::Debug, sync::Arc};

use fuzzpaint_core::blend::{Blend, BlendMode};

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
    /// A shader that does nothing special, filling the viewport with image at set 0, binding 0.
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
                        float alpha;
                    };

                    layout(location = 0) in vec2 uv;
                    layout(location = 0) out vec4 color;

                    void main() {
                        color = texture(src, uv) * alpha;
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
                        float alpha;
                    };

                    layout(location = 0) in vec2 uv;
                    layout(location = 0) out vec4 color;

                    void main() {
                        color = subpassLoad(dst).gbra;
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
}
impl From<BlendInvocationHandle> for BlendImageSource {
    fn from(value: BlendInvocationHandle) -> Self {
        Self::BlendInvocation(value)
    }
}
impl BlendImageSource {
    fn view(&self) -> &Arc<vk::ImageView> {
        match self {
            BlendImageSource::Immediate(image)
            | BlendImageSource::BlendInvocation(BlendInvocationHandle {
                destination_image: image,
                ..
            }) => image,
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

    let src = src.view();
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

                        Ok(v.insert(CompiledBlend::SimpleCoherent(pipe)))
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
            size: std::mem::size_of::<f32>() as u32,
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
            vk::RenderPass::new(
                device.clone(),
                vk::RenderPassCreateInfo {
                    attachments: vec![feedback_attachment],
                    subpasses: vec![subpass],
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
    /// The returned future represents a semaphore signal after which the image is ready,
    /// and has not been flushed to the device.
    pub fn submit(
        &self,
        context: &crate::render_device::RenderContext,
        handle: BlendInvocationHandle,
    ) -> anyhow::Result<()> {
        // Per the vulkan specification, operations that result in binary semaphore signals
        // must be submitted prior to the operations that wait on those semaphores.
        // Thus, we must traverse the blend tree from the bottom-up.

        // Current implementation is non-optimal but that's not the goal at this point uwu

        // Array of Arrays of buffers. Each subsequent array is the next level of the tree, and will be
        // submitted from back to front with semaphores between.
        let mut layers: Vec<Vec<Arc<vk::PrimaryAutoCommandBuffer>>> = Vec::new();
        // After the current walkthrough, which tasks to walk through next?
        let mut next_layer: Vec<BlendInvocationHandle> = vec![handle];

        while !next_layer.is_empty() {
            let mut this_layer: Vec<Arc<vk::PrimaryAutoCommandBuffer>> = Vec::new();
            // Take all the next tasks, and build them. Any subtasks referenced will be added back to the queue.
            for task in std::mem::take(&mut next_layer) {
                // Build and push the commands for this operation
                // (commands can be build prior to images being ready)
                let cb = self.make_blend_commands(
                    context,
                    task.destination_image,
                    task.clear_destination,
                    &task.operations,
                )?;
                this_layer.push(cb);

                // Append subtasks to the queue to be processed
                next_layer.extend(
                    // Move all subtasks
                    task.operations
                        .into_iter()
                        .filter_map(|(image, _)| match image {
                            BlendImageSource::BlendInvocation(inv) => Some(inv),
                            BlendImageSource::Immediate(_) => None,
                        }),
                );
            }

            layers.push(this_layer);
        }

        // now, layers should be filled, from back to front, with commands in the order they must be
        // executed for proper blending.
        for buffers in layers.into_iter().rev() {
            // `buffers` may all be executed in parallel without sync.
            // After all of these, the next iteration can proceed following a semaphore.
            // continue thusly until all is consumed, then return that future!

            let mut chunks = buffers.chunks(3);
            let after = chunks.try_fold(
                vk::sync::now(context.device().clone()).boxed(),
                |after, buffers| -> anyhow::Result<_> {
                    match buffers {
                        // Execute A full chunk, then box.
                        [a, b, c] => Ok(after
                            .then_execute(context.queues().compute().queue().clone(), a.clone())?
                            .then_execute_same_queue(b.clone())?
                            .then_execute_same_queue(c.clone())?
                            .boxed()),
                        // Execute residuals, then box.
                        [a, b] => Ok(after
                            .then_execute(context.queues().compute().queue().clone(), a.clone())?
                            .then_execute_same_queue(b.clone())?
                            .boxed()),
                        [a] => Ok(after
                            .then_execute(context.queues().compute().queue().clone(), a.clone())?
                            .boxed()),
                        // chunks invariant
                        _ => unreachable!(),
                    }
                },
            )?;
            after.then_signal_fence_and_flush()?.wait(None)?;
        }

        Ok(())
    }
    /// Layers will be blended, from front to back of the slice, into a mutable background.
    /// `background` must not be aliased by any image view of `layers` (will it panic or error?)
    ///
    /// Any [`BlendImageSource::BlendInvocation`] items are assumed to have been rendered already, thus
    /// only their destination images are taken into account.
    fn make_blend_commands(
        &self,
        context: &crate::render_device::RenderContext,
        destination_image: Arc<vk::ImageView>,
        clear_destination: bool,
        layers: &[(BlendImageSource, Blend)],
    ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        let mut commands = vk::AutoCommandBufferBuilder::primary(
            context.allocators().command_buffer(),
            context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;

        if clear_destination {
            commands.clear_color_image(vk::ClearColorImageInfo {
                clear_value: [0.0; 4].into(),
                regions: smallvec::smallvec![destination_image.subresource_range().clone(),],
                ..vk::ClearColorImageInfo::image(destination_image.image().clone())
            })?;
        }

        // Still honor the request to clear the image if no layers are provided.
        if layers.is_empty() {
            return Ok(commands.build()?);
        }

        commands
            .begin_render_pass(
                vk::RenderPassBeginInfo {
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
                vk::SubpassBeginInfo::default(),
            )?
            .set_viewport(
                0,
                smallvec::smallvec![vk::Viewport {
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
            self.feedback_layout.clone(),
            1,
            feedback_set,
        )?;

        let mut last_mode = None;
        let mut last_opacity = None;
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
                    CompiledBlend::SimpleCoherent(pipe) | CompiledBlend::Loopback(pipe) => {
                        commands.bind_pipeline_graphics(pipe.clone())?;
                    }
                }
                last_mode = Some((mode, alpha_clip));
            }

            // Inform of the new alpha, if changed
            if last_opacity != Some(opacity) {
                commands.push_constants(self.feedback_layout.clone(), 0, opacity)?;
                last_opacity = Some(opacity);
            }

            // Set the image. The sampler is bundled magically by being baked into the layout itself.
            let sampler_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                self.feedback_layout.set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view(
                    0,
                    image_src.view().clone(),
                )],
                [],
            )?;

            // Bind and draw!
            // Vulkano does not seem to insert pipe barriers here, which makes sense for most uses but is UB for our
            // feedback loops. Uh oh!
            commands
                .bind_descriptor_sets(
                    vk::PipelineBindPoint::Graphics,
                    self.feedback_layout.clone(),
                    0,
                    sampler_set,
                )?
                .draw(3, 1, 0, 0)?;
        }
        commands.end_render_pass(vk::SubpassEndInfo::default())?;
        Ok(commands.build()?)
    }
}

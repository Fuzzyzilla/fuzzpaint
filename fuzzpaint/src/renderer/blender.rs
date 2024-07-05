use crate::vulkano_prelude::*;
use std::{fmt::Debug, sync::Arc};

use fuzzpaint_core::blend::{Blend, BlendMode};
use vulkano::VulkanObject;

/// `fn` to load a shader module on a given device. Equivalent to the signature of the vulkano-generated `<shader>::load`.
type BlendLoader =
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
    Arbitrary(BlendLoader),
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

/// A handle to the result of a previous blend command, can be submitted to another
/// blend builder to create nested blending effects, but otherwise has no effect on it's own.
///
/// If this handle is dropped or forgotten, it is safe to assume the resources it describes
/// are never accessed.
pub struct NestedBlendInvocation {
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
    BlendInvocation(NestedBlendInvocation),
    SolidColor(fuzzpaint_core::color::Color),
}
impl From<NestedBlendInvocation> for BlendImageSource {
    fn from(value: NestedBlendInvocation) -> Self {
        Self::BlendInvocation(value)
    }
}
impl BlendImageSource {
    fn view(&self) -> Option<&Arc<vk::ImageView>> {
        match self {
            BlendImageSource::Immediate(image)
            | BlendImageSource::BlendInvocation(NestedBlendInvocation {
                destination_image: image,
                ..
            }) => Some(image),
            Self::SolidColor(_) => None,
        }
    }
}

/// Checks if `src` requires access to the same subresource as `dest`.
fn does_alias(mutable: &vk::ImageView, immutable: &BlendImageSource) -> bool {
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

    let Some(src) = immutable.view() else {
        // No memory usage, no aliasing!
        return false;
    };
    // Recursively compare source images of nested ops.
    if let BlendImageSource::BlendInvocation(inv) = immutable {
        for (source, _) in &inv.operations {
            if does_alias(mutable, source) {
                return true;
            }
        }
    }
    // Compare base images.
    if mutable.image() == src.image() {
        // Compare subresources
        let dest = mutable.subresource_range();
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
/// Builder-style API for describing and compiling a nested blend operation.
///
/// If this handle is dropped or forgotten, it is safe to assume the resources it describes
/// are never accessed.
pub struct BlendInvocationBuilder {
    engine: Arc<BlendEngine>,
    clear_destination: bool,
    destination_image: Arc<vk::ImageView>,
    // Top of list = first operation.
    // Invariant - none if the (perhaps nested) image memory aliases the `destination_image`
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
    #[must_use = "nested blend must be submitted to another builder to have any effect"]
    pub fn nest(self) -> NestedBlendInvocation {
        NestedBlendInvocation {
            operations: self.operations,
            clear_destination: self.clear_destination,
            destination_image: self.destination_image,
        }
    }
    /// Compile all the blend operations into an executable form. This is a costly operation, and the
    /// returned object should be re-used as much as possible.
    ///
    /// # Errors
    /// There are no invalid states of `self`, but any errors from the device during compilation are returned.
    pub fn build(self) -> anyhow::Result<BlendInvocation> {
        BlendInvocation::compile(
            self.engine,
            NestedBlendInvocation {
                operations: self.operations,
                clear_destination: self.clear_destination,
                destination_image: self.destination_image,
            },
        )
    }
}

/// A compiled, ready-to-execute blend operation.
pub struct BlendInvocation {
    /// Keep device and allocators alive.
    engine: Arc<BlendEngine>,
    /// Array of stages which must be run sequentially, each stage running an array of operations in parallel.
    commands: Vec<Vec<vulkano::command_buffer::sys::UnsafeCommandBuffer>>,
    /// Kept around merely for lifetimes.
    _framebuffers: hashbrown::HashMap<
        ash::vk::ImageView,
        (Arc<vk::Framebuffer>, Arc<vk::PersistentDescriptorSet>),
    >,
    _descriptors: hashbrown::HashMap<ash::vk::ImageView, Arc<vk::PersistentDescriptorSet>>,
}
impl BlendInvocation {
    fn compile(engine: Arc<BlendEngine>, from: NestedBlendInvocation) -> anyhow::Result<Self> {
        // First, Collect resources recursively.
        fn recurse(
            engine: &BlendEngine,
            modes: &mut hashbrown::HashSet<(BlendMode, bool)>,
            framebuffers: &mut hashbrown::HashMap<
                ash::vk::ImageView,
                (Arc<vk::Framebuffer>, Arc<vk::PersistentDescriptorSet>),
            >,
            descriptors: &mut hashbrown::HashMap<
                ash::vk::ImageView,
                Arc<vk::PersistentDescriptorSet>,
            >,
            dest: &Arc<vk::ImageView>,
            sources: &[(BlendImageSource, Blend)],
        ) -> anyhow::Result<()> {
            // Insert framebuffer + feedback attachment for destination image.
            framebuffers.insert(
                dest.handle(),
                (
                    vk::Framebuffer::new(
                        engine.feedback_pass.clone(),
                        vk::FramebufferCreateInfo {
                            attachments: vec![dest.clone()],
                            extent: [dest.image().extent()[0], dest.image().extent()[1]],
                            ..Default::default()
                        },
                    )?,
                    vk::PersistentDescriptorSet::new(
                        engine.context.allocators().descriptor_set(),
                        engine.feedback_layout.set_layouts()[1].clone(),
                        [vk::WriteDescriptorSet::image_view(0, dest.clone())],
                        [],
                    )?,
                ),
            );
            // Insert sources for all children.
            for (source, blend) in sources {
                // Track the mode
                modes.insert((blend.mode, blend.alpha_clip));
                // Track the source image.
                if let Some(view) = source.view() {
                    // Make a descriptor...
                    descriptors.insert(
                        view.handle(),
                        vk::PersistentDescriptorSet::new(
                            engine.context.allocators().descriptor_set(),
                            engine.feedback_layout.set_layouts()[0].clone(),
                            [vk::WriteDescriptorSet::image_view(0, view.clone())],
                            [],
                        )?,
                    );
                }

                if let BlendImageSource::BlendInvocation(inv) = source {
                    recurse(
                        engine,
                        modes,
                        framebuffers,
                        descriptors,
                        &inv.destination_image,
                        &inv.operations,
                    )?;
                }
            }
            Ok(())
        }

        // What modes we need.
        let mut modes = hashbrown::HashSet::new();
        // Framebuffers for all `destination` images.
        let mut framebuffers = hashbrown::HashMap::new();
        // Sampled image Descriptors for every image used as a source (including nested destination images).
        let mut descriptors = hashbrown::HashMap::new();

        recurse(
            &engine,
            &mut modes,
            &mut framebuffers,
            &mut descriptors,
            &from.destination_image,
            &from.operations,
        )?;

        // Compile all the blend pipes we're gonna need for this operation
        let pipes = {
            let mut pipes = hashbrown::HashMap::new();

            // The pipes that we need but were missing.
            let mut needs_compile = Vec::new();
            let read = engine.mode_pipelines.read();

            for needed_mode in modes {
                if let Some(pipe) = read.get(&needed_mode) {
                    // Found it!
                    pipes.insert(needed_mode, pipe.clone());
                } else {
                    // Not found, remember to compile it.
                    needs_compile.push(needed_mode);
                }
            }

            // May need to switch to write, avoid deadlock.
            // (Upgradable-read is as bad as a mutex for this situation, and we can cope with
            // the non-atomicity).
            drop(read);
            if !needs_compile.is_empty() {
                // Uh oh! Do some work to compile the ones we missed.
                let mut write = engine.mode_pipelines.write();

                for (mode, clip) in needs_compile {
                    // The switch from reader to writer was not atomic, so the state may have changed.
                    // Use `entry` API to cope with this.

                    match write.entry((mode, clip)) {
                        hashbrown::hash_map::Entry::Occupied(o) => {
                            // Somebody did it for us in the meantime, how polite~!
                            pipes.insert((mode, clip), o.get().clone());
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            // Still missing, compile
                            let compiled = engine.compile_pipe_for(mode, clip)?;
                            pipes.insert((mode, clip), compiled.clone());
                            // Put in the cache too!
                            v.insert(compiled);
                        }
                    }
                }
            }
            pipes
        };

        // Tasks that can all be executed in parallel on the device.
        // child tasks should be put back.
        let mut next_tasks = vec![from];

        let mut all_commands = vec![vec![]];

        while !next_tasks.is_empty() {
            let cur_tasks = std::mem::take(&mut next_tasks);

            let commands: anyhow::Result<Vec<_>> = cur_tasks
                // this could be parallel compilation if not for the task queue....
                .into_iter()
                .map(|task| {
                    // Compile the task into a command buffer.
                    let res = Self::compile_nonrecurse(
                        &engine,
                        &task,
                        &pipes,
                        &framebuffers,
                        &descriptors,
                    )?;

                    // Push compilation tasks for nested operations onto next loop.
                    for (source, _) in task.operations {
                        if let BlendImageSource::BlendInvocation(inv) = source {
                            next_tasks.push(inv);
                        }
                    }

                    Ok(res)
                })
                .collect();

            all_commands.push(commands?);
        }

        // We start with deepest nested ops first, work up.
        all_commands.reverse();

        Ok(Self {
            engine,
            commands: all_commands,
            _descriptors: descriptors,
            _framebuffers: framebuffers,
        })
    }
    /// Compile a nested blend, without recursing into further nested blends.
    /// It contains a barrier for every input image (also non-recursive).
    ///
    /// Assumes pipes, framebuffers, and descriptors are completely filled.
    fn compile_nonrecurse(
        engine: &BlendEngine,
        op: &NestedBlendInvocation,
        pipes: &hashbrown::HashMap<(BlendMode, bool), CompiledBlend>,
        framebuffers: &hashbrown::HashMap<
            ash::vk::ImageView,
            (Arc<vk::Framebuffer>, Arc<vk::PersistentDescriptorSet>),
        >,
        descriptors: &hashbrown::HashMap<ash::vk::ImageView, Arc<vk::PersistentDescriptorSet>>,
    ) -> anyhow::Result<vulkano::command_buffer::sys::UnsafeCommandBuffer> {
        // Unfortunately, we *need* to use unsafe command buffer here. There is currently
        // an error with Auto command buffer, where pipeline barriers are not inserted correctly
        // between `CompiledBlend::Loopback` pipes leading to race conditions, and there is no way to do it manually
        // other than to lower to `unsafe`!
        unsafe {
            let mut commands = vulkano::command_buffer::sys::UnsafeCommandBufferBuilder::new(
                engine.context.allocators().command_buffer(),
                engine.context.queues().graphics().idx(),
                vulkano::command_buffer::CommandBufferLevel::Primary,
                vulkano::command_buffer::sys::CommandBufferBeginInfo {
                    usage: vk::CommandBufferUsage::MultipleSubmit,
                    inheritance_info: None,
                    ..Default::default()
                },
            )?;

            // Still honor the request to clear the image if no layers are provided.
            // In the not empty case, it's handled by a clear_attachment instead.
            if op.operations.is_empty() {
                if op.clear_destination {
                    // Wait for exclusive access...

                    // We need to make sure that the previous command buffer isn't accessing the image we're about to clear.
                    let destination_barrier = {
                        // We could make a barrier-per-image... Seems pricey for the driver to handle.
                        let barrier = vulkano::sync::ImageMemoryBarrier {
                            // We can't know what happened before, so source is maximally strong
                            src_access: vulkano::sync::AccessFlags::MEMORY_WRITE,
                            src_stages: vulkano::sync::PipelineStages::BOTTOM_OF_PIPE,

                            // We *do* know what we're gonna do, though!
                            // We can't clear until all red/writes from before complete.
                            dst_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                            // Only need CLEAR but that's extension...
                            dst_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,

                            old_layout: vk::ImageLayout::General,
                            new_layout: vk::ImageLayout::General,

                            subresource_range: op.destination_image.subresource_range().clone(),

                            ..vulkano::sync::ImageMemoryBarrier::image(
                                op.destination_image.image().clone(),
                            )
                        };

                        vulkano::sync::DependencyInfo {
                            // Can't be by-region, as clears don't work that way.
                            dependency_flags: vulkano::sync::DependencyFlags::empty(),
                            image_memory_barriers: smallvec::smallvec![barrier],
                            ..Default::default()
                        }
                    };

                    commands.pipeline_barrier(&destination_barrier)?;
                    commands.clear_color_image(&vk::ClearColorImageInfo {
                        clear_value: [0.0; 4].into(),
                        regions: smallvec::smallvec![op
                            .destination_image
                            .subresource_range()
                            .clone(),],
                        ..vk::ClearColorImageInfo::image(op.destination_image.image().clone())
                    })?;
                }
                return Ok(commands.build()?);
            }

            // Insert a frankly gigantic barrier, ensuring prior image access is complete.

            let giant_barrier_of_doom = {
                let dest_barrier = vulkano::sync::ImageMemoryBarrier {
                    // We can't know what happened before, so source is maximally strong
                    src_access: vulkano::sync::AccessFlags::MEMORY_WRITE,
                    src_stages: vulkano::sync::PipelineStages::BOTTOM_OF_PIPE,

                    // We *do* know what we're gonna do, though!
                    // We can't feedback or blend until all red/writes from before complete.
                    dst_access: vulkano::sync::AccessFlags::COLOR_ATTACHMENT_READ
                        | vulkano::sync::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vulkano::sync::AccessFlags::INPUT_ATTACHMENT_READ,
                    dst_stages: vulkano::sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT
                        | vulkano::sync::PipelineStages::FRAGMENT_SHADER,

                    subresource_range: op.destination_image.subresource_range().clone(),

                    old_layout: vk::ImageLayout::General,
                    new_layout: vk::ImageLayout::General,

                    ..vulkano::sync::ImageMemoryBarrier::image(op.destination_image.image().clone())
                };
                let src_barrier = vulkano::sync::ImageMemoryBarrier {
                    // We can't know what happened before, so source is maximally strong for writes only.
                    src_access: vulkano::sync::AccessFlags::MEMORY_WRITE,
                    src_stages: vulkano::sync::PipelineStages::BOTTOM_OF_PIPE,

                    // Only sampling
                    dst_access: vulkano::sync::AccessFlags::SHADER_READ,
                    dst_stages: vulkano::sync::PipelineStages::FRAGMENT_SHADER,

                    old_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                    new_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                    // Dummy image for _ne -w-;;
                    ..vulkano::sync::ImageMemoryBarrier::image(op.destination_image.image().clone())
                };

                vulkano::sync::DependencyInfo {
                    // Can't be by-region, as samplers don't work that way.
                    // Could split it up into two barriers, by region and global? Hmmm..
                    dependency_flags: vulkano::sync::DependencyFlags::empty(),
                    // Barrier on the output image and allll inputs.
                    image_memory_barriers: std::iter::once(dest_barrier)
                        .chain(op.operations.iter().filter_map(|(src, _)| {
                            src.view().map(|view| {
                                let mut src_barrier = src_barrier.clone();
                                src_barrier.image = view.image().clone();
                                src_barrier.subresource_range = view.subresource_range().clone();

                                if matches!(src, BlendImageSource::BlendInvocation(_)) {
                                    src_barrier.new_layout = vk::ImageLayout::General;
                                    src_barrier.old_layout = vk::ImageLayout::General;
                                }

                                src_barrier
                            })
                        }))
                        .collect(),
                    ..Default::default()
                }
            };

            commands.pipeline_barrier(&giant_barrier_of_doom)?;

            let (framebuffer, feedback_descriptor) =
                framebuffers.get(&op.destination_image.handle()).unwrap();

            commands
                // Implicit external barrier here due to external dependency in the subpass!
                .begin_render_pass(
                    &vk::RenderPassBeginInfo {
                        render_pass: engine.feedback_pass.clone(),
                        clear_values: vec![None],
                        ..vk::RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    &vk::SubpassBeginInfo::default(),
                )?
                .set_viewport(
                    0,
                    &[vk::Viewport {
                        depth_range: 0.0..=1.0,
                        offset: [0.0; 2],
                        extent: [
                            op.destination_image.image().extent()[0] as f32,
                            op.destination_image.image().extent()[1] as f32,
                        ],
                    }],
                )?;

            commands.bind_descriptor_sets(
                vk::PipelineBindPoint::Graphics,
                &engine.feedback_layout,
                0,
                &[
                    // Initialize to dummy image, as it's required to have well-formed descriptor even
                    // if it's dynamically unused.
                    engine.dummy_image_descriptor.clone().into(),
                    feedback_descriptor.clone().into(),
                ],
            )?;

            // Self-dependency barrier. MUST be a subset of the pipeline dependency flags.
            let feedback_barrier = {
                let destination_image_barrier = vulkano::sync::ImageMemoryBarrier {
                    src_access: engine.self_dependency_flags.src_access,
                    src_stages: engine.self_dependency_flags.src_stages,

                    dst_access: engine.self_dependency_flags.dst_access,
                    dst_stages: engine.self_dependency_flags.dst_stages,

                    new_layout: vk::ImageLayout::General,
                    old_layout: vk::ImageLayout::General,

                    queue_family_ownership_transfer: None,
                    subresource_range: vk::ImageSubresourceRange {
                        array_layers: 0..1,
                        aspects: vk::ImageAspects::COLOR,
                        mip_levels: 0..1,
                    },
                    ..vulkano::sync::ImageMemoryBarrier::image(op.destination_image.image().clone())
                };
                vulkano::sync::DependencyInfo {
                    dependency_flags: engine.self_dependency_flags.dependency_flags,
                    image_memory_barriers: smallvec::smallvec![destination_image_barrier],
                    ..Default::default()
                }
            };

            if op.clear_destination {
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
                            op.destination_image.image().extent()[0],
                            op.destination_image.image().extent()[1],
                        ],
                        array_layers: 0..1,
                    }],
                )?;
            }

            // Whether we just wrote to the destination image on the last loop.
            // Clear counts as a write!
            let mut had_write = op.clear_destination;
            // Whether the current pipe will read the destination image. It is UB
            // for a read to occur after a write without a barrier.
            let mut will_read = false;
            let mut last_mode = None;

            let mut last_constants = None;
            for (image_src, blend) in &op.operations {
                let Blend {
                    mode,
                    alpha_clip,
                    opacity,
                } = *blend;
                // bind a new pipeline if changed from last iter
                if last_mode != Some((mode, alpha_clip)) {
                    let pipe = pipes.get(&(mode, alpha_clip)).unwrap();
                    match pipe {
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
                    BlendImageSource::BlendInvocation(NestedBlendInvocation {
                        destination_image: view,
                        ..
                    })
                    | BlendImageSource::Immediate(view) => {
                        commands.bind_descriptor_sets(
                            vk::PipelineBindPoint::Graphics,
                            &engine.feedback_layout,
                            0,
                            &[descriptors.get(&view.handle()).unwrap().clone().into()],
                        )?;

                        let constants = shaders::Constants::new_image(opacity);
                        if last_constants != Some(constants) {
                            commands.push_constants(&engine.feedback_layout, 0, &constants)?;
                            last_constants = Some(constants);
                        }
                    }
                    &BlendImageSource::SolidColor(color) => {
                        let constants = shaders::Constants::new_solid(color.alpha_multipy(
                            fuzzpaint_core::util::FiniteF32::new(opacity).unwrap_or_default(),
                        ));

                        if last_constants != Some(constants) {
                            commands.push_constants(&engine.feedback_layout, 0, &constants)?;
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
    /// Execute the entire blend tree.
    ///
    /// # Safety
    /// ## Access
    /// * This invocation must not be run on the device in parallel with itself.
    /// * This invocation host object must outlive the duration of execution on the device.
    /// * All input images must be externally synchronized and retain shared access
    ///   by the graphics queue during the operation.
    /// * All output images must be externally synchronized and retain exclusive access
    ///   by the graphics queue during the operation.
    /// ## Layout
    /// * All destination images must be in `GENERAL` layout.
    // (we don't specify that `clear`ed destination images may be any layout, since
    // there's no way to communicate back to vulkano that the transition occured)
    /// * All source images must be in `SHADER_READ_ONLY_OPTIMAL` layout.
    // Todo: input sync? As-is, the user must wait on host side, not ideal.
    pub unsafe fn execute(&self) -> anyhow::Result<()> {
        for buffers in &self.commands {
            // `buffers` may all be executed in parallel without sync.
            // After all of these, the next iteration can proceed following a barrier.
            let graphics = self.engine.context.queues().graphics().queue();
            let graphics_handle = graphics.handle();
            let pfn_submit = self.engine.context.device().fns().v1_0.queue_submit;

            let raw_buffers = buffers
                .iter()
                .map(vulkano::VulkanObject::handle)
                .collect::<Vec<_>>();
            // No need to wait semaphores since each buffer contains an overkill barrier. Fixme!
            let submission = ash::vk::SubmitInfo::builder().command_buffers(&raw_buffers);

            // Synchronize queue externally using vulkano's mechanisms!
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

        // Fixme: fence/semaphore instead of eager wait idle.
        self.engine
            .context
            .queues()
            .graphics()
            .queue()
            .with(|mut q| q.wait_idle())?;

        Ok(())
    }
    /// Like `execute`, except only running necessary calculations for one image source being changed.
    /// No guarantees are made about this accessing fewer resources than a full execution.
    ///
    /// # Safety:
    /// See [`BlendInvocation::execute`]
    pub unsafe fn execute_delta(&self, _changed: &vk::Image) -> anyhow::Result<()> {
        // Todo! This is a valid implementation, however.
        // Safety forwarded to caller.
        unsafe { self.execute() }
    }
}

#[derive(Copy, Clone, Debug, thiserror::Error)]
pub enum ExecuteDeltaError {
    #[error("the changed image was not a source")]
    NotASource,
}

pub struct BlendEngine {
    context: Arc<crate::render_device::RenderContext>,
    /// Layout for all blend operations.
    feedback_layout: Arc<vk::PipelineLayout>,
    /// Spawns fragments across the entire viewport. No outputs.
    fullscreen_vert: vk::EntryPoint,
    /// Passes fragments from input attachment 0, set 0, binding 0, unchanged.
    coherent_frag: vk::EntryPoint,
    /// Pipes need a renderpass to build against. We don't have the pass at the time of pipe compilation, tho!
    feedback_pass: Arc<vk::RenderPass>,
    /// (mode, clip) -> prepared blend pipe.
    mode_pipelines: parking_lot::RwLock<hashbrown::HashMap<(BlendMode, bool), CompiledBlend>>,
    /// Data needed to specify self-dependency barrier.
    self_dependency_flags: vk::SubpassDependency,
    /// An image with undefined content, in `SHADER_READ_ONLY_OPTIMAL` layout, "usable" for sampling and nothing else.
    /// (Although it is undefined behavior to actually access.)
    /// Vulkan requires that any desciptor set that is statically-used be in a fully valid state, even if it is
    /// dynamically unused. Use this for such cases.
    dummy_image_descriptor: Arc<vk::PersistentDescriptorSet>,
}
impl BlendEngine {
    /// Compile the blend logic for a given mode. Does *not* access the mode cache or check if it was already compiled.
    fn compile_pipe_for(&self, mode: BlendMode, clip: bool) -> anyhow::Result<CompiledBlend> {
        // Fetch the equation
        let logic = BlendLogic::of(mode, clip);
        // Compile it!
        match logic {
            BlendLogic::Simple(equation) => {
                let pipe = vk::GraphicsPipeline::new(
                    self.context.device().clone(),
                    None,
                    vulkano::pipeline::graphics::GraphicsPipelineCreateInfo {
                        stages: smallvec::smallvec![
                            vk::PipelineShaderStageCreateInfo::new(self.fullscreen_vert.clone()),
                            vk::PipelineShaderStageCreateInfo::new(self.coherent_frag.clone())
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

                Ok(CompiledBlend::SimpleCoherent(pipe))
            }
            BlendLogic::Arbitrary(load) => {
                let shader = load(self.context.device().clone())?
                    .entry_point("main")
                    .ok_or_else(|| anyhow::anyhow!("entry point `main` not found"))?;

                let pipe = vk::GraphicsPipeline::new(
                    self.context.device().clone(),
                    None,
                    vulkano::pipeline::graphics::GraphicsPipelineCreateInfo {
                        stages: smallvec::smallvec![
                            vk::PipelineShaderStageCreateInfo::new(self.fullscreen_vert.clone()),
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

                Ok(CompiledBlend::Loopback(pipe))
            }
        }
    }
    /// Create a dummy image descriptor set. See [`BlendEngine::dummy_image_descriptor`] for motivation.
    fn make_dummy_image_descriptor(
        context: &crate::render_device::RenderContext,
        layout: Arc<vk::DescriptorSetLayout>,
    ) -> anyhow::Result<Arc<vk::PersistentDescriptorSet>> {
        // No need to populate the image, since it's a logic error to read anyway.
        let image = vk::Image::new(
            context.allocators().memory().clone(),
            vulkano::image::ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: vulkano::format::Format::R8_UNORM,
                extent: [1; 3],
                array_layers: 1,
                mip_levels: 1,
                samples: vk::SampleCount::Sample1,
                tiling: vulkano::image::ImageTiling::Optimal,
                usage: vk::ImageUsage::SAMPLED,
                sharing: vk::Sharing::Exclusive,
                initial_layout: vk::ImageLayout::Undefined,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;
        let view = vk::ImageView::new(
            image.clone(),
            vk::ImageViewCreateInfo {
                format: image.format(),
                ..vk::ImageViewCreateInfo::from_image(&image)
            },
        )?;
        // Insert a dummy barrier to transition the image. Expensive!
        unsafe {
            let mut cb = vulkano::command_buffer::sys::UnsafeCommandBufferBuilder::new(
                context.allocators().command_buffer(),
                context.queues().graphics().idx(),
                vulkano::command_buffer::CommandBufferLevel::Primary,
                vulkano::command_buffer::sys::CommandBufferBeginInfo {
                    usage: vk::CommandBufferUsage::OneTimeSubmit,
                    inheritance_info: None,
                    ..Default::default()
                },
            )?;
            let dummy_transition = vk::sync::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::Undefined,
                new_layout: vk::ImageLayout::ShaderReadOnlyOptimal,

                // None.
                src_stages: vk::sync::PipelineStages::TOP_OF_PIPE,
                src_access: vk::sync::AccessFlags::empty(),

                dst_stages: vk::sync::PipelineStages::FRAGMENT_SHADER,
                dst_access: vk::sync::AccessFlags::SHADER_READ,

                subresource_range: image.subresource_range(),

                ..vk::sync::ImageMemoryBarrier::image(image)
            };
            cb.pipeline_barrier(&vk::sync::DependencyInfo {
                image_memory_barriers: smallvec::smallvec![dummy_transition],
                ..Default::default()
            })?;

            let cb = cb.build()?;
            let cb = [cb.handle()];

            let submission = ash::vk::SubmitInfo::builder().command_buffers(&cb);

            let graphics = context.queues().graphics().queue();
            let graphics_handle = graphics.handle();
            let pfn_submit = context.device().fns().v1_0.queue_submit;

            // Synchronize externally
            graphics.with(|mut lock| -> anyhow::Result<()> {
                (pfn_submit)(
                    graphics_handle,
                    1,
                    &submission.build(),
                    ash::vk::Fence::null(),
                )
                .result()?;
                // cb, image must outlive execution. Weh.
                lock.wait_idle()?;
                Ok(())
            })?;
        }
        vk::PersistentDescriptorSet::new(
            context.allocators().descriptor_set(),
            layout,
            [vk::WriteDescriptorSet::image_view_with_layout(
                0,
                vulkano::descriptor_set::DescriptorImageViewInfo {
                    image_view: view,
                    image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                },
            )],
            [],
        )
        .map_err(Into::into)
    }
    pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Arc<Self>> {
        let device = context.device();

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
            size: std::mem::size_of::<shaders::Constants>()
                .try_into()
                .unwrap(),
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
                src_access: vk::sync::AccessFlags::MEMORY_WRITE,

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

        let dummy_image_descriptor =
            Self::make_dummy_image_descriptor(&context, feedback_layout.set_layouts()[0].clone())?;

        Ok(Self {
            context,
            feedback_layout,
            fullscreen_vert,
            coherent_frag,
            feedback_pass,
            mode_pipelines: hashbrown::HashMap::new().into(),
            self_dependency_flags,
            dummy_image_descriptor,
        }
        .into())
    }
    /// Begin a blend operation with the engine. Use the returned object to describe and compile a GPU blend operation.
    #[must_use = "use the result to build an operation"]
    pub fn start(
        self: Arc<Self>,
        destination_image: Arc<vk::ImageView>,
        clear_destination: bool,
    ) -> BlendInvocationBuilder {
        BlendInvocationBuilder {
            engine: self,
            clear_destination,
            destination_image,
            operations: Vec::new(),
        }
    }
}

use crate::vulkano_prelude::*;
use std::{fmt::Debug, sync::Arc};

use fuzzpaint_core::blend::{Blend, BlendMode};
use vulkano::VulkanObject;

/// Providing a "quoted" GLSL snippit, accepting premultiplied RGBA `vec4 c_src` and `vec4 c_dst`, `return` the new color.
/// Logic need not handle the additional requirement that transparent dst returns src unchanged, that
/// is handled by the rest of the shader logic.
macro_rules! blend_noclip {
    ($blend_expr:literal) => {{
        vulkano_shaders::shader! {
            ty: "fragment",
            define: [
                ("EXPR", $blend_expr),
            ],
            path: "src/shaders/blend_no_clip.frag",
        }
        crate::renderer::blender::BlendLogic::Arbitrary(load)
    }};
}
/// Providing a "quoted" GLSL snippit, accepting an opaque RGBA `vec4 c_src` and premultiplied `vec4 c_dst`,
/// `return` the new color. Logic need not handle the additional clipping logic, that is handled by the rest
/// of the shader logic.
macro_rules! blend_clip {
    ($blend_expr:literal) => {{
        vulkano_shaders::shader! {
            ty: "fragment",
            define: [
                ("EXPR", $blend_expr),
            ],
            path: "src/shaders/blend_clip.frag",
        }
        crate::renderer::blender::BlendLogic::Arbitrary(load)
    }};
}

/// `fn` to load a shader module on a given device. Equivalent to the signature of the vulkano-generated `<shader>::load`.
type BlendLoader =
    fn(Arc<vk::Device>) -> Result<Arc<vk::ShaderModule>, vk::Validated<vk::VulkanError>>;

enum BlendLogic {
    /// The blend can be represented as a standard blend function. Hardware accelerated, pipelinable, and coherent. Nice :3
    Simple(vk::AttachmentBlend),
    /// Provide a Load function for a shader to compute `A (+) B` for arbitrary blend logic. Still pipeliend, but noncoherent 3:
    Arbitrary(BlendLoader),
}
impl From<vk::AttachmentBlend> for BlendLogic {
    fn from(value: vk::AttachmentBlend) -> Self {
        Self::Simple(value)
    }
}
impl From<BlendLoader> for BlendLogic {
    fn from(value: BlendLoader) -> Self {
        Self::Arbitrary(value)
    }
}
impl BlendLogic {
    /// Get the logic needed to perform a blend.
    fn of(blend: BlendMode, clip: bool) -> Self {
        use vk::{AttachmentBlend, BlendFactor, BlendOp};

        // Big ol' note to self: BlendOp::  {Min, Max} *silently ignore factors*. Only ever uses `One`.
        // Don't waste any time trying to write coherent Clipped Lighten or Un/clipped Darken! :P

        // When writing new operations here, remember that:
        // * if the result alpha is zero, RGB must also be zero.
        // * if the dst alpha is zero and not clip, then use src directly.
        // * if the src alpha is zero, dst should be unchanged.
        // * if clip, then the dst alpha should be unchanged.
        // This serves as a decent litmus test for if something is horribly borked.

        // The logic for usual behaviours of the alpha channel, color channels must be overridden.
        let alpha_clip = AttachmentBlend {
            // Keep alpha from dest.
            src_alpha_blend_factor: BlendFactor::Zero,
            dst_alpha_blend_factor: BlendFactor::One,
            ..Default::default()
        };

        let alpha_no_clip = AttachmentBlend {
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
            ..Default::default()
        };

        // Verification is based on parity testing with Krita
        match (blend, clip) {
            // Verified
            (BlendMode::Normal, false) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::One,
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..alpha_no_clip
            }
            .into(),
            // Verified
            (BlendMode::Normal, true) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::DstAlpha,
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..alpha_clip
            }
            .into(),
            // Verified
            (BlendMode::Add, false) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::One,
                dst_color_blend_factor: BlendFactor::One,
                ..alpha_no_clip
            }
            .into(),
            // `[Src * DstAlpha] + [Dst]` is subtly wrong. (red + translucent white) = translucent red, should be white.
            // This is strange, but only in relation to an art software that clips at white. This is expected behavior
            // given that we allow brighter-than-white colors. (The red fringe is >100% red being faded out with alpha)
            //
            // Somehow, Krita does not have this fringe, even with unclamped floating point color. Not sure how!
            (BlendMode::Add, true) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::DstAlpha,
                dst_color_blend_factor: BlendFactor::One,
                ..alpha_clip
            }
            .into(),
            // `[Src * Dst] + [Dst * (1 - SrcAlpha)]` is subtly wrong, (red * transparent) = black, should be red.
            // So, use arbitrary to patch it with more complex logic.
            // Verified
            (BlendMode::Multiply, false) => {
                blend_noclip! {"return c_dst * c_src + c_dst * (1.0 - c_src.a);"}
            }
            // Verified
            (BlendMode::Multiply, true) => AttachmentBlend {
                // Funny! No mul op, so instead "Normal" op with the left side having a multiply factor.
                src_color_blend_factor: BlendFactor::DstColor,
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..alpha_clip
            }
            .into(),
            // Verified
            (BlendMode::Screen, false) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::One,
                dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
                ..alpha_no_clip
            }
            .into(),
            // Verified
            (BlendMode::Screen, true) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::DstAlpha,
                dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
                ..alpha_clip
            }
            .into(),
            // Very wrong
            (BlendMode::Darken, false) => blend_noclip!("return min(c_src, c_dst);"),
            (BlendMode::Darken, true) => blend_clip!("return min(c_src, c_dst);"),
            // Verified
            (BlendMode::Lighten, false) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::One,
                dst_color_blend_factor: BlendFactor::One,
                color_blend_op: BlendOp::Max,
                ..alpha_no_clip
            }
            .into(),
            // Very wrong
            (BlendMode::Lighten, true) => blend_clip!("return max(c_src, c_dst);"),
            // Unique exception to the "if clip, then the dst alpha should be unchanged" rule, as this
            // is the only mode that can *decrease* image opacity.
            // Verified, both clip and not.
            (BlendMode::Erase, _) => AttachmentBlend {
                src_color_blend_factor: BlendFactor::Zero,
                src_alpha_blend_factor: BlendFactor::Zero,
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                ..Default::default()
            }
            .into(),
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
impl CompiledBlend {
    fn reads_input_attachment(&self) -> bool {
        match self {
            Self::SimpleCoherent(_) => false,
            Self::Loopback(_) => true,
        }
    }
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
    commands: Vec<Vec<Arc<vk::PrimaryAutoCommandBuffer>>>,
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
                        [vk::WriteDescriptorSet::image_view_with_layout(
                            0,
                            vulkano::descriptor_set::DescriptorImageViewInfo {
                                image_layout: vk::ImageLayout::General,
                                image_view: dest.clone(),
                            },
                        )],
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
                            [vk::WriteDescriptorSet::image_view_with_layout(
                                0,
                                vulkano::descriptor_set::DescriptorImageViewInfo {
                                    image_view: view.clone(),
                                    image_layout: if matches!(
                                        source,
                                        BlendImageSource::BlendInvocation(_)
                                    ) {
                                        vk::ImageLayout::General
                                    } else {
                                        vk::ImageLayout::ShaderReadOnlyOptimal
                                    },
                                },
                            )],
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

            for (needed_mode, needed_clip) in modes {
                if let Some((unclipped_pipe, clipped_pipe)) = read.get(&needed_mode) {
                    let pipe = if needed_clip {
                        clipped_pipe
                    } else {
                        unclipped_pipe
                    };

                    if let Some(pipe) = pipe {
                        // Found it!
                        pipes.insert((needed_mode, needed_clip), pipe.clone());
                        continue;
                    }
                }
                // fallthrough = Not found, remember to compile it.
                needs_compile.push((needed_mode, needed_clip));
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

                    match write.entry(mode) {
                        hashbrown::hash_map::Entry::Occupied(mut o) => {
                            let (unclipped_pipe, clipped_pipe) = o.get_mut();
                            let pipe = if clip { clipped_pipe } else { unclipped_pipe };

                            if let Some(pipe) = pipe {
                                // Somebody did it for us in the meantime, how polite~!
                                pipes.insert((mode, clip), pipe.clone());
                            } else {
                                // Still missing, compile
                                let compiled = engine.compile_pipe_for(mode, clip)?;
                                pipes.insert((mode, clip), compiled.clone());
                                // Put in the open cache slot too!
                                *pipe = Some(compiled);
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            // Still missing, compile
                            let compiled = engine.compile_pipe_for(mode, clip)?;
                            pipes.insert((mode, clip), compiled.clone());

                            // Make a new entry in the cache too, in the right position:
                            let pair = if clip {
                                (None, Some(compiled))
                            } else {
                                (Some(compiled), None)
                            };
                            v.insert(pair);
                        }
                    }
                }
            }
            pipes
        };

        // Tasks that can all be executed in parallel on the device.
        // child tasks should be put back.
        let mut next_tasks = vec![from];

        let mut all_commands = vec![];

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

            let commands = commands?;
            all_commands.push(commands);
        }

        // We start with deepest nested ops first, work up.
        all_commands.reverse();

        Ok(Self {
            engine,
            commands: all_commands,
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
    ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        // Unfortunately, we *need* to use unsafe command buffer here. There is currently
        // an error with Auto command buffer, where pipeline barriers are not inserted correctly
        // between `CompiledBlend::Loopback` pipes leading to race conditions, and there is no way to do it manually
        // other than to lower to `unsafe`!
        let mut commands = vk::AutoCommandBufferBuilder::primary(
            engine.context.allocators().command_buffer(),
            engine.context.queues().graphics().idx(),
            vk::CommandBufferUsage::MultipleSubmit,
        )?;

        // Still honor the request to clear the image if no layers are provided.
        // In the not empty case, it's handled by a clear_attachment instead.
        if op.operations.is_empty() {
            if op.clear_destination {
                commands.clear_color_image(vk::ClearColorImageInfo {
                    clear_value: [0.0; 4].into(),
                    regions: smallvec::smallvec![op.destination_image.subresource_range().clone(),],
                    ..vk::ClearColorImageInfo::image(op.destination_image.image().clone())
                })?;
            }
            return Ok(commands.build()?);
        }

        let (framebuffer, feedback_descriptor) =
            framebuffers.get(&op.destination_image.handle()).unwrap();

        let render_pass_begin = vk::RenderPassBeginInfo {
            render_pass: engine.feedback_pass.clone(),
            clear_values: vec![None],
            ..vk::RenderPassBeginInfo::framebuffer(framebuffer.clone())
        };

        commands
            // Implicit external barrier here due to external dependency in the subpass!
            .begin_render_pass(render_pass_begin.clone(), vk::SubpassBeginInfo::default())?
            .set_viewport(
                0,
                smallvec::smallvec![vk::Viewport {
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
            engine.feedback_layout.clone(),
            0,
            (
                // Initialize to dummy image, as it's required to have well-formed descriptor even
                // if it's dynamically unused.
                engine.dummy_image_descriptor.clone(),
                feedback_descriptor.clone(),
            ),
        )?;

        if op.clear_destination {
            // Clear before any blends occur. This is properly barrier'd
            // by setting write to `true` initially.
            commands.clear_attachments(
                smallvec::smallvec![vulkano::command_buffer::ClearAttachment::Color {
                    color_attachment: 0,
                    clear_value: [0.0; 4].into(),
                }],
                smallvec::smallvec![vulkano::command_buffer::ClearRect {
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
                        commands.bind_pipeline_graphics(pipe.clone())?;
                    }
                    CompiledBlend::Loopback(pipe) => {
                        will_read = true;
                        commands.bind_pipeline_graphics(pipe.clone())?;
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
                        engine.feedback_layout.clone(),
                        0,
                        descriptors.get(&view.handle()).unwrap().clone(),
                    )?;

                    let constants = shaders::Constants::new_image(opacity);
                    if last_constants != Some(constants) {
                        commands.push_constants(engine.feedback_layout.clone(), 0, constants)?;
                        last_constants = Some(constants);
                    }
                }
                &BlendImageSource::SolidColor(color) => {
                    let constants = shaders::Constants::new_solid(color.alpha_multipy(
                        fuzzpaint_core::util::FiniteF32::new(opacity).unwrap_or_default(),
                    ));

                    if last_constants != Some(constants) {
                        commands.push_constants(engine.feedback_layout.clone(), 0, constants)?;
                        last_constants = Some(constants);
                    }
                }
            }

            if had_write && will_read {
                // Insert a barrier to ensure last loop's write completes before this loop's read.
                // Okay okay so...
                // I went on an entire week long expedition to use vkCmdPipelineBarrier here with a subpass self-dependency,
                // it was glorious! Lovely! Amazing and fast! Alas, it is impossible to inform vulkano of the changes made
                // during an external command buffer (vulkano::command_buffer::sys::UnsafeCommandBuffer), and so despite
                // the blend code working a treat, it caused UB outside due to incorrect image layouts from vulkano working off
                // of outdated info... The only option was to let unsafe command buffers permeate the whole app so that I am
                // in control of layouts and barriers, a non-trivial task this deep into it.

                // I am not pleased by end-begin dance, it's surely substantially slower but it allows vulkano to insert the
                // necessary barrier itself, which then updates internal structures to avoid that UB in other places. Heck.

                // Marc is working on a graph implementation for vulkano, allowing raw vulkan code to interact and communicate
                // with vulkano, fixing this issue in some future release. Thank you marc!
                commands.end_render_pass(vk::SubpassEndInfo::default())?;
                commands.begin_render_pass(
                    render_pass_begin.clone(),
                    vk::SubpassBeginInfo::default(),
                )?;
            }

            commands.draw(3, 1, 0, 0)?;
            had_write = true;
        }
        commands.end_render_pass(vk::SubpassEndInfo::default())?;
        Ok(commands.build()?)
    }
    /// Execute the entire blend tree.
    ///
    // Todo: input sync? As-is, the user must wait on host side, not ideal.
    pub fn execute(&self) -> anyhow::Result<()> {
        for buffers in &self.commands {
            // `buffers` may all be executed in parallel without sync.
            // This is not possible to express in vulkano, blegh.
            let mut future = self.engine.context.now().boxed();
            for buffer in buffers {
                future = future
                    .then_execute(
                        self.engine.context.queues().graphics().queue().clone(),
                        buffer.clone(),
                    )?
                    .boxed();
            }
            // Fixme: fence/semaphore instead of eager wait idle.
            future.then_signal_fence_and_flush()?.wait(None)?;
        }

        Ok(())
    }
    /// Like `execute`, except only running necessary calculations for one image source being changed.
    /// No guarantees are made about this accessing fewer resources than a full execution.
    ///
    /// # Safety:
    /// See [`BlendInvocation::execute`]
    pub fn execute_delta(&self, _changed: &vk::Image) -> anyhow::Result<()> {
        // Todo! This is a valid implementation, however.
        self.execute()
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
    /// mode -> (unclipped pipe, clipped pipe).
    mode_pipelines: parking_lot::RwLock<
        hashbrown::HashMap<BlendMode, (Option<CompiledBlend>, Option<CompiledBlend>)>,
    >,
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
                            attachments: vec![vk::ColorBlendAttachmentState {
                                blend: None,
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
        // Vulkano executes a barrier just-in-time to get it from undefined to needed ShaderReadOnly layout.
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

            vk::RenderPass::new(
                device.clone(),
                vk::RenderPassCreateInfo {
                    attachments: vec![feedback_attachment],
                    subpasses: vec![subpass],
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

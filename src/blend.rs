use crate::vulkano_prelude::*;
use std::{default, fmt::Debug, sync::Arc};

#[derive(
    strum::AsRefStr,
    PartialEq,
    Eq,
    strum::EnumIter,
    Copy,
    Clone,
    Hash,
    Debug,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(u8)]
pub enum BlendMode {
    Normal,
    Add,
    Multiply,
    Overlay,
}
impl Default for BlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Blend mode for an object, including a mode, opacity modulate, and alpha clip
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Blend {
    pub mode: BlendMode,
    pub opacity: f32,
    /// If alpha clip enabled, it should not affect background alpha, krita style!
    pub alpha_clip: bool,
}
impl Default for Blend {
    fn default() -> Self {
        Self {
            mode: Default::default(),
            opacity: 1.0,
            alpha_clip: false,
        }
    }
}

mod shaders {
    pub const INPUT_IMAGE_SET: u32 = 0;
    pub const OUTPUT_IMAGE_SET: u32 = 1;

    /// Every blend shader specializes on the 2d size of workgroups to be selected at runtime.
    /// Z size is always one (for now, maybe in multisampling z can represent sample index)
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub struct WorkgroupSizeConstants {
        pub x: u32,
        pub y: u32,
    }
    /// The push constants to customize blending. Corresponds to fields in super::Blend.
    /// Every blend shader must accept this struct format!
    // Noteably, NOT AnyBitPattern nor Pod, bool32 has invalid states
    #[derive(bytemuck::Zeroable, vulkano::buffer::BufferContents, Clone, Copy, PartialEq)]
    #[repr(C)]
    pub struct BlendConstants {
        pub opacity: f32,
        /// Bool32: only 1 and 0 are valid!!
        clip: u32,
    }
    impl BlendConstants {
        pub fn new(opacity: f32, clip: bool) -> Self {
            Self {
                opacity,
                clip: if clip { 1 } else { 0 },
            }
        }
    }
    /// Push constants to specify the rectangle to blend. Todo!
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    #[repr(C)]
    pub struct BlendRect {
        pub origin: [u32; 2],
        pub size: [u32; 2],
    }

    pub mod normal {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/shaders/blend/blend_one.comp",
            include: ["src/shaders/blend"],
            define: [("BLEND_NORMAL", "."), ("MODE_FUNC", "blend_normal")],
        }
    }
    pub mod add {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/shaders/blend/blend_one.comp",
            include: ["src/shaders/blend"],
            define: [("BLEND_ADD", "."), ("MODE_FUNC", "blend_add")],
        }
    }
    pub mod multiply {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/shaders/blend/blend_one.comp",
            include: ["src/shaders/blend"],
            define: [("BLEND_MULTIPLY", "."), ("MODE_FUNC", "blend_multiply")],
        }
    }
    pub mod overlay {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/shaders/blend/blend_one.comp",
            include: ["src/shaders/blend"],
            define: [("BLEND_OVERLAY", "."), ("MODE_FUNC", "blend_overlay")],
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

/// Checks if `src`` requires access to the same subresource as `dest`.
fn does_alias(dest: &vk::ImageView, src: &BlendImageSource) -> bool {
    let src = src.view();
    if dest.image() != src.image() {
        false
    } else {
        // Compare subresources
        let dest = dest.subresource_range();
        let src = src.subresource_range();

        // The simplest programming problems make my brain overheat
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

        // Array overlaps, mips overlap, AND aspect overlaps? then aliases!
        ranges_intersect(dest.array_layers.clone(), src.array_layers.clone())
            && ranges_intersect(dest.mip_levels.clone(), src.mip_levels.clone())
            && dest.aspects.intersects(src.aspects)
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
        self.operations.reverse()
    }
    /// Build the invocation. This handle can be used in other blend invocations as a source,
    /// or it may be provided to [BlendEngine::submit] to begin device execution of the blend operation.
    ///
    /// If instead this result is discarded, it is safe to assume the resources described in this operation
    /// are never accessed.
    #[must_use = "The blend handle must be submitted to the engine for blending to occur"]
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
    workgroup_size: (u32, u32),
    // Based on chosen workgroup_size, device's max workgroup count, and device image size.
    max_image_size: (u32, u32),
    shader_layout: Arc<vk::PipelineLayout>,
    mode_pipelines: std::collections::HashMap<BlendMode, Arc<vk::ComputePipeline>>,
}
impl BlendEngine {
    fn build_pipeline(
        device: Arc<vk::Device>,
        layout: Arc<vk::PipelineLayout>,
        size: shaders::WorkgroupSizeConstants,
        entry_point: Arc<vulkano::shader::ShaderModule>,
    ) -> anyhow::Result<Arc<vk::ComputePipeline>> {
        let mut specialization =
            ahash::HashMap::<u32, vk::SpecializationConstant>::with_capacity_and_hasher(
                2,
                Default::default(),
            );
        specialization.insert(0, size.x.into());
        specialization.insert(1, size.y.into());
        let pipeline = vk::ComputePipeline::new(
            device,
            None,
            vk::ComputePipelineCreateInfo::stage_layout(
                vk::PipelineShaderStageCreateInfo::new(
                    entry_point
                        .specialize(specialization)?
                        .entry_point("main")
                        .ok_or_else(|| anyhow::anyhow!("Entry point not found"))?,
                ),
                layout,
            ),
        )?;
        Ok(pipeline)
    }
    pub fn new(device: Arc<vk::Device>) -> anyhow::Result<Self> {
        // compute the workgroup size, specified as specialization constants
        let properties = device.physical_device().properties();
        let workgroup_size = {
            // Todo: better alg for this lol
            let largest_square = (properties.max_compute_work_group_invocations as f64)
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

        // Build fixed layout for all blend processes
        let mut input_image_bindings = std::collections::BTreeMap::new();
        input_image_bindings.insert(
            0,
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                immutable_samplers: Default::default(),
                stages: vk::ShaderStages::COMPUTE,
                ..vk::DescriptorSetLayoutBinding::descriptor_type(vk::DescriptorType::StorageImage)
            },
        );
        let output_image_bindings = input_image_bindings.clone();

        let input_image_layout = vk::DescriptorSetLayout::new(
            device.clone(),
            vk::DescriptorSetLayoutCreateInfo {
                bindings: input_image_bindings,
                ..Default::default()
            },
        )?;
        let output_image_layout = vk::DescriptorSetLayout::new(
            device.clone(),
            vk::DescriptorSetLayoutCreateInfo {
                bindings: output_image_bindings,
                ..Default::default()
            },
        )?;

        let shader_layout = vk::PipelineLayout::new(
            device.clone(),
            vk::PipelineLayoutCreateInfo {
                set_layouts: vec![input_image_layout, output_image_layout],
                push_constant_ranges: vec![vk::PushConstantRange {
                    offset: 0,
                    size: 24, // f32 + bool32 + Rect:(u32 * 4)
                    stages: vk::ShaderStages::COMPUTE,
                }],
                ..vk::PipelineLayoutCreateInfo::default()
            },
        )?;
        let size = shaders::WorkgroupSizeConstants {
            x: workgroup_size.0,
            y: workgroup_size.1,
        };

        let mut modes = std::collections::HashMap::new();
        /// Very smol inflexible macro to compile and insert one blend mode program from the `shaders` module into the `modes` map.
        macro_rules! build_mode {
            ($mode:expr, $namespace:ident) => {
                let mode: BlendMode = $mode;
                let prev = modes.insert(
                    mode,
                    Self::build_pipeline(
                        device.clone(),
                        shader_layout.clone(),
                        size,
                        shaders::$namespace::load(device.clone())?,
                    )?,
                );
                assert!(
                    prev.is_none(),
                    "Overwrote blend program {mode:?}. Did you typo the name?"
                );
            };
        }

        // It is unreasonably cool how well the rust-analyzer autocomplete works here :O
        build_mode!(BlendMode::Normal, normal);
        build_mode!(BlendMode::Add, add);
        build_mode!(BlendMode::Multiply, multiply);
        build_mode!(BlendMode::Overlay, overlay);

        Ok(Self {
            shader_layout,
            max_image_size,
            workgroup_size,
            mode_pipelines: modes,
        })
    }
    /// Begin a blend operation with the engine.
    /// The destination image must be available at the time of calling `submit`.
    pub fn start(
        &self,
        destination_image: Arc<vk::ImageView>,
        clear_destination: bool,
    ) -> BlendInvocationBuilder {
        BlendInvocationBuilder {
            clear_destination,
            destination_image,
            operations: Default::default(),
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
        let mut layers: Vec<Vec<Arc<vk::PrimaryAutoCommandBuffer>>> = Default::default();
        // After the current walkthrough, which tasks to walk through next?
        let mut next_layer: Vec<BlendInvocationHandle> = vec![handle];

        while !next_layer.is_empty() {
            let mut this_layer: Vec<Arc<vk::PrimaryAutoCommandBuffer>> = Default::default();
            // Take all the next tasks, and build them. Any subtasks referenced will be added back to the queue.
            for task in std::mem::take(&mut next_layer).into_iter() {
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
                            _ => None,
                        }),
                );
            }

            layers.push(this_layer);
        }

        // now, layers should be filled, from back to front, with commands in the order they must be
        // executed for proper blending.

        let mut after = vk::sync::now(context.device().clone()).boxed();

        for buffers in layers.into_iter().rev() {
            // `buffers` may all be executed in parallel without sync.
            // After all of these, the next iteration can proceed following a semaphore.
            // continue thusly until all is consumed, then return that future!

            let mut chunks = buffers.chunks(3);
            after = chunks.try_fold(
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
    /// Any [BlendImageSource::BlendInvocation] items are assumed to have been rendered already, thus
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
            context.queues().compute().idx(),
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

        // Compute the number of workgroups to dispatch for a given image
        // Or, None if the number of workgroups exceeds the maximum the device supports.
        let output_size = destination_image.image().extent();
        let get_dispatch_size = |dimensions: [u32; 3]| -> Option<[u32; 3]> {
            let x = dimensions[0].min(output_size[0]);
            let y = dimensions[1].min(output_size[1]);
            if x > self.max_image_size.0 || y > self.max_image_size.1 {
                None
            } else {
                Some([
                    x.div_ceil(self.workgroup_size.0),
                    y.div_ceil(self.workgroup_size.1),
                    1,
                ])
            }
        };

        let output_set = vk::PersistentDescriptorSet::new(
            context.allocators().descriptor_set(),
            self.shader_layout.set_layouts()[shaders::OUTPUT_IMAGE_SET as usize].clone(),
            [vk::WriteDescriptorSet::image_view(
                0,
                destination_image.clone(),
            )],
            [],
        )?;
        commands.bind_descriptor_sets(
            vk::PipelineBindPoint::Compute,
            self.shader_layout.clone(),
            shaders::OUTPUT_IMAGE_SET,
            vec![output_set],
        )?;

        let mut last_mode = None;
        let mut last_blend_settings = None;
        for (
            image_src,
            Blend {
                mode,
                alpha_clip,
                opacity,
            },
        ) in layers
        {
            // bind a new pipeline if changed from last iter
            if last_mode != Some(*mode) {
                let Some(program) = self.mode_pipelines.get(mode).map(Arc::clone) else {
                    anyhow::bail!("Blend mode {:?} unsupported", mode)
                };
                commands.bind_pipeline_compute(program)?;
                last_mode = Some(*mode);
            }
            // Push new clip/alpha constants if different from last iter
            // As per https://registry.khronos.org/vulkan/site/guide/latest/push_constants.html#pc-lifetime,
            // I believe push constants should remain across compatible pipeline binds
            let constants = shaders::BlendConstants::new(*opacity, *alpha_clip);
            if Some(constants) != last_blend_settings {
                commands.push_constants(self.shader_layout.clone(), 0, constants)?;
                last_blend_settings = Some(constants);
            }

            let input_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                self.shader_layout.set_layouts()[shaders::INPUT_IMAGE_SET as usize].clone(),
                [vk::WriteDescriptorSet::image_view(
                    0,
                    image_src.view().clone(),
                )],
                [],
            )?;
            commands
                .bind_descriptor_sets(
                    vk::PipelineBindPoint::Compute,
                    self.shader_layout.clone(),
                    shaders::INPUT_IMAGE_SET,
                    vec![input_set],
                )?
                .dispatch(
                    get_dispatch_size(image_src.view().image().extent())
                        .ok_or_else(|| anyhow::anyhow!("Image too large to blend!"))?,
                )?;
        }
        Ok(commands.build()?)
    }
}

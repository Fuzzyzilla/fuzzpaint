use std::{fmt::Debug, sync::Arc};

#[derive(strum::AsRefStr, PartialEq, Eq, strum::EnumIter, Copy, Clone, Hash, Debug)]
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

mod shaders {
    pub const INPUT_IMAGE_SET: u32 = 0;
    pub const OUTPUT_IMAGE_SET: u32 = 1;
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub struct WorkgroupSizeConstants {
        pub x: u32,
        pub y: u32,
    }
    // Safety: They all share the same source which unconditionally defines two constants:
    // workgroup size x:0, workgroup size y:1
    unsafe impl vulkano::shader::SpecializationConstants for WorkgroupSizeConstants {
        fn descriptors() -> &'static [vulkano::shader::SpecializationMapEntry] {
            &[
                vulkano::shader::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                },
                vulkano::shader::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4,
                },
            ]
        }
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

use vulkano::image::ImageAccess;

use crate::vk;
pub struct BlendEngine {
    workgroup_size: (u32, u32),
    // Based on chosen workgroup_size and device's max workgroup count
    max_image_size: (u32, u32),
    shader_layout: Arc<vk::PipelineLayout>,
    mode_pipelines: std::collections::HashMap<BlendMode, Arc<vk::ComputePipeline>>,
}
impl BlendEngine {
    fn build_pipeline(
        device: Arc<vk::Device>,
        layout: Arc<vk::PipelineLayout>,
        size: shaders::WorkgroupSizeConstants,
        entry_point: vulkano::shader::EntryPoint,
    ) -> anyhow::Result<Arc<vk::ComputePipeline>> {
        // Ensure that the unsafe assumptions of WorkgroupSizeConstants matches reality.
        #[cfg(debug_assertions)]
        {
            use vulkano::shader::SpecializationConstantRequirements as Req;
            let mut constants = entry_point.specialization_constant_requirements();
            debug_assert!(matches!(
                constants.next(),
                Some((1, Req { size: 4 })) | Some((0, Req { size: 4 }))
            ));
            debug_assert!(matches!(
                constants.next(),
                Some((1, Req { size: 4 })) | Some((0, Req { size: 4 }))
            ));
            debug_assert!(matches!(constants.next(), None));
        };

        let pipeline = vk::ComputePipeline::with_pipeline_layout(
            device.clone(),
            entry_point,
            &size,
            layout,
            None, // Cache would be ideal here, todo!
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
        // Max image size, based on max num of workgroups and chosen workgroup size
        let max_image_size = (
            (workgroup_size.0).saturating_mul(properties.max_compute_work_group_count[0]),
            (workgroup_size.1).saturating_mul(properties.max_compute_work_group_count[1]),
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
            vulkano::descriptor_set::layout::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                variable_descriptor_count: false,
                immutable_samplers: Default::default(),
                stages: vulkano::shader::ShaderStages::COMPUTE,
                ..vulkano::descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(
                    vulkano::descriptor_set::layout::DescriptorType::StorageImage,
                )
            },
        );
        let output_image_bindings = input_image_bindings.clone();

        let input_image_layout = vk::DescriptorSetLayout::new(
            device.clone(),
            vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                bindings: input_image_bindings,
                push_descriptor: false,
                ..Default::default()
            },
        )?;
        let output_image_layout = vk::DescriptorSetLayout::new(
            device.clone(),
            vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                bindings: output_image_bindings,
                push_descriptor: false,
                ..Default::default()
            },
        )?;

        let shader_layout = vk::PipelineLayout::new(
            device.clone(),
            vulkano::pipeline::layout::PipelineLayoutCreateInfo {
                set_layouts: vec![input_image_layout, output_image_layout],
                ..vulkano::pipeline::layout::PipelineLayoutCreateInfo::default()
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
                modes.insert(
                    $mode,
                    Self::build_pipeline(
                        device.clone(),
                        shader_layout.clone(),
                        size,
                        shaders::$namespace::load(device.clone())?
                            .entry_point("main")
                            .unwrap(),
                    )?,
                );
            };
        }
        
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
    /// Layers will be blended, in order, into mutable background.
    /// Background can be initialized to solid color, transparent, first layer, precomp'd image, ect!
    pub fn blend(
        &self,
        context: &crate::render_device::RenderContext,
        background: Arc<vk::ImageView<vk::StorageImage>>,
        layers: &[(BlendMode, Arc<vk::ImageView<vk::StorageImage>>)],
    ) -> anyhow::Result<vk::PrimaryAutoCommandBuffer> {
        if layers.is_empty() {
            anyhow::bail!("No layers to blend.")
        }

        // Compute the number of workgroups to dispatch for a given image
        // Or, None if the number of workgroups exceeds the maximum the device supports.
        let output_size = background.image().dimensions();
        let get_dispatch_size = |dimensions: vk::ImageDimensions| -> Option<[u32; 3]> {
            let x = dimensions.width().min(output_size.width());
            let y = dimensions.height().min(output_size.height());
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

        let mut commands = vk::AutoCommandBufferBuilder::primary(
            context.allocators().command_buffer(),
            context.queues().compute().idx(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        let output_set = vk::PersistentDescriptorSet::new(
            context.allocators().descriptor_set(),
            self.shader_layout.set_layouts()[shaders::OUTPUT_IMAGE_SET as usize].clone(),
            [vk::WriteDescriptorSet::image_view(0, background)],
        )?;
        commands.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Compute,
            self.shader_layout.clone(),
            shaders::OUTPUT_IMAGE_SET,
            vec![output_set],
        );

        let mut last_mode = None;
        for (mode, image) in layers {
            // Only bind a new pipeline if changed from last iter
            if last_mode != Some(*mode) {
                let Some(program) = self.mode_pipelines.get(mode).map(Arc::clone) else {
                    anyhow::bail!("Blend mode {:?} unsupported", mode)
                };
                commands.bind_pipeline_compute(program);
                last_mode = Some(*mode);
            }

            let input_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                self.shader_layout.set_layouts()[shaders::INPUT_IMAGE_SET as usize].clone(),
                [vk::WriteDescriptorSet::image_view(0, image.clone())],
            )?;
            commands
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    self.shader_layout.clone(),
                    shaders::INPUT_IMAGE_SET,
                    vec![input_set],
                )
                .dispatch(
                    get_dispatch_size(image.image().dimensions())
                        .ok_or_else(|| anyhow::anyhow!("Image too large to blend!"))?,
                )?;
        }
        Ok(commands.build()?)
    }
}

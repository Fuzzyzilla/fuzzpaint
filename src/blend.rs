use std::sync::Arc;

#[derive(strum::AsRefStr, PartialEq, Eq, strum::EnumIter, Copy, Clone, Hash)]
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

use crate::vk;
pub struct BlendEngine {
    workgroup_size: (u32, u32),
    // Based on chosen workgroup_size and max workgroup count
    max_image_size: (usize, usize),
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
        #[cfg(Debug)]
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
            None,
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
            workgroup_size.0 as usize * properties.max_compute_work_group_count[0] as usize,
            workgroup_size.1 as usize * properties.max_compute_work_group_count[1] as usize,
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

        modes.insert(
            BlendMode::Normal,
            Self::build_pipeline(
                device.clone(),
                shader_layout.clone(),
                size,
                shaders::normal::load(device.clone())?
                    .entry_point("main")
                    .unwrap(),
            )?,
        );
        modes.insert(
            BlendMode::Add,
            Self::build_pipeline(
                device.clone(),
                shader_layout.clone(),
                size,
                shaders::add::load(device.clone())?
                    .entry_point("main")
                    .unwrap(),
            )?,
        );
        modes.insert(
            BlendMode::Multiply,
            Self::build_pipeline(
                device.clone(),
                shader_layout.clone(),
                size,
                shaders::multiply::load(device.clone())?
                    .entry_point("main")
                    .unwrap(),
            )?,
        );
        modes.insert(
            BlendMode::Overlay,
            Self::build_pipeline(
                device.clone(),
                shader_layout.clone(),
                size,
                shaders::overlay::load(device.clone())?
                    .entry_point("main")
                    .unwrap(),
            )?,
        );
        Ok(Self {
            shader_layout,
            max_image_size,
            workgroup_size,
            mode_pipelines: modes,
        })
    }
}

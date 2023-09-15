use crate::vk;
use std::sync::Arc;
pub mod interface {
    #[derive(super::vk::Vertex, super::vk::BufferContents, Copy, Clone)]
    #[repr(C)]
    pub struct OutputStrokeVertex {
        #[format(R32G32_SFLOAT)]
        pub pos: [f32; 2],
        #[format(R32G32_SFLOAT)]
        pub uv: [f32; 2],
        #[format(R32G32B32A32_SFLOAT)]
        pub color: [f32; 4],
        #[format(R32_SFLOAT)]
        pub erase: f32,
    }
    #[derive(super::vk::Vertex, super::vk::BufferContents, Copy, Clone)]
    #[repr(C)]
    pub struct InputStrokeInfo {
        // Indices into inputStrokeVertices buffer
        #[format(R32_UINT)]
        pub start_point_idx: u32,
        #[format(R32_UINT)]
        pub num_points: u32,

        // Indices into outputStrokeVertices
        #[format(R32_UINT)]
        pub out_vert_offset: u32,
        #[format(R32_UINT)]
        pub out_vert_limit: u32,

        // Number of pixels between each stamp
        #[format(R32_SFLOAT)]
        pub density: f32,
        // The CPU will dictate how many groups to allocate to this work.
        // Mesh shaders would make this all nicer ;)
        #[format(R32_UINT)]
        pub start_group: u32,
        #[format(R32_UINT)]
        pub num_groups: u32,

        // Color and eraser settings
        #[format(R32G32B32A32_SFLOAT)]
        pub modulate: [f32; 4],
        // Bool32
        #[format(R32_UINT)]
        pub eraser: u32,
    }
    #[derive(super::vk::Vertex, super::vk::BufferContents, Copy, Clone)]
    #[repr(C)]
    pub struct InputStrokeVertex {
        #[format(R32G32_SFLOAT)]
        pub pos: [f32; 2],
        #[format(R32_SFLOAT)]
        pub pressure: f32,
        #[format(R32_SFLOAT)]
        pub dist: f32,
    }
    pub type OutputStrokeInfo = vulkano::command_buffer::DrawIndirectCommand;
}

mod shaders {
    pub mod tessellate {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "./src/shaders/tessellate_stamp.comp",
        }
    }
}

pub struct GpuStampTess {
    context: Arc<crate::render_device::RenderContext>,
    pipeline: Arc<vk::ComputePipeline>,
    input_descriptor: Arc<vk::DescriptorSetLayout>,
    output_descriptor: Arc<vk::DescriptorSetLayout>,
    layout: Arc<vk::PipelineLayout>,
}
impl GpuStampTess {
    fn make_layout(
        device: Arc<vk::Device>,
    ) -> anyhow::Result<(
        Arc<vk::PipelineLayout>,
        Arc<vk::DescriptorSetLayout>,
        Arc<vk::DescriptorSetLayout>,
    )> {
        // Interface consists of two sets, each with a two storage buffers.
        let buffer_binding = vulkano::descriptor_set::layout::DescriptorSetLayoutBinding {
            descriptor_count: 1,
            variable_descriptor_count: false,
            stages: vulkano::shader::ShaderStages::COMPUTE,
            ..vulkano::descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(
                vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
            )
        };

        let mut input_bindings = std::collections::BTreeMap::new();
        input_bindings.insert(0, buffer_binding.clone());
        input_bindings.insert(1, buffer_binding.clone());
        let output_bindings = input_bindings.clone();
        let inputs = vulkano::descriptor_set::layout::DescriptorSetLayout::new(
            device.clone(),
            vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                push_descriptor: false,
                bindings: input_bindings,
                ..Default::default()
            },
        )?;
        let outputs = vulkano::descriptor_set::layout::DescriptorSetLayout::new(
            device.clone(),
            vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                push_descriptor: false,
                bindings: output_bindings,
                ..Default::default()
            },
        )?;

        Ok((
            vk::PipelineLayout::new(
                device,
                vulkano::pipeline::layout::PipelineLayoutCreateInfo {
                    push_constant_ranges: Vec::new(),
                    set_layouts: vec![inputs.clone(), outputs.clone()],
                    ..Default::default()
                },
            )?,
            inputs,
            outputs,
        ))
    }
    pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        let shader = shaders::tessellate::load(context.device().clone())?;
        let entry = shader.entry_point("main").unwrap();
        let (layout, input_descriptor, output_descriptor) =
            Self::make_layout(context.device().clone())?;
        let pipeline = vk::ComputePipeline::with_pipeline_layout(
            context.device().clone(),
            entry,
            &shaders::tessellate::SpecializationConstants::default(),
            layout.clone(),
            None,
        )?;

        Ok(Self {
            context,
            layout,
            pipeline,
            input_descriptor,
            output_descriptor,
        })
    }
    /// Tessellate some strokes!
    /// `packed_points` should contain all the stroke data back-to-back, in order.
    /// Returns a semaphore for when the compute completes, the vertex buffer, and the draw indirection buffer.
    pub fn tess(
        &self,
        strokes: &[crate::ImmutableStroke],
        packed_points: vulkano::buffer::subbuffer::Subbuffer<[interface::InputStrokeVertex]>,
    ) -> anyhow::Result<(
        vulkano::sync::future::SemaphoreSignalFuture<impl vk::sync::GpuFuture>,
        vk::Subbuffer<[interface::OutputStrokeVertex]>,
        vk::Subbuffer<[interface::OutputStrokeInfo]>,
    )> {
        let mut point_index_counter = 0;
        let mut group_index_counter = 0;
        let mut vertex_output_index_counter = 0;
        let input_infos = vk::Buffer::from_iter(
            self.context.allocators().memory(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::INDIRECT_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                ..Default::default()
            },
            strokes.iter().map(|stroke| {
                const DENSITY: f32 = 1.0;

                let num_expected_stamps = stroke
                    .points
                    .last()
                    .map(|last| (last.dist / DENSITY).ceil() as u32)
                    .unwrap_or(0);
                let num_expected_verts = num_expected_stamps * 6;
                let num_groups = num_expected_stamps.div_ceil(32);

                let info = interface::InputStrokeInfo {
                    start_point_idx: point_index_counter,
                    num_points: stroke.points.len() as u32,
                    out_vert_offset: vertex_output_index_counter,
                    out_vert_limit: num_expected_verts,
                    start_group: group_index_counter,
                    num_groups,
                    modulate: stroke.brush.color_modulate,
                    density: DENSITY,
                    eraser: if stroke.brush.is_eraser { 1 } else { 0 },
                };

                point_index_counter += stroke.points.len() as u32;
                group_index_counter += num_groups;
                vertex_output_index_counter += num_expected_verts;

                info
            }),
        )?;
        let output_infos = vk::Buffer::new_slice::<interface::OutputStrokeInfo>(
            self.context.allocators().memory(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::INDIRECT_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                ..Default::default()
            },
            strokes.len() as u64,
        )?;
        let output_verts = vk::Buffer::new_slice::<interface::OutputStrokeVertex>(
            self.context.allocators().memory(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                ..Default::default()
            },
            vertex_output_index_counter as u64,
        )?;

        let input_descriptor = vk::PersistentDescriptorSet::new(
            self.context.allocators().descriptor_set(),
            self.input_descriptor.clone(),
            [
                vk::WriteDescriptorSet::buffer(0, input_infos),
                vk::WriteDescriptorSet::buffer(1, packed_points),
            ],
        )?;
        let output_descriptor = vk::PersistentDescriptorSet::new(
            self.context.allocators().descriptor_set(),
            self.input_descriptor.clone(),
            [
                vk::WriteDescriptorSet::buffer(0, output_infos.clone()),
                vk::WriteDescriptorSet::buffer(1, output_verts.clone()),
            ],
        )?;

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().compute().idx(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        command_buffer
            // Unwrap ok - Output infos is aligned to u32
            .fill_buffer(output_infos.clone().try_cast_slice().unwrap(), 0u32)?
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Compute,
                self.layout.clone(),
                0,
                (input_descriptor, output_descriptor),
            )
            .dispatch([group_index_counter, 1, 1])?;
        let command_buffer = command_buffer.build()?;

        use vk::sync::GpuFuture;
        let future = vk::sync::now(self.context.device().clone())
            .then_execute(
                self.context.queues().compute().queue().clone(),
                command_buffer,
            )?
            .then_signal_semaphore_and_flush()?;

        Ok((future, output_verts, output_infos))
    }
}

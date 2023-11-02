use crate::vulkano_prelude::*;
use std::sync::Arc;
pub mod interface {
    #[derive(super::vk::Vertex, super::vk::BufferContents, Copy, Clone, Debug)]
    // Match align with GLSL std430.
    #[repr(C, align(16))]
    pub struct OutputStrokeVertex {
        #[format(R32G32_SFLOAT)]
        pub pos: [f32; 2],
        #[format(R32G32_SFLOAT)]
        pub uv: [f32; 2],
        #[format(R32G32B32A32_SFLOAT)]
        pub color: [f32; 4],
        #[format(R32_SFLOAT)]
        pub erase: f32,
        #[format(R32G32B32_SFLOAT)]
        pub pad: [f32; 3],
    }
    pub type OutputStrokeInfo = vulkano::command_buffer::DrawIndirectCommand;
}

mod shaders {
    pub mod tessellate {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "./src/shaders/tessellate_stamp.comp",
        }

        impl From<crate::StrokePoint> for InputStrokeVertex {
            fn from(value: crate::StrokePoint) -> Self {
                Self {
                    dist: value.dist,
                    pos: value.pos,
                    pressure: value.pressure,
                }
            }
        }
    }
}

pub struct GpuStampTess {
    context: Arc<crate::render_device::RenderContext>,
    pipeline: Arc<vk::ComputePipeline>,
    input_descriptor: Arc<vk::DescriptorSetLayout>,
    output_descriptor: Arc<vk::DescriptorSetLayout>,
    layout: Arc<vk::PipelineLayout>,
    work_size: u32,
}
impl GpuStampTess {
    fn make_layout(
        device: Arc<vk::Device>,
    ) -> anyhow::Result<(
        Arc<vk::PipelineLayout>,
        Arc<vk::DescriptorSetLayout>,
        Arc<vk::DescriptorSetLayout>,
    )> {
        // Interface consists of two sets. Input with three buffers, output with two.
        let buffer_binding = vk::DescriptorSetLayoutBinding {
            descriptor_count: 1,
            stages: vk::ShaderStages::COMPUTE,
            ..vk::DescriptorSetLayoutBinding::descriptor_type(vk::DescriptorType::StorageBuffer)
        };

        let mut input_bindings = std::collections::BTreeMap::new();
        input_bindings.insert(0, buffer_binding.clone());
        input_bindings.insert(1, buffer_binding.clone());
        let output_bindings = input_bindings.clone();
        input_bindings.insert(2, buffer_binding.clone());

        let inputs = vk::DescriptorSetLayout::new(
            device.clone(),
            vk::DescriptorSetLayoutCreateInfo {
                bindings: input_bindings,
                ..Default::default()
            },
        )?;
        let outputs = vk::DescriptorSetLayout::new(
            device.clone(),
            vk::DescriptorSetLayoutCreateInfo {
                bindings: output_bindings,
                ..Default::default()
            },
        )?;

        Ok((
            vk::PipelineLayout::new(
                device,
                vk::PipelineLayoutCreateInfo {
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
        let properties = context.physical_device().properties();
        // Highest number of workers we're allowed to dispatch with [X, 1, 1] shape.
        let work_size: u32 = properties
            .max_compute_work_group_invocations
            .min(properties.max_compute_work_group_size[0]);

        let shader = shaders::tessellate::load(context.device().clone())?;
        // Specialize workgroup shape
        let mut specialize = ahash::HashMap::with_capacity_and_hasher(1, Default::default());
        specialize.insert(0, work_size.into());
        let entry = shader.specialize(specialize)?.entry_point("main").unwrap();
        log::info!("Tess workgroup size: {}", work_size);

        let (layout, input_descriptor, output_descriptor) =
            Self::make_layout(context.device().clone())?;

        let pipeline = vk::ComputePipeline::new(
            context.device().clone(),
            None,
            vk::ComputePipelineCreateInfo::stage_layout(
                vk::PipelineShaderStageCreateInfo::new(entry),
                layout.clone(),
            ),
        )?;

        Ok(Self {
            context,
            layout,
            pipeline,
            input_descriptor,
            output_descriptor,
            work_size,
        })
    }
    /// Tessellate some strokes
    /// Will automatically load stroke points into a buffer.
    pub fn tess(
        &self,
        strokes: &[&crate::state::stroke_collection::ImmutableStroke],
    ) -> anyhow::Result<(
        vk::SemaphoreSignalFuture<impl vk::sync::GpuFuture>,
        vk::Subbuffer<[interface::OutputStrokeVertex]>,
        vk::Subbuffer<[interface::OutputStrokeInfo]>,
    )> {
        let point_repo = crate::repositories::points::global();
        let count = strokes
            .iter()
            .map(|stroke| point_repo.len_of(stroke.point_collection))
            .try_fold(0, |fold, optional_length| {
                optional_length.and_then(|length| u64::checked_add(length as u64, fold))
            });
        // Would be nice to differentiate these errors, but currently we don't handle many errors
        // anyway...
        let count = count.ok_or_else(|| {
            anyhow::anyhow!("Packed point buffer too long, or contains invalid point collections!")
        })?;

        if count == 0 {
            anyhow::bail!("Tess invoked on zero points!")
        }
        let packed_points = vk::Buffer::new_slice::<shaders::tessellate::InputStrokeVertex>(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            count,
        )?;
        // Copy every vertex in order
        {
            let mut write = packed_points.write()?;
            let mut cursor = 0;
            for stroke in strokes.iter() {
                let points = point_repo.try_get(stroke.point_collection)?;
                write[cursor..(cursor + points.len())]
                    .iter_mut()
                    .zip(points.iter())
                    .for_each(|(into, from)| *into = (*from).into());
                cursor += points.len();
            }
        }

        self.tess_buffer(strokes, packed_points)
    }
    /// Tessellate some strokes!
    /// `packed_points` should contain all the stroke data back-to-back, in order.
    /// Returns a semaphore for when the compute completes, the vertex buffer, and the draw indirection buffer.
    pub fn tess_buffer(
        &self,
        strokes: &[&crate::state::stroke_collection::ImmutableStroke],
        packed_points: vk::Subbuffer<[shaders::tessellate::InputStrokeVertex]>,
    ) -> anyhow::Result<(
        vk::SemaphoreSignalFuture<impl GpuFuture>,
        vk::Subbuffer<[interface::OutputStrokeVertex]>,
        vk::Subbuffer<[interface::OutputStrokeInfo]>,
    )> {
        if strokes.is_empty() {
            anyhow::bail!("Tess invoked on empty zero strokes!")
        }
        let mut point_index_counter = 0;
        let mut group_index_counter = 0;
        let mut vertex_output_index_counter = 0;

        let mut num_groups_per_info = Vec::new();

        let point_repo = crate::repositories::points::global();

        let input_infos = vk::Buffer::from_iter(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            strokes.iter().map(|stroke| {
                let density = stroke.brush.spacing_px;
                let (num_expected_stamps, num_points) = {
                    // If not found, ignore by claiming 0 stamps.
                    if let Ok(points) = point_repo.try_get(stroke.point_collection) {
                        let dist = points
                            .last()
                            .map(|last| (last.dist / density).ceil() as u32)
                            .unwrap_or(0);
                        let num_points = points.len() as u32;

                        (dist, num_points)
                    } else {
                        (0, 0)
                    }
                };
                let num_expected_verts = num_expected_stamps * 6;
                let num_groups = num_expected_stamps.div_ceil(self.work_size);

                let info = shaders::tessellate::InputStrokeInfo {
                    start_point_idx: point_index_counter,
                    num_points,
                    out_vert_offset: vertex_output_index_counter,
                    out_vert_limit: num_expected_verts,
                    start_group: group_index_counter,
                    num_groups,
                    modulate: stroke.brush.color_modulate,
                    density,
                    size_mul: stroke.brush.size_mul,
                    is_eraser: if stroke.brush.is_eraser { 1.0 } else { 0.0 },
                };

                num_groups_per_info.push(num_groups);
                point_index_counter += num_points;
                group_index_counter += num_groups;
                vertex_output_index_counter += num_expected_verts;

                // Returning just info here results in misaligned structures.
                // This bug took SO long to find, thank you Marc I owe you my life.
                // the `12` magic comes from expansion of `inputStrokeInfo`
                vulkano::padded::Padded::<_, 12>::from(info)
            }),
        )?;

        if group_index_counter == 0 {
            anyhow::bail!("Stroke too short to tessellate.")
        }
        // One element per workgroup, telling it which info to work on.
        let input_map = vk::Buffer::new_slice::<u32>(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                // Transfer dest for clearing
                usage: vk::BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            group_index_counter as u64,
        )?;
        let mut current_idx = 0u32;
        input_map
            .write()?
            .iter_mut()
            .zip(num_groups_per_info.into_iter().flat_map(|num| {
                // For this stroke, a `num` groups are created.
                // Repeat `num` identical "pointers" to the info.
                current_idx += 1;
                std::iter::repeat(current_idx - 1).take(num as usize)
            }))
            .for_each(|(map, info_idx)| *map = info_idx);

        let output_infos = vk::Buffer::new_slice::<interface::OutputStrokeInfo>(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                // Transfer dest for clearing
                usage: vk::BufferUsage::STORAGE_BUFFER
                    | vk::BufferUsage::INDIRECT_BUFFER
                    | vk::BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            strokes.len() as u64,
        )?;
        let output_verts = vk::Buffer::new_slice::<interface::OutputStrokeVertex>(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_output_index_counter as u64,
        )?;

        let input_descriptor = vk::PersistentDescriptorSet::new(
            self.context.allocators().descriptor_set(),
            self.input_descriptor.clone(),
            [
                vk::WriteDescriptorSet::buffer(0, input_infos),
                vk::WriteDescriptorSet::buffer(1, input_map),
                vk::WriteDescriptorSet::buffer(2, packed_points),
            ],
            [],
        )?;
        let output_descriptor = vk::PersistentDescriptorSet::new(
            self.context.allocators().descriptor_set(),
            self.output_descriptor.clone(),
            [
                vk::WriteDescriptorSet::buffer(0, output_infos.clone()),
                vk::WriteDescriptorSet::buffer(1, output_verts.clone()),
            ],
            [],
        )?;

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().compute().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;

        command_buffer
            .fill_buffer(output_infos.clone().reinterpret(), 0u32)?
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                vk::PipelineBindPoint::Compute,
                self.layout.clone(),
                0,
                (input_descriptor, output_descriptor),
            )?
            .dispatch([group_index_counter, 1, 1])?;
        let command_buffer = command_buffer.build()?;

        log::trace!(
            "Dispatched {} tessellation workgroups for {} strokes",
            group_index_counter,
            strokes.len()
        );

        let future = vk::sync::now(self.context.device().clone())
            .then_execute(
                self.context.queues().compute().queue().clone(),
                command_buffer,
            )?
            .then_signal_semaphore();

        Ok((future, output_verts, output_infos))
    }
}

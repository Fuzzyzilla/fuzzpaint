use std::sync::Arc;

mod egui_impl;
pub mod gpu_err;
pub mod vulkano_prelude;
pub mod window;
use gpu_err::GpuResult;
use rand::{SeedableRng, Rng};
use vulkano::command_buffer;
use vulkano_prelude::*;
pub mod render_device;
pub mod stylus_events;

use anyhow::Result as AnyResult;

#[derive(vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct StrokePointUnpacked {
    #[format(R32G32_SFLOAT)]
    pos: [f32;2],
    #[format(R32_SFLOAT)]
    pressure: f32,
}

mod test_renderer_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
        #version 460

        layout(push_constant) uniform Matrix {
            mat4 mvp;
        } push_matrix;

        layout(location = 0) in vec2 pos;
        layout(location = 1) in float pressure;

        layout(location = 0) flat out vec4 color;

        void main() {
            vec4 position_2d = push_matrix.mvp * vec4(pos, 0.0, 1.0);
            color = vec4(vec3(pressure), 1.0);
            gl_Position = vec4(position_2d.xy, 0.0, 1.0);
        }"
    }
}mod test_renderer_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: r"
        #version 460
        layout(location = 0) flat in vec4 color;

        layout(location = 0) out vec4 out_color;

        void main() {
            out_color = color;
        }"
    }
}
mod dist_field_comp {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
        #version 460
        layout(set = 0, binding = 0, rgba16ui) uniform restrict readonly uimage2D inBuffer;
        layout(set = 1, binding = 0, rgba16ui) uniform restrict writeonly uimage2D outBuffer;
        
        layout(constant_id = 0) const uint step_size = 1;
        
        #define SENTINAL (uint(65535))
        const uvec2 SENTINAL2 = uvec2(SENTINAL);
        
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        void main() {
            ivec2 self_position = ivec2(gl_GlobalInvocationID.xy);
            uvec2 size = imageSize(inBuffer).xy;
        
            //Self is outside of the image, skip!
            if (any(greaterThanEqual(self_position, size))) return;
        
            uvec4 self_value = uvec4(SENTINAL2, SENTINAL2);
        
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    ivec2 sample_pos = ivec2(i, j) * int(step_size) + self_position;
                    uvec4 sample_value = uvec4(SENTINAL2, SENTINAL2);
        
                    // Perform sample!
                    if (
                        all(greaterThanEqual(sample_pos, ivec2(0)))
                        && all(lessThan(sample_pos, ivec2(size)))
                    ){
                        sample_value = imageLoad(inBuffer, sample_pos);
                    }
        
                    // We found an inner pixel! Compare with stored value, choose the closest one.
                    if (sample_value.r != SENTINAL) {
                        // There wasn't a pixel before, so take this one.
                        if (sample_value.r == SENTINAL) {
                            self_value.rg = sample_value.rg;
                        } else {
                            //Compare distances - both are non-sentinal so we must choose closest:
                            ivec2 to_self = self_position - ivec2(self_value.rg);
                            ivec2 to_sample = self_position - ivec2(sample_value.rg);
        
                            // Fast dist compare - is sample closer than self?
                            if (dot(to_sample, to_sample) < dot(to_self, to_self)) {
                                self_value.rg = sample_value.rg;
                            }
                        }
                    }
        
                    // We found an outer pixel! Compare with stored value, choose the closest one.
                    if (sample_value.b != SENTINAL) {
                        // There wasn't a pixel before, so take this one.
                        if (sample_value.b == SENTINAL) {
                            self_value.ba = sample_value.ba;
                        } else {
                            //Compare distances - both are non-sentinal so we must choose closest:
                            ivec2 to_self = self_position - ivec2(self_value.ba);
                            ivec2 to_sample = self_position - ivec2(sample_value.ba);
        
                            // Fast dist compare - is sample closer than self?
                            if (dot(to_sample, to_sample) < dot(to_self, to_self)) {
                                self_value.ba = sample_value.ba;
                            }
                        }
                    }
                }
            }
        
            imageStore(outBuffer, self_position, uvec4(self_value));
        }
        "
    }
}mod dist_field_init {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r#"
        #version 460
        layout(set = 0, binding = 0, rgba16f) uniform restrict readonly image2D inBuffer;
        layout(set = 1, binding = 0, rgba16ui) uniform restrict writeonly uimage2D outBuffer;
        const uvec2 SENTINAL = uvec2(65535);

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        void main() {
            uvec2 self_position = gl_GlobalInvocationID.xy;
            uvec2 size = imageSize(inBuffer).xy;
            //Outside image bound, skip.
            if (any(greaterThanEqual(self_position, size))) return;

            // Initialize an image with sentinal values,
            // except for where the color image is opaque.
            // There, initialize with pixel coords.

            float self_alpha = imageLoad(inBuffer, ivec2(self_position)).a;
            bool is_inside = self_alpha > 0.1;

            //RG -> coords of the Nearest "Inside" pixel
            //BA -> coords of the Nearest "Outside" pixel
            uvec4 value = is_inside ? uvec4(self_position, SENTINAL) : uvec4(SENTINAL, self_position);

            imageStore(outBuffer, ivec2(self_position), value);
        }
        "#
    }
}

const DOCUMENT_DIMENSION : u32 = 1080;
fn make_voronoi(
    render_context: Arc<render_device::RenderContext>,
    color_image: Arc<vk::StorageImage>,
    future: vk::sync::future::SemaphoreSignalFuture<impl vk::sync::GpuFuture>
) -> AnyResult<(Arc<vk::ImageView<impl vk::ImageAccess>>, vk::sync::future::FenceSignalFuture<impl vk::sync::GpuFuture>, Arc<vulkano::query::QueryPool>)> {
    //dimensions of color image must be strictly less than 65535
    let voronoi_format = vk::Format::R16G16B16A16_UINT;
    const KERNEL_SIZE : [u16; 2] = [8u16; 2];

    let pingpong_buffers_array = 
        vk::StorageImage::with_usage(
            render_context.allocators().memory(),
            vulkano::image::ImageDimensions::Dim2d { width: DOCUMENT_DIMENSION, height: DOCUMENT_DIMENSION, array_layers: 2 },
            voronoi_format,
            //TRANSFER_DST required for vkCmdClearColorImage
            vk::ImageUsage::STORAGE | vk::ImageUsage::TRANSFER_DST,
            vk::ImageCreateFlags::empty(),
            [render_context.queues().compute().idx()],
        )?;
    let pingpong_views = [
        vk::ImageView::new(
            pingpong_buffers_array.clone(),
            vk::ImageViewCreateInfo{
                subresource_range: vk::ImageSubresourceRange {
                    array_layers: 0..1,
                    aspects: vk::ImageAspects::COLOR,
                    mip_levels: 0..1,
                },
                format: Some(voronoi_format),
                view_type: vulkano::image::ImageViewType::Dim2d,
                ..Default::default()
            }
        )?,
        vk::ImageView::new(
            pingpong_buffers_array.clone(),
            vk::ImageViewCreateInfo{
                subresource_range: vk::ImageSubresourceRange {
                    array_layers: 1..2,
                    aspects: vk::ImageAspects::COLOR,
                    mip_levels: 0..1,
                },
                format: Some(voronoi_format),
                view_type: vulkano::image::ImageViewType::Dim2d,
                ..Default::default()
            }
        )?,
    ];
    let color_view = vk::ImageView::new_default(color_image.clone())?;

    
    // Set up explicit pipeline layout to ensure all stages are perfectly compatible.
    let storage_image_binding =
        vulkano::descriptor_set::layout::DescriptorSetLayoutBinding{
            stages: vulkano::shader::ShaderStages::COMPUTE,
            ..vulkano::descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(vulkano::descriptor_set::layout::DescriptorType::StorageImage)
        };

    // Input and output binding have the same exact contents
    let mut input_pingpong_binding = std::collections::BTreeMap::new();
    input_pingpong_binding.insert(0, storage_image_binding.clone());
    
    let mut output_pingpong_binding = std::collections::BTreeMap::new();
    output_pingpong_binding.insert(0, storage_image_binding.clone());

    // binding for the init stage, taking in a color image.
    let mut input_color_binding = std::collections::BTreeMap::new();
    input_color_binding.insert(0, storage_image_binding.clone());

    // Input and output are on different sets, but have the same layout.
    let input_pingpong_layout = vulkano::descriptor_set::layout::DescriptorSetLayout::new(
        render_context.device().clone(),
        vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
            bindings: input_pingpong_binding,
            push_descriptor: false,
            ..Default::default()
        }
    )?;
    let output_pingpong_layout = vulkano::descriptor_set::layout::DescriptorSetLayout::new(
        render_context.device().clone(),
        vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
            bindings: output_pingpong_binding,
            push_descriptor: false,
            ..Default::default()
        }
    )?;
    // For init stage
    let input_color_layout = vulkano::descriptor_set::layout::DescriptorSetLayout::new(
        render_context.device().clone(),
        vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
            bindings: input_color_binding,
            push_descriptor: false,
            ..Default::default()
        }
    )?;

    // Full layout
    let pingpong_layout = vulkano::pipeline::layout::PipelineLayout::new(
        render_context.device().clone(),
        vulkano::pipeline::layout::PipelineLayoutCreateInfo{
            push_constant_ranges: Vec::new(), // empty!
            set_layouts: vec![
                input_pingpong_layout.clone(),
                output_pingpong_layout.clone()
            ],
            ..Default::default()
        }
    )?;

    // First set is the color input, second set is the same pingpong output.
    let init_layout = vulkano::pipeline::layout::PipelineLayout::new(
        render_context.device().clone(),
        vulkano::pipeline::layout::PipelineLayoutCreateInfo{
            push_constant_ranges: Vec::new(), // empty!
            set_layouts: vec![
                input_color_layout.clone(),
                output_pingpong_layout.clone()
            ],
            ..Default::default()
        }
    )?;

    // Shader to init pingpong buffers from color image.
    let init_shader = dist_field_init::load(render_context.device().clone())?;

    let init_pipeline = vk::ComputePipeline::with_pipeline_layout(
        render_context.device().clone(),
        init_shader.entry_point("main").unwrap(),
        &dist_field_init::SpecializationConstants::default(),
        init_layout.clone(),
        None)?;
    
    let step_sizes : Vec<u32> = {
        // Find the previous power of two.
        // Step size should not step over the whole image, but it should be
        // the largest power of two that will fit.
        // WELL THAT'S A HANDY BUILTIN INNIT
        let mut step_size = DOCUMENT_DIMENSION.next_power_of_two() >> 1;

        //Each power of two from step_size down to 1
        std::iter::from_fn(move || {
            if step_size == 0 {
                None
            } else {
                let this_size = step_size;
                step_size >>= 1;

                Some(this_size)
            }
        }).collect()
    };

    //Shader to pingpong until we reach a full voronoi diagrams

    let step_shader = dist_field_comp::load(render_context.device().clone())?;
    let step_shader_entry = step_shader.entry_point("main").unwrap();

    let pingpong_pipelines : AnyResult<Vec<_>> = step_sizes.into_iter()
        .map(|size| -> AnyResult<(u32, Arc<vk::ComputePipeline>)> {
            Ok(
                (
                    size,
                    vk::ComputePipeline::with_pipeline_layout(
                        render_context.device().clone(),
                        step_shader_entry.clone(),
                        &dist_field_comp::SpecializationConstants{
                            step_size: size
                        },
                        pingpong_layout.clone(),
                        //No cache yet
                        None
                    )?
                )
            )
        }).collect();
    
    let pingpong_pipelines = pingpong_pipelines?;

    // Descriptors with explicit layout
    let input_color_descriptor = vk::PersistentDescriptorSet::new(
        render_context.allocators().descriptor_set(),
        input_pingpong_layout.clone(),
        [
            vk::WriteDescriptorSet::image_view(0, color_view.clone())
        ]
    )?;

    // for using view 0 or 1 as input or output (for pingpong-ing)
    let input_pingpong_descriptors = [
        vk::PersistentDescriptorSet::new(
            render_context.allocators().descriptor_set(),
            input_pingpong_layout.clone(),
            [
                vk::WriteDescriptorSet::image_view(0, pingpong_views[0].clone())
            ]
        )?,
        vk::PersistentDescriptorSet::new(
            render_context.allocators().descriptor_set(),
            input_pingpong_layout.clone(),
            [
                vk::WriteDescriptorSet::image_view(0, pingpong_views[1].clone())
            ]
        )?,
    ];

    let output_pingpong_descriptors = [
        vk::PersistentDescriptorSet::new(
            render_context.allocators().descriptor_set(),
            output_pingpong_layout.clone(),
            [
                vk::WriteDescriptorSet::image_view(0, pingpong_views[0].clone())
            ]
        )?,
        vk::PersistentDescriptorSet::new(
            render_context.allocators().descriptor_set(),
            output_pingpong_layout.clone(),
            [
                vk::WriteDescriptorSet::image_view(0, pingpong_views[1].clone())
            ]
        )?,
    ];

    let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
        render_context.allocators().command_buffer(),
        render_context.queues().compute().idx(),
        // for now.... Could re-use the queues and images and views and...
        vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
    )?;

    //Why unstable :V
    let div_ciel = |num : u32, denom : u32| -> u32 {
        (num + denom - 1) / denom
    };
    let dispatch_size = {
        [
            div_ciel(DOCUMENT_DIMENSION, KERNEL_SIZE[0] as u32),
            div_ciel(DOCUMENT_DIMENSION, KERNEL_SIZE[1] as u32),
            1
        ]
    };

    let mut previous_output_pingpong = 0;
    
    let second_view_region = vk::ImageSubresourceRange{
        array_layers: 1..2,
        aspects: vk::ImageAspects::COLOR,
        mip_levels: 0..1,
    };

    let performance_queries = vulkano::query::QueryPool::new(
        render_context.device().clone(),
        vulkano::query::QueryPoolCreateInfo {
            query_count: 3,
            ..vulkano::query::QueryPoolCreateInfo::query_type(vulkano::query::QueryType::Timestamp)
        }
    )?;

    //Safety: query 0 is not accessed anywhere but here.
    unsafe {
        command_buffer.write_timestamp(performance_queries.clone(), 0, vulkano::sync::PipelineStage::TopOfPipe)?;
    }

    //Weirdly, there are no barrier commands available - are they automagically inserted?
    command_buffer.bind_pipeline_compute(init_pipeline)
        .bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Compute,
            init_layout.clone(),
            0,
            vec![
                input_color_descriptor.clone(),
                output_pingpong_descriptors[0].clone()
            ]
        )
        .dispatch(dispatch_size)?
        //Clear the second buffer to all sentinal values.
        //Can't clear an ImageView, so clear the region corresponding to that view.
        .clear_color_image(vk::ClearColorImageInfo{
            clear_value: vulkano::format::ClearColorValue::Uint([u16::MAX as u32; 4]),
            regions: smallvec::smallvec![second_view_region],
            ..vk::ClearColorImageInfo::image(pingpong_buffers_array.clone())
        })?;

    //Safety: query 1 is not accessed anywhere but here.
    unsafe {
        command_buffer.write_timestamp(performance_queries.clone(), 1, vulkano::sync::PipelineStage::AllCommands)?;
    }


    for (size, pipeline) in pingpong_pipelines {
        let (in_desc, out_desc) = if previous_output_pingpong == 0 {
            // First pingpong was output, now is input!
            previous_output_pingpong = 1;
            (
                input_pingpong_descriptors[0].clone(),
                output_pingpong_descriptors[1].clone()
            )
        } else {
            // Second pingpong was output, now is input!
            previous_output_pingpong = 0;
            (
                input_pingpong_descriptors[1].clone(),
                output_pingpong_descriptors[0].clone()
            )
        };

        command_buffer
            .bind_pipeline_compute(pipeline)
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Compute,
                pingpong_layout.clone(),
                0,
                vec! [
                    in_desc,
                    out_desc,
                ],
            )
            .dispatch(dispatch_size)?;
    }

    //Safety: query 2 is not accessed anywhere but here.
    unsafe {
        command_buffer.write_timestamp(performance_queries.clone(), 2, vulkano::sync::PipelineStage::BottomOfPipe)?;
    }

    let command_buffer = command_buffer.build()?;

    let fence = future
        .then_execute(render_context.queues().compute().queue().clone(), command_buffer)?
        .then_signal_fence_and_flush()?;

    Ok(
        (
            pingpong_views[previous_output_pingpong].clone(),
            fence,
            performance_queries
        )
    )
}

fn make_test_image(render_context: Arc<render_device::RenderContext>) -> AnyResult<(Arc<vk::StorageImage>, vk::sync::future::SemaphoreSignalFuture<impl vk::sync::GpuFuture>)> {
    let document_format = vk::Format::R16G16B16A16_SFLOAT;
    let document_dimension = DOCUMENT_DIMENSION;
    let document_buffer = vk::StorageImage::with_usage(
        render_context.allocators().memory(),
        vulkano::image::ImageDimensions::Dim2d { width: document_dimension, height: document_dimension, array_layers: 1 },
        document_format,
        vk::ImageUsage::COLOR_ATTACHMENT | vk::ImageUsage::SAMPLED | vk::ImageUsage::STORAGE,
        vk::ImageCreateFlags::empty(),
        [render_context.queues().graphics().idx()]
    )?;

    let document_view = vk::ImageView::new_default(document_buffer.clone())?;

    let render_pass = vulkano::single_pass_renderpass!(
        render_context.device().clone(),
        attachments: {
            document: {
                load: Clear,
                store: Store,
                format: document_format,
                samples: 1,
            },
        },
        pass: {
            color: [document],
            depth_stencil: {},
        },
    )?;
    let document_framebuffer = vk::Framebuffer::new(
        render_pass.clone(),
        vk::FramebufferCreateInfo {
            attachments: vec![
                document_view
            ],
            ..Default::default()
        }
    )?;

    let vert = test_renderer_vert::load(render_context.device().clone())?;
    let frag = test_renderer_frag::load(render_context.device().clone())?;

    let pipeline = vk::GraphicsPipeline::start()
        .render_pass(render_pass.first_subpass())
        .vertex_shader(vert.entry_point("main").unwrap(), test_renderer_vert::SpecializationConstants::default())
        .fragment_shader(frag.entry_point("main").unwrap(), test_renderer_frag::SpecializationConstants::default())
        .vertex_input_state(StrokePointUnpacked::per_vertex())
        .input_assembly_state(vk::InputAssemblyState {
            topology: vk::PartialStateMode::Fixed(vk::PrimitiveTopology::LineStrip),
            ..Default::default()
        })
        .color_blend_state(vk::ColorBlendState::default().blend_alpha())
        .rasterization_state(vk::RasterizationState::default())
        .viewport_state(
            vk::ViewportState::viewport_fixed_scissor_irrelevant(
                [
                    vk::Viewport{
                        depth_range: 0.0..1.0,
                        dimensions: [document_dimension as f32, document_dimension as f32],
                        origin: [0.0; 2],
                    }
                ]
            )
        )
        .build(render_context.device().clone())?;

    /* square
    let points = [
        StrokePointUnpacked {
            pos: [100.0, 100.0],
            pressure: 0.1,
        },
        StrokePointUnpacked {
            pos: [1000.0, 100.0],
            pressure: 0.8,
        },
        StrokePointUnpacked {
            pos: [1000.0, 1000.0],
            pressure: 0.4,
        },
        StrokePointUnpacked {
            pos: [100.0, 1000.0],
            pressure: 1.0,
        },
        StrokePointUnpacked {
            pos: [100.0, 100.0],
            pressure: 0.1,
        },
    ];*/
    let points = {    
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let mut rand_point = move || {
            StrokePointUnpacked {
                pos: [rng.gen_range(0.0..1080.0), rng.gen_range(0.0..1080.0)],
                pressure: rng.gen_range(0.0..1.0)
            }
        };

        [   
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
            rand_point(),
        ]
    };
    let points_buf = vk::Buffer::from_data(
        render_context.allocators().memory(),
        vulkano::buffer::BufferCreateInfo {
            usage: vk::BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        vulkano::memory::allocator::AllocationCreateInfo { usage: vk::MemoryUsage::Upload, ..Default::default() },
        points
    )?;
    let pipeline_layout = pipeline.layout().clone();
    let matrix = cgmath::ortho(0.0, document_dimension as f32, document_dimension as f32, 0.0, -1.0, 0.0);

    let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            render_context.allocators().command_buffer(),
            render_context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit
        )?;
    command_buffer.begin_render_pass(vk::RenderPassBeginInfo{
            clear_values: vec![
                Some(vk::ClearValue::Float([0.0; 4]))
            ],
            ..vk::RenderPassBeginInfo::framebuffer(document_framebuffer)
        }, vk::SubpassContents::Inline)?
        .bind_pipeline_graphics(pipeline)
        .push_constants(pipeline_layout, 0,
            test_renderer_vert::Matrix{
                mvp: matrix.into()
            }
        )
        .bind_vertex_buffers(0, [points_buf])
        .draw(points.len() as u32, 1, 0, 0)?
        .end_render_pass()?;
    let command_buffer = command_buffer.build()?;
    
    let image_rendered_semaphore = render_context.now()
        .then_execute(render_context.queues().graphics().queue().clone(), command_buffer)?
        .then_signal_semaphore_and_flush()?;

    Ok(
        (document_buffer, image_rendered_semaphore)
    )
}

fn load_document_image(
    render_context: Arc<render_device::RenderContext>,
    path: &std::path::Path
) -> AnyResult<(Arc<vk::StorageImage>, vk::sync::future::SemaphoreSignalFuture<impl vk::sync::GpuFuture>)> {
    let image = image::open(path)?;
    if image.width() != DOCUMENT_DIMENSION || image.height() != DOCUMENT_DIMENSION {
        anyhow::bail!(
            "Wrong image size"
        );
    }
    let image = image.into_rgba8();

    let image_buffer = vk::Buffer::from_iter(
        render_context.allocators().memory(),
        vk::BufferCreateInfo {
            usage: vk::BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        vk::AllocationCreateInfo {
            usage: vk::MemoryUsage::Upload,
            ..Default::default()
        },
        image.into_raw().iter()
            .map(|&component| {
                vulkano::half::f16::from_f32(component as f32 / 255.0)
            })
    )?;

    let image = vk::StorageImage::new(
        render_context.allocators().memory(),
        vk::ImageDimensions::Dim2d { width: DOCUMENT_DIMENSION, height: DOCUMENT_DIMENSION, array_layers: 1 },
        vk::Format::R16G16B16A16_SFLOAT,
        [
            render_context.queues().compute().idx(),
        ],
    )?;

    let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
        render_context.allocators().command_buffer(),
        render_context.queues().compute().idx(),
        vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit
    )?;

    command_buffer
        .copy_buffer_to_image(vk::CopyBufferToImageInfo::buffer_image(image_buffer, image.clone()))?;

    let command_buffer = command_buffer.build()?;

    let future =
        render_context.now()
            .then_execute(render_context.queues().compute().queue().clone(), command_buffer)?
            .then_signal_semaphore_and_flush()?;
    
    Ok(
        (
            image,
            future
        )
    )
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    env_logger::init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    //let (image, future) = make_test_image(render_context.clone())?;
    let (image, future) = load_document_image(render_context.clone(), &std::path::PathBuf::from("test-data/orbs.png"))?;
    let (voronoi, future, timeline_queries) = make_voronoi(render_context.clone(), image, future)?;

    let window_renderer = window_surface.with_render_surface(render_surface, render_context.clone())?;
    println!("Made render context!");

    future.wait(None)?;

    let mut results = [0u64; 3];

    //Timeline queries will be done now!
    timeline_queries.queries_range(0..3).unwrap()
        .get_results(&mut results, vulkano::query::QueryResultFlags::WAIT)?;

    let init_time = results[1] - results[0];
    let pass_time = results[2] - results[1];

    let nanos_per_tick = render_context.physical_device().properties().timestamp_period;
    let init_nanos = init_time as f32 * nanos_per_tick;
    let pass_nanos = pass_time as f32 * nanos_per_tick;

    println!("Init took {}us, passes took {}us", init_nanos / 1000.0, pass_nanos / 1000.0);

    window_renderer.run();
}

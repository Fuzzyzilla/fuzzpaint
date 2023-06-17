#![feature(array_chunks)]

use std::{sync::Arc, ops::Deref};
mod egui_impl;
pub mod gpu_err;
pub mod vulkano_prelude;
pub mod window;
use cgmath::Matrix4;
use gpu_err::GpuResult;
use rand::{SeedableRng, Rng};
use vulkano::{command_buffer::{self, AutoCommandBufferBuilder}, format};
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

const DOCUMENT_DIMENSION : u32 = 1080;

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
) -> AnyResult<(Arc<vk::StorageImage>, vk::sync::future::FenceSignalFuture<impl vk::sync::GpuFuture>)> {
    let image = image::open(path)?;
    if image.width() != DOCUMENT_DIMENSION || image.height() != DOCUMENT_DIMENSION {
        anyhow::bail!(
            "Wrong image size"
        );
    }
    let image = image.into_rgba8();

    let image_data : Vec<_> = 
        image.into_vec()
            .array_chunks::<4>()
            .flat_map(|&[r, g, b, a]| {
                // Lol algebraic optimization
                let a = a as f32 / 255.0 / 255.0;
                let rgba = [
                    r as f32 * a,
                    g as f32 * a,
                    b as f32 * a,
                    a * 255.0,
                ];
                rgba
            })
            .map(vulkano::half::f16::from_f32)
            .collect();

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
        image_data.into_iter()
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
            .then_signal_fence_and_flush()?;
    
    Ok(
        (
            image,
            future
        )
    )
}

/// Proxy called into by the window renderer to perform the necessary synchronization and such to render the screen
/// behind the Egui content.
trait PreviewRenderProxy {
    /// Create the render commands for this frame. Assume used resources are borrowed until a matching "render_complete" for this
    /// frame idx is called.
    fn render(&mut self, swapchain_image_idx: u32)
        -> AnyResult<Arc<vk::PrimaryAutoCommandBuffer>>;

    /// When the future of a previous render has completed
    fn render_complete(&mut self, idx : u32);
    fn surface_changed(&mut self, render_surface: &render_device::RenderSurface);
}


mod document_preview_shaders{
    pub mod vertex {
        vulkano_shaders::shader!{
            ty: "vertex",
            src:r"
            #version 460
            
            layout(push_constant) uniform Matrix {
                mat4 mat;
            } matrix;

            layout(location = 0) in vec2 pos;

            layout(location = 0) out vec2 out_uv;

            void main() {
                out_uv = vec2(pos.x, 1.0 - pos.y);
                gl_Position = matrix.mat * vec4(pos, 0.0, 1.0);
            }"
        }
    }
    pub mod fragment {
        vulkano_shaders::shader!{
            ty: "fragment",
            src:r"
            #version 460

            layout(set = 0, binding = 0) uniform sampler2D image;

            layout(location = 0) in vec2 uv;

            layout(location = 0) out vec4 color;

            void main() {
                vec4 col = texture(image, uv);
                col.rgb *= col.rgb;
                col.rgb *= col.a;
                color = col;
            }"
        }
    }
    pub mod transparency_grid {
        vulkano_shaders::shader!{
            ty: "fragment",
            src:r"
            #version 460

            const float LIGHT = 0.8;
            const float DARK = 0.7;
            const uint SIZE = uint(16);

            layout(location = 0) in vec2 _; ///Dummy input for interface matching

            layout(location = 0) out vec4 color;

            void main() {
                uvec2 grid_coords = uvec2(gl_FragCoord.xy) / SIZE;
                bool is_light = (grid_coords.x + grid_coords.y) % 2 == 0;

                color = vec4(vec3(is_light ? LIGHT : DARK), 1.0);
            }"
        }
    }
}

#[derive(vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct DocumentVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
}
struct DocumentViewportPreviewProxy {
    render_context: Arc<render_device::RenderContext>,

    test_image_binding : Arc<vk::PersistentDescriptorSet>,

    render_pass: Arc<vk::RenderPass>,
    // List of framebuffers for the swapchain, lazily created as they're needed.
    framebuffers: Vec<Arc<vk::Framebuffer>>,
    prerecorded_command_buffers: Vec<Arc<vk::PrimaryAutoCommandBuffer>>,

    viewport_dimensions: [u32; 2],

    pipeline: Arc<vk::GraphicsPipeline>,
    transparency_pipeline: Arc<vk::GraphicsPipeline>,

    transform_matrix: [[f32; 4]; 4],
    vertex_buffer: vulkano::buffer::Subbuffer<[DocumentVertex; 4]>,
    index_buffer: vulkano::buffer::Subbuffer<[u16]>,

}
impl DocumentViewportPreviewProxy {
    fn new(render_surface: &render_device::RenderSurface, image: Arc<vk::StorageImage>)
        -> AnyResult<Self> {
        let render_pass = vulkano::single_pass_renderpass!(
            render_surface.context().device().clone(),
            attachments: {
                document: {
                    load: Clear,
                    store: Store,
                    format: render_surface.format(),
                    samples: 1,
                },
            },
            pass: {
                color: [document],
                depth_stencil: {},
            },
        )?;
        let test_image = vk::ImageView::new_default(image)?;
        let sampler = vk::Sampler::new(
            render_surface.context().device().clone(),
            vk::SamplerCreateInfo{
                min_filter: vk::Filter::Linear,
                mag_filter: vk::Filter::Nearest,
                ..Default::default()
            }
        )?;

        let vertices = [
            DocumentVertex {
                pos: [0.0, 0.0],
            },
            DocumentVertex {
                pos: [1.0, 0.0],
            },
            DocumentVertex {
                pos: [0.0, 1.0],
            },
            DocumentVertex {
                pos: [1.0, 1.0],
            },
        ];
        let indices : [u16; 6] = [
            0, 1, 2,
            1, 2, 3,
        ];
    
        let vertex_staging_buf = vk::Buffer::from_data(
            render_surface.context().allocators().memory(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                usage: vulkano::memory::allocator::MemoryUsage::Upload,
                ..Default::default()
            },
            vertices
        )?;
        let index_staging_buf = vk::Buffer::from_iter(
            render_surface.context().allocators().memory(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                usage: vulkano::memory::allocator::MemoryUsage::Upload,
                ..Default::default()
            },
            indices.into_iter()
        )?;

        let vertex_shader = document_preview_shaders::vertex::load(render_surface.context().device().clone())?;
        let fragment_shader = document_preview_shaders::fragment::load(render_surface.context().device().clone())?;
        let transparency_grid = document_preview_shaders::transparency_grid::load(render_surface.context().device().clone())?;
        let vertex_shader = vertex_shader.entry_point("main").unwrap();
        let fragment_shader = fragment_shader.entry_point("main").unwrap();
        let transparency_grid = transparency_grid.entry_point("main").unwrap();

        let mut blend_premul = vk::ColorBlendState::new(1);
        blend_premul.attachments[0].blend = Some(vk::AttachmentBlend{
            alpha_source: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
            color_source: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
            alpha_destination: vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
            color_destination: vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
            alpha_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
            color_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
        });

        let size = render_surface.extent();

        let pipeline = vk::GraphicsPipeline::start()
            .vertex_shader(vertex_shader.clone(), document_preview_shaders::vertex::SpecializationConstants::default())
            .fragment_shader(fragment_shader, document_preview_shaders::fragment::SpecializationConstants::default())
            .vertex_input_state(DocumentVertex::per_vertex())
            .rasterization_state(vk::RasterizationState::default().cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::None))
            .color_blend_state(blend_premul.clone())
            .render_pass(render_pass.clone().first_subpass())
            .viewport_state(vk::ViewportState::viewport_dynamic_scissor_irrelevant())
            .build(render_surface.context().device().clone())?;

        let transparency_pipeline = vk::GraphicsPipeline::start()
            .vertex_shader(vertex_shader, document_preview_shaders::vertex::SpecializationConstants::default())
            .fragment_shader(transparency_grid, document_preview_shaders::transparency_grid::SpecializationConstants::default())
            .vertex_input_state(DocumentVertex::per_vertex())
            .rasterization_state(vk::RasterizationState::default().cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::None))
            .color_blend_state(blend_premul.clone())
            .render_pass(render_pass.clone().first_subpass())
            .viewport_state(vk::ViewportState::viewport_dynamic_scissor_irrelevant())
            .build(render_surface.context().device().clone())?;


        let test_image_binding = vk::PersistentDescriptorSet::new(
            render_surface.context().allocators().descriptor_set(),
            pipeline.layout().set_layouts()[0].clone(),
            [
                vk::WriteDescriptorSet::image_view_sampler(0, test_image.clone(), sampler)
            ]
        )?;
        let margin = 25.0;

        //Total size, to "fit" image. Use the smallest of both dimensions.
        let image_size_px = size[0].min(size[1]) as f32 - (2.0 * margin);
        let x = (size[0] as f32 - image_size_px) / 2.0;
        let y = (size[1] as f32 - image_size_px) / 2.0;
        let xform = 
            cgmath::ortho(0.0, size[0] as f32, size[1] as f32, 0.0, -1.0, 1.0) *
            Matrix4::from_translation(cgmath::Vector3 { x, y, z: 0.0 }) *
            Matrix4::from_scale(image_size_px as f32);

        let mut s = Self {
            render_context: render_surface.context().clone(),
            framebuffers: Vec::new(),
            prerecorded_command_buffers: Vec::new(),
            index_buffer: index_staging_buf,
            vertex_buffer: vertex_staging_buf,
            pipeline,
            transparency_pipeline,
            render_pass,
            transform_matrix: xform.into(),
            viewport_dimensions: size,
            test_image_binding,
        };
        s.surface_changed(
            render_surface
        );

        Ok(s)
    }
    fn recalc_matrix(&mut self) {
        let size = self.viewport_dimensions;
        let margin = 25.0;
        //Total size, to "fit" image. Use the smallest of both dimensions.
        let image_size_px = size[0].min(size[1]) as f32 - (2.0 * margin);
        let x = (size[0] as f32 - image_size_px) / 2.0;
        let y = (size[1] as f32 - image_size_px) / 2.0;
        let xform = 
            cgmath::ortho(0.0, size[0] as f32, size[1] as f32, 0.0, -1.0, 1.0) *
            Matrix4::from_translation(cgmath::Vector3 { x, y, z: 0.0 }) *
            Matrix4::from_scale(image_size_px as f32);
        
        self.transform_matrix = xform.into();
    } 
    fn record_commandbuffers(&mut self) {
        //Drop old buffers-- RESOURCES MIGHT STILL BE IN USE ON ANOTHER THREAD!
        self.prerecorded_command_buffers = Vec::new();

        if self.framebuffers.is_empty() {
            log::error!("Cannot record commandbuffers with no framebuffers");
        }
        let command_buffers : AnyResult<Vec<_>> = self.framebuffers.iter()
            .map(|framebuffer| -> AnyResult<vk::PrimaryAutoCommandBuffer> {
                let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                    self.render_context.allocators().command_buffer(),
                    self.render_context.queues().graphics().idx(),
                    command_buffer::CommandBufferUsage::MultipleSubmit,
                )?;

                command_buffer
                    .begin_render_pass(
                        vk::RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.2, 0.2, 0.2, 1.0].into())
                            ],
                            ..vk::RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        command_buffer::SubpassContents::Inline
                    )?
                    .bind_pipeline_graphics(
                        self.transparency_pipeline.clone()
                    )
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .bind_index_buffer(self.index_buffer.clone())
                    .set_viewport(0,
                        [
                            vk::Viewport {
                                depth_range: 0.0..1.0,
                                dimensions: [self.viewport_dimensions[0] as f32, self.viewport_dimensions[1] as f32],
                                origin: [0.0; 2]
                            }
                        ]
                    )
                    .push_constants(
                        self.transparency_pipeline.layout().clone(),
                        0,
                        document_preview_shaders::vertex::Matrix{
                            mat: self.transform_matrix
                        }
                    )
                    .draw_indexed(6, 1, 0, 0, 0)?
                    .bind_pipeline_graphics(self.pipeline.clone())
                    .bind_descriptor_sets(
                        vulkano::pipeline::PipelineBindPoint::Graphics,
                        self.pipeline.layout().clone(),
                        0,
                        vec![self.test_image_binding.clone()]
                    )
                    .draw_indexed(6, 1, 0, 0, 0)?
                    .end_render_pass()?;


                Ok(command_buffer.build()?)
            })
            .collect();
        
        if let Ok(buffers) = command_buffers {
            self.prerecorded_command_buffers = buffers.into_iter()
                .map(Arc::new)
                .collect();
        } else {
            log::error!("Failed to record preview command buffers");
        }
    }
}
impl PreviewRenderProxy for DocumentViewportPreviewProxy {
    fn render<'a>(&'a mut self, idx: u32)
            -> AnyResult<Arc<vk::PrimaryAutoCommandBuffer>> {
        let Some(buffer) = self.prerecorded_command_buffers.get(idx as usize)
        else {
            anyhow::bail!("No buffer found for this frame!")
        };
        
        Ok(buffer.clone())
    }
    fn render_complete(&mut self, idx : u32) {
        
    }
    fn surface_changed(&mut self, render_surface: &render_device::RenderSurface) {

        if render_surface.context().device() != self.pipeline.device() {
            panic!("Wrong device used to recreate preview proxy!")
        }

        self.framebuffers = Vec::new();
        let framebuffers : AnyResult<Vec<_>> = render_surface.swapchain_images().iter()
            .map(|image| -> AnyResult<_> {
                // Todo: duplication of view resources.
                let view = vk::ImageView::new_default(image.clone())?;

                let framebuffer = vk::Framebuffer::new(
                    self.render_pass.clone(),
                    vk::FramebufferCreateInfo{
                        attachments: vec![view],
                        ..Default::default()
                    }
                );

                Ok(framebuffer?)
            })
            .collect();

        self.framebuffers = framebuffers.expect("Failed to create proxy framebuffers.");

        self.viewport_dimensions = render_surface.extent();
        self.recalc_matrix();
        //Todo: rebuild pipeline with new format/size, if changed.
        self.record_commandbuffers();
    }
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    env_logger::init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    //let (image, future) = make_test_image(render_context.clone())?;
    let (image, future) = load_document_image(render_context.clone(), &std::path::PathBuf::from("./test-data/sit.png"))?;

    future.wait(None)?;

    let document_view = Box::new(DocumentViewportPreviewProxy::new(&render_surface, image)?);
    let window_renderer = window_surface.with_render_surface(render_surface, render_context.clone(), document_view)?;
    println!("Made render context!");

    window_renderer.run();
}

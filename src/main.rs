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
pub mod DocumentViewportProxy;

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

fn make_test_image(render_context: Arc<render_device::RenderContext>) -> AnyResult<(Arc<vk::StorageImage>, vk::sync::future::FenceSignalFuture<impl vk::sync::GpuFuture>)> {
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
        .then_signal_fence_and_flush()?;

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
pub trait PreviewRenderProxy {
    /// Create the render commands for this frame. Assume used resources are borrowed until a matching "render_complete" for this
    /// frame idx is called.
    fn render(&mut self, swapchain_image_idx: u32)
        -> AnyResult<Arc<vk::PrimaryAutoCommandBuffer>>;

    /// When the future of a previous render has completed
    fn render_complete(&mut self, idx : u32);
    fn surface_changed(&mut self, render_surface: &render_device::RenderSurface);
}

struct DocumentPreviewRenderer {
    /// The composited document
    document_image: Arc<vk::StorageImage>,

    /// The preview of whatever the user is doing right now - 
    /// Scratch space for in-progress strokes, brush preview, ect.
    live_image: Arc<vk::StorageImage>,


}

struct SillyDocument {

}

fn listener(mut event_stream: tokio::sync::broadcast::Receiver<stylus_events::StylusEventFrame>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    loop {
        match runtime.block_on(event_stream.recv()) {
            Ok(event_frame) => {
                for event in event_frame.iter() {
                    //log::trace!("{event:?}")
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(num)) => {
                log::warn!("Lost {num} stylus frames!");
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                return 
            }
        }
    }
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    env_logger::builder().filter_level(log::LevelFilter::max()).init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    // Test image generators.
    //let (image, future) = make_test_image(render_context.clone())?;
    //let (image, future) = load_document_image(render_context.clone(), &std::path::PathBuf::from("/home/aspen/Pictures/thesharc.png"))?;
    //future.wait(None)?;

    let document_view = Box::new(DocumentViewportProxy::DocumentViewportPreviewProxy::new(&render_surface)?);
    let window_renderer = window_surface.with_render_surface(render_surface, render_context.clone(), document_view)?;

    let event_stream = window_renderer.stylus_events();

    std::thread::spawn(move || listener(event_stream));

    window_renderer.run();
}

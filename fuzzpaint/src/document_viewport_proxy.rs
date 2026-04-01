use crate::vulkano_prelude::*;
use std::sync::Arc;

use crate::{AnyResult, render_device, view_transform::ViewInfo};

mod shaders {
    pub mod vertex {
        vulkano_shaders::shader! {
            ty: "vertex",
            src:r"
            #version 460
            
            layout(push_constant) uniform Matrix {
                mat4 mat;
            } matrix;

            layout(location = 0) out vec2 out_uv;

            void main() {
                vec4 pos = vec4(
                    float(gl_VertexIndex & 1),
                    float((gl_VertexIndex & 2) / 2),
                    0.0,
                    1.0
                );
                out_uv = vec2(pos.x, 1.0 - pos.y);
                gl_Position = matrix.mat * pos;
            }"
        }
    }
    pub mod fragment {
        vulkano_shaders::shader! {
            ty: "fragment",
            src:r"
            #version 460

            const float LIGHT = 0.8;
            const float DARK = 0.7;
            const uint SIZE = uint(16);

            layout(set = 0, binding = 0) uniform sampler2D image;

            layout(location = 0) in vec2 uv;

            layout(location = 0) out vec4 color;

            void main() {
                uvec2 grid_coords = uvec2(gl_FragCoord.xy) / SIZE;
                bool is_light = (grid_coords.x + grid_coords.y) % 2 == 0;
                vec3 grid_color = 1.0 - vec3(vec3(is_light ? LIGHT : DARK));

                vec4 col = texture(image, uv);
                // col is pre-multiplied, grid color is not. Combine!
                color = vec4(grid_color * (1.0 - col.a) + col.rgb, 1.0);
            }"
        }
    }
}

/// An acquired image from the proxy. Will become the current image when dropped,
/// or after a user-provided GPU fence.
pub struct ImageGuard<'proxy> {
    proxy: &'proxy Proxy,
    image: Arc<vk::ImageView>,
    is_submitted: bool,
}
impl ImageGuard<'_> {
    /// Submit this image for display immediately. The image should be done writing by the device, as it
    /// will be used for reading without synchronizing!
    pub fn submit_now(self) {
        // mwehehehehe. sneaky delegate to Drop::drop >:3
    }
    /// Submit this image for display, after the given fence finishes. The image should be done writing
    /// at the time of the fence, as it will be used for reading as soon as the fence is signalled.
    pub fn submit_with_fence(
        mut self,
        fence: vk::sync::future::FenceSignalFuture<Box<dyn GpuFuture + Send>>,
    ) {
        // Surpress default drop behavior.
        self.is_submitted = true;

        // Place this fence within the proxy.
        let mut write = self.proxy.swap_after.write();
        // This should not be possible - if this proxy exists, there should have been no
        // outstanding writes waiting.
        assert!(write.is_empty());
        *write = SwapAfter::Fence(fence);
    }
}
impl Drop for ImageGuard<'_> {
    fn drop(&mut self) {
        if self.is_submitted {
            return;
        }
        self.is_submitted = true;
        // Place an immediate swap into the proxy.
        let mut write = self.proxy.swap_after.write();
        // This should not be possible - if this proxy exists, there should have been no
        // outstanding writes waiting.
        assert!(write.is_empty());
        *write = SwapAfter::Now;
    }
}
impl std::ops::Deref for ImageGuard<'_> {
    type Target = Arc<vk::ImageView>;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

enum SwapAfter<Future: GpuFuture> {
    Fence(vk::sync::future::FenceSignalFuture<Future>),
    Now,
    Empty,
}
impl<Future: GpuFuture> SwapAfter<Future> {
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

/// Collection of all the data that is derived from the surface.
/// Everything else is """immutable""", whereas this all needs to be mutable.
/// When the surface changes, a new one is made and a quick Arc pointer swap is all that is needed.
///
/// The dynamic transforms very much spoiled the purpose of this struct, evaluate if it can be re-merged into
/// the main struct.
struct SurfaceData {
    context: Arc<crate::render_device::RenderContext>,
    pipeline: Arc<vk::GraphicsPipeline>,
    framebuffers: Box<[Arc<vk::Framebuffer>]>,
    document_image_bindings: [Arc<vk::PersistentDescriptorSet>; 2],
    // Lazily recorded command buffers. Must be rebuilt on viewport size/document view change.
    // indexed by swapchain idx, then by image idx
    prerecorded_command_buffers: Vec<[std::sync::OnceLock<Arc<vk::PrimaryAutoCommandBuffer>>; 2]>,
    transform: ViewInfo,
    surface_dimensions: [u32; 2],
}
impl SurfaceData {
    fn new(
        context: Arc<render_device::RenderContext>,
        render_surface: &render_device::RenderSurface,
        render_pass: Arc<vk::RenderPass>,
        pipeline: Arc<vk::GraphicsPipeline>,
        document_image_bindings: &[Arc<vk::PersistentDescriptorSet>; 2],

        document_transform: ViewInfo,
    ) -> Self {
        let framebuffers: AnyResult<Vec<_>> = render_surface
            .swapchain_images()
            .unwrap()
            .iter()
            .map(|image| -> AnyResult<_> {
                // Todo: duplication of view resources.
                let view = vk::ImageView::new_default(image.clone())?;

                let framebuffer = vk::Framebuffer::new(
                    render_pass.clone(),
                    vk::FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                );

                Ok(framebuffer?)
            })
            .collect();
        let framebuffers = framebuffers.unwrap().into_boxed_slice();

        let mut prerecorded_command_buffers =
            Vec::with_capacity(render_surface.swapchain_images().unwrap().len());
        prerecorded_command_buffers.resize_with(prerecorded_command_buffers.capacity(), || {
            [std::sync::OnceLock::new(), std::sync::OnceLock::new()]
        });

        Self {
            context,
            pipeline,
            surface_dimensions: render_surface.extent(),

            prerecorded_command_buffers,

            framebuffers,
            document_image_bindings: [
                document_image_bindings[0].clone(),
                document_image_bindings[1].clone(),
            ],

            transform: document_transform,
        }
    }
    fn get_commands(
        &self,
        swapchain_idx: u32,
        image_idx: usize,
    ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        // Try to fetch from the cache:
        let cached = self
            .prerecorded_command_buffers
            .get(swapchain_idx as usize)
            .and_then(|bufs| bufs.get(image_idx))
            .and_then(|lock| lock.get());
        if let Some(cached) = cached {
            return Ok(cached.clone());
        }
        // Didn't find a cached ver. build it:
        let framebuffer = self
            .framebuffers
            .get(swapchain_idx as usize)
            .ok_or_else(|| anyhow::anyhow!("Swapchain idx out of bounds"))?;
        let image_binding = self
            .document_image_bindings
            .get(image_idx)
            .ok_or_else(|| anyhow::anyhow!("Image idx out of bounds"))?;

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().graphics().idx(),
            vk::CommandBufferUsage::MultipleSubmit,
        )?;

        let matrix: [[f32; 4]; 4] = {
            let transform = self.transform.into_pixel_perfect_similarity();

            // Scale the 1x1 mesh up to the size of the document
            let base_xform = ultraviolet::Mat4::from_nonuniform_scale(ultraviolet::Vec3 {
                x: crate::DOCUMENT_DIMENSION as f32,
                y: crate::DOCUMENT_DIMENSION as f32,
                z: 1.0,
            });
            // Then position the document in screenspace according to the view transform
            let mat4 = transform.into_mat4();

            // Then project screenspace down to clipspace.
            let proj = crate::vk::projection::orthographic_vk(
                0.0,
                self.surface_dimensions[0] as f32,
                0.0,
                self.surface_dimensions[1] as f32,
                -1.0,
                1.0,
            );
            let proj = proj * mat4 * base_xform;
            proj.into()
        };
        command_buffer
            .begin_render_pass(
                vk::RenderPassBeginInfo {
                    clear_values: vec![Some([0.05, 0.05, 0.05, 1.0].into())],
                    ..vk::RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                vk::SubpassBeginInfo {
                    contents: vk::SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                vk::PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                vec![image_binding.clone()],
            )?
            .set_viewport(
                0,
                smallvec::smallvec![vk::Viewport {
                    depth_range: 0.0..=1.0,
                    extent: [
                        self.surface_dimensions[0] as f32,
                        self.surface_dimensions[1] as f32,
                    ],
                    offset: [0.0; 2],
                }],
            )?
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                shaders::vertex::Matrix { mat: matrix },
            )?
            .draw(4, 1, 0, 0)?
            .end_render_pass(vk::SubpassEndInfo::default())?;

        let command_buffer = command_buffer.build()?;

        // Try to insert into the cache
        if let Some(lock) = self
            .prerecorded_command_buffers
            .get(swapchain_idx as usize)
            .and_then(|bufs| bufs.get(image_idx))
        {
            let _ = lock.set(command_buffer.clone());
        }

        Ok(command_buffer)
    }
    fn clear_cache(&mut self) {
        // Take and discard all cached command buffers
        for [a, b] in &mut self.prerecorded_command_buffers {
            a.take();
            b.take();
        }
    }
    fn set_transform(&mut self, transform: ViewInfo) {
        self.transform = transform;
        self.clear_cache();
    }
}

/// An double-buffering interface between the asynchronous edit->render pipeline of documents
/// and the synchronous redrawing of the many swapchain images.
/// (Because dealing with one image is easier than potentially many, as we don't care about excess framerate)
/// Provides a method to get a drawable buffer asynchronously, and handles drawing that to the screen
/// whenever needed by the swapchain.
pub struct Proxy {
    render_context: Arc<render_device::RenderContext>,

    document_transform: tokio::sync::RwLock<ViewInfo>,

    // Double buffer data =========
    document_images: [Arc<vk::ImageView>; 2],
    document_image_bindings: [Arc<vk::PersistentDescriptorSet>; 2],

    // Sync + Swap data ===========
    /// After this fence is completed, a swap occurs.
    /// If this is not none, it implies both buffers are in use.
    swap_after: parking_lot::RwLock<SwapAfter<Box<dyn GpuFuture + Send>>>,
    /// A buffer is available for writing if this notify is set.
    write_ready_notify: tokio::sync::Notify,
    /// Which buffer is the swapchain reading from?
    read_buf: std::sync::atomic::AtomicU8,

    // Static render data ============
    render_pass: Arc<vk::RenderPass>,
    pipeline: Arc<vk::GraphicsPipeline>,

    // Surface-derived render data ===============
    surface_data: tokio::sync::RwLock<SurfaceData>,
}

impl Proxy {
    pub fn new(render_surface: &render_device::RenderSurface, view: ViewInfo) -> AnyResult<Self> {
        // Only one frame-in-flight - Keep an additional buffer for writing to.
        const NUM_DOCUMENT_BUFFERS: u32 = 2;

        let document_image_array = vk::Image::new(
            render_surface.context().allocators().memory().clone(),
            vk::ImageCreateInfo {
                image_type: vk::ImageType::Dim2d,
                format: crate::DOCUMENT_FORMAT,
                extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                array_layers: NUM_DOCUMENT_BUFFERS,
                // Too many!!
                usage: vk::ImageUsage::COLOR_ATTACHMENT
                    | vk::ImageUsage::INPUT_ATTACHMENT
                    | vk::ImageUsage::SAMPLED
                    | vk::ImageUsage::TRANSFER_DST
                    | vk::ImageUsage::STORAGE,
                sharing: vk::Sharing::Exclusive,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let initialize_future = {
            let context = render_surface.context();

            let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                context.allocators().command_buffer(),
                context.queues().compute().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;

            command_buffer.clear_color_image(vk::ClearColorImageInfo {
                image_layout: vk::ImageLayout::General,
                clear_value: [0.0; 4].into(),
                regions: smallvec::smallvec![vk::ImageSubresourceRange {
                    array_layers: 0..1,
                    aspects: vk::ImageAspects::COLOR,
                    mip_levels: 0..1,
                },],
                ..vk::ClearColorImageInfo::image(document_image_array.clone())
            })?;

            let command_buffer = command_buffer.build()?;

            context
                .now()
                .then_execute(context.queues().compute().queue().clone(), command_buffer)?
                .then_signal_fence_and_flush()?
        };
        // Wait on the future at the end of init
        let _defer = defer::defer(move || initialize_future.wait(None).unwrap());

        let document_image_views = [
            vk::ImageView::new(
                document_image_array.clone(),
                vk::ImageViewCreateInfo {
                    subresource_range: vk::ImageSubresourceRange {
                        array_layers: 0..1,
                        aspects: vk::ImageAspects::COLOR,
                        mip_levels: 0..1,
                    },
                    view_type: vk::ImageViewType::Dim2d,
                    ..vk::ImageViewCreateInfo::from_image(&document_image_array)
                },
            )?,
            vk::ImageView::new(
                document_image_array.clone(),
                vk::ImageViewCreateInfo {
                    subresource_range: vk::ImageSubresourceRange {
                        array_layers: 1..2,
                        aspects: vk::ImageAspects::COLOR,
                        mip_levels: 0..1,
                    },
                    view_type: vk::ImageViewType::Dim2d,
                    ..vk::ImageViewCreateInfo::from_image(&document_image_array)
                },
            )?,
        ];

        let render_pass = vulkano::single_pass_renderpass!(
            render_surface.context().device().clone(),
            attachments: {
                document: {
                    format: render_surface.format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [document],
                depth_stencil: {},
            },
        )?;

        let sampler = vk::Sampler::new(
            render_surface.context().device().clone(),
            vk::SamplerCreateInfo {
                min_filter: vk::Filter::Linear,
                mag_filter: vk::Filter::Nearest,
                ..Default::default()
            },
        )?;

        let vertex_shader = shaders::vertex::load(render_surface.context().device().clone())?;
        let fragment_shader = shaders::fragment::load(render_surface.context().device().clone())?;

        // "main" is the only valid GLSL entry point name, ok to unwrap.
        let vertex_shader = vertex_shader.entry_point("main").unwrap();
        let fragment_shader = fragment_shader.entry_point("main").unwrap();

        let vertex_stage = vk::PipelineShaderStageCreateInfo::new(vertex_shader);
        let fragment_stage = vk::PipelineShaderStageCreateInfo::new(fragment_shader);

        let no_blend = vk::ColorBlendState::with_attachment_states(
            1,
            vk::ColorBlendAttachmentState::default(),
        );

        let matrix_push_constant = vk::PushConstantRange {
            offset: 0,
            stages: vk::ShaderStages::VERTEX,
            size: std::mem::size_of::<shaders::vertex::Matrix>() as u32,
        };
        let layout = vk::PipelineLayout::new(
            render_surface.context().device().clone(),
            vk::PipelineLayoutCreateInfo {
                set_layouts: vec![vk::DescriptorSetLayout::new(
                    render_surface.context().device().clone(),
                    vk::DescriptorSetLayoutCreateInfo {
                        bindings: [(
                            0,
                            vk::DescriptorSetLayoutBinding {
                                descriptor_count: 1,
                                stages: vk::ShaderStages::FRAGMENT,
                                ..vk::DescriptorSetLayoutBinding::descriptor_type(
                                    vk::DescriptorType::CombinedImageSampler,
                                )
                            },
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )?],
                push_constant_ranges: vec![matrix_push_constant],
                ..Default::default()
            },
        )?;
        let pipeline = vk::GraphicsPipeline::new(
            render_surface.context().device().clone(),
            None,
            vk::GraphicsPipelineCreateInfo {
                color_blend_state: Some(no_blend),
                input_assembly_state: Some(vk::InputAssemblyState {
                    topology: vk::PrimitiveTopology::TriangleStrip,
                    primitive_restart_enable: false,
                    ..Default::default()
                }),
                multisample_state: Some(vk::MultisampleState::default()),
                rasterization_state: Some(vk::RasterizationState {
                    cull_mode: vk::CullMode::None,
                    ..Default::default()
                }),
                vertex_input_state: Some(
                    // Vertices are generated in-shader from VertexIndex
                    vk::VertexInputState::new(),
                ),
                // One viewport and scissor, scissor irrelevant and viewport dynamic
                viewport_state: Some(vk::ViewportState::default()),
                dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                subpass: Some(render_pass.clone().first_subpass().into()),
                stages: smallvec::smallvec![vertex_stage, fragment_stage],
                ..vk::GraphicsPipelineCreateInfo::layout(layout.clone())
            },
        )?;
        let document_image_bindings = [
            vk::PersistentDescriptorSet::new(
                render_surface.context().allocators().descriptor_set(),
                layout.set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0,
                    document_image_views[0].clone(),
                    sampler.clone(),
                )],
                [],
            )?,
            vk::PersistentDescriptorSet::new(
                render_surface.context().allocators().descriptor_set(),
                layout.set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0,
                    document_image_views[1].clone(),
                    sampler,
                )],
                [],
            )?,
        ];

        let surface_data = SurfaceData::new(
            render_surface.context().clone(),
            render_surface,
            render_pass.clone(),
            pipeline.clone(),
            &document_image_bindings,
            view,
        );

        let notify = tokio::sync::Notify::new();
        // Start as notified - write buffer is available immediately.
        notify.notify_one();

        Ok(Self {
            render_context: render_surface.context().clone(),

            document_transform: view.into(),

            pipeline,
            render_pass,

            swap_after: SwapAfter::Empty.into(),
            read_buf: 0.into(),
            write_ready_notify: notify,

            document_images: document_image_views,
            document_image_bindings,

            surface_data: surface_data.into(),
        })
    }
    /// Internal use only. After the user's buffer is deemed swappable, the read index in switched over and returned.
    /// Furthermore, the old read buffer is signalled as being writable to any waiting users. New read idx is returned.
    fn swap(&self) -> usize {
        // Unsure of the proper ordering here. It's not the hottest path, so the strictest one should be okeyyyy.
        let idx = self
            .read_buf
            .fetch_xor(1, std::sync::atomic::Ordering::SeqCst) as usize;
        self.write_ready_notify.notify_one();
        idx
    }
    /// Read the proxy - returns the index of the current read buffer. Internally swaps if a render is complete.
    /// # Safety
    ///
    /// A call to `read` implies any use of previously read image is complete. This must be synchronized externally!!
    pub unsafe fn read(&self) -> usize {
        let mut lock = self.swap_after.write();
        match &*lock {
            // Nothin to do
            SwapAfter::Empty => self.read_buf.load(std::sync::atomic::Ordering::SeqCst) as usize,
            // Immediate swap
            SwapAfter::Now => {
                *lock = SwapAfter::Empty;
                self.swap()
            }
            // Swap if the fence is signalled - without waiting. If not, do nothing.
            SwapAfter::Fence(fence) => {
                if fence.is_signaled().unwrap() {
                    *lock = SwapAfter::Empty;
                    self.swap()
                } else {
                    self.read_buf.load(std::sync::atomic::Ordering::SeqCst) as usize
                }
            }
        }
    }
    /// Returns true if a new image was submitted that hasn't been
    /// acquired yet
    pub fn redraw_requested(&self) -> bool {
        match &*self.swap_after.read() {
            SwapAfter::Empty => false,
            SwapAfter::Now => true,
            SwapAfter::Fence(fence) => fence.is_signaled().unwrap(),
        }
    }
    pub async fn write(&self) -> ImageGuard<'_> {
        self.write_ready_notify.notified().await;
        assert!(self.swap_after.read().is_empty());
        // We are now the sole writer. Hopefully. Return the proxy:
        ImageGuard {
            // Return whichever image is *not* the read buf. Uhm uh ordering??
            image: self.document_images
                [(self.read_buf.load(std::sync::atomic::Ordering::SeqCst) ^ 1) as usize]
                .clone(),
            is_submitted: false,
            proxy: self,
        }
    }
    pub fn insert_document_transform(&self, new: ViewInfo) {
        *self.document_transform.blocking_write() = new;
        let mut surface = self.surface_data.blocking_write();
        surface.transform = new;
        surface.clear_cache();
    }
    pub fn get_view_transform(&self) -> ViewInfo {
        *self.document_transform.blocking_read()
    }
    pub async fn get_view_transform_async(&self) -> ViewInfo {
        *self.document_transform.read().await
    }
}
impl Proxy {
    #[deny(unsafe_op_in_unsafe_fn)]
    pub unsafe fn render(
        &self,
        swapchain_image: Arc<vk::Image>,
        swapchain_idx: u32,
    ) -> AnyResult<Arc<vk::PrimaryAutoCommandBuffer>> {
        // Safety: contract forwarded to the contract of this fn.
        let image_idx = unsafe { self.read() };
        let read = self.surface_data.blocking_read();
        read.get_commands(swapchain_idx, image_idx)
    }
    pub fn surface_changed(&self, render_surface: &render_device::RenderSurface) {
        let transform = *self.document_transform.blocking_read();

        let new = SurfaceData::new(
            self.render_context.clone(),
            render_surface,
            self.render_pass.clone(),
            self.pipeline.clone(),
            &self.document_image_bindings,
            transform,
        );
        *self.surface_data.blocking_write() = new;
    }
    pub fn has_update(&self) -> bool {
        self.redraw_requested()
    }
}

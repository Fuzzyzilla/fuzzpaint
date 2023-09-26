use std::sync::Arc;

use crate::{gizmos::Gizmooooo, *};

/// Proxy called into by the window renderer to perform the necessary synchronization and such to render the screen
/// behind the Egui content.
pub trait PreviewRenderProxy {
    /// Create the render commands for this frame. Assume used resources are borrowed until a matching "render_complete" for this
    /// frame idx is called.
    /// # Safety
    ///
    /// the previous render should be finished before the return result is executed.
    unsafe fn render(
        &self,
        swapchain_image: Arc<vk::SwapchainImage>,
        swapchain_image_idx: u32,
    ) -> AnyResult<smallvec::SmallVec<[Arc<vk::PrimaryAutoCommandBuffer>; 2]>>;
    /// The window surface has been invalidated and remade.
    fn surface_changed(&self, render_surface: &render_device::RenderSurface);
    /// Is this proxy requesting a redraw?
    fn has_update(&self) -> bool;
    /// The area used for this viewport has changed. Not the same as the surface - rather, the central area
    /// between UI elements where this proxy is visible. Proxies should still initialize the whole screen, however.
    fn viewport_changed(&self, position: ultraviolet::Vec2, size: ultraviolet::Vec2);

    /// The cursor requested by the preview, or None for default.
    fn cursor(&self) -> Option<crate::gizmos::CursorOrInvisible>;
}

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
pub struct DocumentViewportPreviewProxyImageGuard<'proxy> {
    proxy: &'proxy DocumentViewportPreviewProxy,
    image: Arc<vk::ImageView<vk::StorageImage>>,
    is_submitted: bool,
}
impl DocumentViewportPreviewProxyImageGuard<'_> {
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
        let mut write = self.proxy.swap_after.write().unwrap();
        // This should not be possible - if this proxy exists, there should have been no
        // outstanding writes waiting.
        assert!(write.is_empty());
        *write = SwapAfter::Fence(fence);
    }
}
impl Drop for DocumentViewportPreviewProxyImageGuard<'_> {
    fn drop(&mut self) {
        if self.is_submitted {
            return;
        }
        self.is_submitted = true;
        // Place an immediate swap into the proxy.
        let mut write = self.proxy.swap_after.write().unwrap();
        // This should not be possible - if this proxy exists, there should have been no
        // outstanding writes waiting.
        assert!(write.is_empty());
        *write = SwapAfter::Now;
    }
}
impl std::ops::Deref for DocumentViewportPreviewProxyImageGuard<'_> {
    type Target = Arc<vk::ImageView<vk::StorageImage>>;
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
        match self {
            Self::Empty => true,
            _ => false,
        }
    }
}

/// Collection of all the data that is derived from the surface.
/// Everything else is """immutable""", whereas this all needs to be mutable.
/// When the surface changes, a new one is made and a quick Arc pointer swap is all that is needed.
///
/// The dynamic transforms very much spoiled the purpose of this struct, evaluate if it can be re-merged into
/// the main struct.
struct ProxySurfaceData {
    context: Arc<crate::render_device::RenderContext>,
    render_pass: Arc<vk::RenderPass>,
    pipeline: Arc<vk::GraphicsPipeline>,
    framebuffers: Box<[Arc<vk::Framebuffer>]>,
    document_image_bindings: [Arc<vk::PersistentDescriptorSet>; 2],
    // Lazily recorded command buffers. Must be rebuilt on viewport size/document view change.
    // indexed by swapchain idx, then by image idx
    prerecorded_command_buffers: Vec<[std::sync::OnceLock<Arc<vk::PrimaryAutoCommandBuffer>>; 2]>,
    cached_matrix: std::sync::OnceLock<[[f32; 4]; 4]>,
    transform: crate::view_transform::DocumentTransform,
    view_pos: cgmath::Point2<f32>,
    view_size: cgmath::Vector2<f32>,
    surface_dimensions: [u32; 2],
}
impl ProxySurfaceData {
    fn new(
        context: Arc<render_device::RenderContext>,
        render_surface: &render_device::RenderSurface,
        render_pass: Arc<vk::RenderPass>,
        pipeline: Arc<vk::GraphicsPipeline>,
        document_image_bindings: &[Arc<vk::PersistentDescriptorSet>; 2],

        viewport_pos: cgmath::Point2<f32>,
        viewport_size: cgmath::Vector2<f32>,
        document_transform: crate::view_transform::DocumentTransform,
    ) -> Self {
        let framebuffers: AnyResult<Vec<_>> = render_surface
            .swapchain_images()
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
            Vec::with_capacity(render_surface.swapchain_images().len());
        prerecorded_command_buffers.resize_with(prerecorded_command_buffers.capacity(), || {
            [std::sync::OnceLock::new(), std::sync::OnceLock::new()]
        });

        Self {
            context,
            pipeline,
            render_pass,
            surface_dimensions: render_surface.extent(),

            prerecorded_command_buffers,

            framebuffers,
            document_image_bindings: [
                document_image_bindings[0].clone(),
                document_image_bindings[1].clone(),
            ],

            transform: document_transform,
            view_pos: viewport_pos,
            view_size: viewport_size,
            cached_matrix: Default::default(),
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
            command_buffer::CommandBufferUsage::MultipleSubmit,
        )?;

        let matrix = self
            .cached_matrix
            .get_or_try_init(|| -> anyhow::Result<_> {
                let transform = match &self.transform {
                    view_transform::DocumentTransform::Fit(f) => f
                        .make_transform(
                            cgmath::vec2(
                                crate::DOCUMENT_DIMENSION as f32,
                                crate::DOCUMENT_DIMENSION as f32,
                            ),
                            self.view_pos,
                            self.view_size,
                        )
                        .ok_or_else(|| anyhow::anyhow!("Malformed document transform"))?,
                    view_transform::DocumentTransform::Transform(t) => t.clone(),
                };

                let base_xform = ultraviolet::Mat4::from_nonuniform_scale(ultraviolet::Vec3 {
                    x: crate::DOCUMENT_DIMENSION as f32,
                    y: crate::DOCUMENT_DIMENSION as f32,
                    z: 1.0,
                });
                // convert cgmath to ultraviolet (todo, switch all to ultraviolet)
                let mat4: cgmath::Matrix4<f32> = transform.into();
                let mat4: [[f32; 4]; 4] = mat4.into();
                let mat4: ultraviolet::Mat4 = mat4.into();

                let proj = crate::vk::projection::orthographic_vk(
                    0.0,
                    self.surface_dimensions[0] as f32,
                    0.0,
                    self.surface_dimensions[1] as f32,
                    -1.0,
                    1.0,
                );
                let proj = proj * mat4 * base_xform;
                let transform_matrix: [[f32; 4]; 4] = proj.into();
                Ok(transform_matrix)
            })?;
        command_buffer
            .begin_render_pass(
                vk::RenderPassBeginInfo {
                    clear_values: vec![Some([0.05, 0.05, 0.05, 1.0].into())],
                    ..vk::RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                command_buffer::SubpassContents::Inline,
            )?
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                vec![image_binding.clone()],
            )
            .set_viewport(
                0,
                [vk::Viewport {
                    depth_range: 0.0..1.0,
                    dimensions: [
                        self.surface_dimensions[0] as f32,
                        self.surface_dimensions[1] as f32,
                    ],
                    origin: [0.0; 2],
                }],
            )
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                shaders::vertex::Matrix { mat: *matrix },
            )
            .draw(4, 1, 0, 0)?
            .end_render_pass()?;

        let command_buffer = Arc::new(command_buffer.build()?);

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
        for [a, b] in self.prerecorded_command_buffers.iter_mut() {
            a.take();
            b.take();
        }
        self.cached_matrix.take();
    }
    fn set_transform(&mut self, transform: crate::view_transform::DocumentTransform) {
        self.transform = transform;
        self.clear_cache();
    }
    fn set_viewport_size(&mut self, pos: cgmath::Point2<f32>, size: cgmath::Vector2<f32>) {
        self.view_pos = pos;
        self.view_size = size;
        // Only the fit transform needs to be recalc'd on viewport resize.
        if let crate::view_transform::DocumentTransform::Fit(..) = self.transform {
            self.clear_cache()
        }
    }
}

/// An double-buffering interface between the asynchronous edit->render pipeline of documents
/// and the synchronous redrawing of the many swapchain images.
/// (Because dealing with one image is easier than potentially many, as we don't care about excess framerate)
/// Provides a method to get a drawable buffer asynchronously, and handles drawing that to the screen
/// whenever needed by the swapchain.
pub struct DocumentViewportPreviewProxy {
    render_context: Arc<render_device::RenderContext>,

    document_transform: tokio::sync::RwLock<crate::view_transform::DocumentTransform>,
    viewport: parking_lot::RwLock<(cgmath::Point2<f32>, cgmath::Vector2<f32>)>,

    // Double buffer data =========
    document_images: [Arc<vk::ImageView<vk::StorageImage>>; 2],
    document_image_bindings: [Arc<vk::PersistentDescriptorSet>; 2],

    // Sync + Swap data ===========
    /// After this fence is completed, a swap occurs.
    /// If this is not none, it implies both buffers are in use.
    swap_after: std::sync::RwLock<SwapAfter<Box<dyn GpuFuture + Send>>>,
    /// A buffer is available for writing if this notify is set.
    write_ready_notify: tokio::sync::Notify,
    /// Which buffer is the swapchain reading from?
    read_buf: std::sync::atomic::AtomicU8,

    // Static render data ============
    render_pass: Arc<vk::RenderPass>,
    pipeline: Arc<vk::GraphicsPipeline>,
    gizmo_renderer: Arc<crate::gizmos::renderer::GizmoRenderer>,

    // Surface-derived render data ===============
    surface_data: tokio::sync::RwLock<ProxySurfaceData>,

    // User render data ============
    cursor: std::sync::RwLock<Option<crate::gizmos::CursorOrInvisible>>,
    tool_render_as: std::sync::RwLock<crate::pen_tools::RenderAs>,
}

impl DocumentViewportPreviewProxy {
    pub fn new(render_surface: &render_device::RenderSurface) -> AnyResult<Self> {
        // Only one frame-in-flight - Keep an additional buffer for writing to.
        let num_document_buffers = 2u32;

        let document_image_array = vk::StorageImage::with_usage(
            render_surface.context().allocators().memory(),
            vk::ImageDimensions::Dim2d {
                width: crate::DOCUMENT_DIMENSION,
                height: crate::DOCUMENT_DIMENSION,
                array_layers: num_document_buffers,
            },
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsage::COLOR_ATTACHMENT
                | vk::ImageUsage::SAMPLED
                | vk::ImageUsage::TRANSFER_DST
                | vk::ImageUsage::STORAGE,
            vk::ImageCreateFlags::empty(),
            [render_surface.context().queues().graphics().idx()],
        )?;

        // synchronously clear the read buffer, to move it into a well-defined format.
        // Not sure how to do this cleaner :V
        let initialize_future = {
            let context = render_surface.context();

            let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                context.allocators().command_buffer(),
                context.queues().compute().idx(),
                command_buffer::CommandBufferUsage::OneTimeSubmit,
            )?;

            command_buffer.clear_color_image(command_buffer::ClearColorImageInfo {
                image_layout: vulkano::image::ImageLayout::General,
                clear_value: [0.0; 4].into(),
                regions: smallvec::smallvec![vk::ImageSubresourceRange {
                    array_layers: 0..1,
                    aspects: vk::ImageAspects::COLOR,
                    mip_levels: 0..1,
                },],
                ..command_buffer::ClearColorImageInfo::image(document_image_array.clone())
            })?;

            let command_buffer = command_buffer.build()?;

            context
                .now()
                .then_execute(context.queues().compute().queue().clone(), command_buffer)?
                .then_signal_fence_and_flush()?
        };

        let document_image_views = [
            vk::ImageView::new(
                document_image_array.clone(),
                vk::ImageViewCreateInfo {
                    subresource_range: vk::ImageSubresourceRange {
                        array_layers: 0..1,
                        aspects: vk::ImageAspects::COLOR,
                        mip_levels: 0..1,
                    },
                    view_type: vulkano::image::view::ImageViewType::Dim2d,
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
                    view_type: vulkano::image::view::ImageViewType::Dim2d,
                    ..vk::ImageViewCreateInfo::from_image(&document_image_array)
                },
            )?,
        ];

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

        let mut no_blend = vk::ColorBlendState::new(1);
        no_blend.attachments[0].blend = None;

        let pipeline = vk::GraphicsPipeline::start()
            .vertex_shader(vertex_shader.clone(), ())
            .fragment_shader(fragment_shader, ())
            .vertex_input_state(vulkano::pipeline::graphics::vertex_input::VertexInputState::new())
            .rasterization_state(
                vk::RasterizationState::default()
                    .cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::None),
            )
            .input_assembly_state(
                vk::InputAssemblyState::new().topology(vk::PrimitiveTopology::TriangleStrip),
            )
            .color_blend_state(no_blend)
            .render_pass(render_pass.clone().first_subpass())
            .viewport_state(vk::ViewportState::viewport_dynamic_scissor_irrelevant())
            .build(render_surface.context().device().clone())?;

        let document_image_bindings = [
            vk::PersistentDescriptorSet::new(
                render_surface.context().allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0,
                    document_image_views[0].clone(),
                    sampler.clone(),
                )],
            )?,
            vk::PersistentDescriptorSet::new(
                render_surface.context().allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0,
                    document_image_views[1].clone(),
                    sampler,
                )],
            )?,
        ];

        let viewport_pos = [0.0, 0.0].into();
        let viewport_size = [
            render_surface.extent()[0] as f32,
            render_surface.extent()[1] as f32,
        ]
        .into();
        let document_transform = crate::view_transform::DocumentTransform::default();

        let surface_data = ProxySurfaceData::new(
            render_surface.context().clone(),
            render_surface,
            render_pass.clone(),
            pipeline.clone(),
            &document_image_bindings,
            viewport_pos,
            viewport_size,
            document_transform.clone(),
        );

        let notify = tokio::sync::Notify::new();
        // Start as notified - write buffer is available immediately.
        notify.notify_one();

        // Wait for initialization to finish.
        initialize_future.wait(None)?;

        let gizmo_renderer =
            crate::gizmos::renderer::GizmoRenderer::new(render_surface.context().clone())?;
        /*
        let test_gizmo_collection = {
            use crate::gizmos::*;
            let mut collection = Collection::new(transform::GizmoTransform {
                position: ultraviolet::Vec2 { x: 10.0, y: 10.0 },
                origin_pinning: transform::GizmoOriginPinning::Document,
                scale_pinning: transform::GizmoTransformPinning::Viewport,
                rotation: 0.0,
                rotation_pinning: transform::GizmoTransformPinning::Viewport,
            });
            let square = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: GizmoVisual::Shape {
                    shape: RenderShape::Rectangle {
                        position: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                        size: ultraviolet::Vec2 { x: 20.0, y: 20.0 },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [128, 255, 255, 255],
                },
                hit_shape: GizmoShape::None,
                hover_cursor: CursorOrInvisible::Invisible,
                interaction: GizmoInteraction::None,
                transform: transform::GizmoTransform::inherit_all(),
            };
            let square2 = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: GizmoVisual::Shape {
                    shape: RenderShape::Rectangle {
                        position: ultraviolet::Vec2 { x: 15.0, y: 8.0 },
                        size: ultraviolet::Vec2 { x: 40.0, y: 10.0 },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [128, 0, 200, 255],
                },
                hit_shape: GizmoShape::None,
                hover_cursor: CursorOrInvisible::Invisible,
                interaction: GizmoInteraction::None,
                transform: transform::GizmoTransform {
                    origin_pinning: transform::GizmoOriginPinning::Inherit,
                    rotation_pinning: transform::GizmoTransformPinning::Document,
                    ..transform::GizmoTransform::inherit_all()
                },
            };
            let circle = Gizmo {
                grab_cursor: CursorOrInvisible::Invisible,
                visual: GizmoVisual::Shape {
                    shape: RenderShape::Ellipse {
                        origin: ultraviolet::Vec2 { x: 10.0, y: 0.0 },
                        radii: ultraviolet::Vec2 { x: 20.0, y: 20.0 },
                        rotation: 0.0,
                    },
                    texture: None,
                    color: [128, 0, 0, 128],
                },
                hit_shape: GizmoShape::None,
                hover_cursor: CursorOrInvisible::Invisible,
                interaction: GizmoInteraction::None,
                transform: transform::GizmoTransform {
                    scale_pinning: transform::GizmoTransformPinning::Document,
                    ..transform::GizmoTransform::inherit_all()
                },
            };
            collection.push_top(square);
            collection.push_top(square2);
            collection.push_bottom(circle);
            collection
        };*/

        Ok(Self {
            render_context: render_surface.context().clone(),

            document_transform: document_transform.into(),
            viewport: (viewport_pos, viewport_size).into(),

            pipeline,
            render_pass,

            swap_after: SwapAfter::Empty.into(),
            read_buf: 0.into(),
            write_ready_notify: notify,

            document_images: document_image_views,
            document_image_bindings,

            surface_data: surface_data.into(),
            gizmo_renderer: gizmo_renderer.into(),

            cursor: None.into(),
            tool_render_as: pen_tools::RenderAs::None.into(),
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
        let mut lock = self.swap_after.write().unwrap();
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
        match &*self.swap_after.read().unwrap() {
            SwapAfter::Empty => false,
            SwapAfter::Now => true,
            SwapAfter::Fence(fence) => fence.is_signaled().unwrap(),
        }
    }
    pub async fn write(&self) -> DocumentViewportPreviewProxyImageGuard<'_> {
        self.write_ready_notify.notified().await;
        assert!(self.swap_after.read().unwrap().is_empty());
        // We are now the sole writer. Hopefully. Return the proxy:
        DocumentViewportPreviewProxyImageGuard {
            // Return whichever image is *not* the read buf. Uhm uh ordering??
            image: self.document_images
                [(self.read_buf.load(std::sync::atomic::Ordering::SeqCst) ^ 1) as usize]
                .clone(),
            is_submitted: false,
            proxy: &self,
        }
    }
    /// The area of the screen where the document is visible has changed
    pub fn viewport_changed(&self, position: cgmath::Point2<f32>, size: cgmath::Vector2<f32>) {
        *self.viewport.write() = (position, size);
        self.surface_data
            .blocking_write()
            .set_viewport_size(position, size);
    }
    pub async fn insert_document_transform(&self, new: crate::view_transform::DocumentTransform) {
        *self.document_transform.write().await = new.clone();
        self.surface_data.write().await.set_transform(new);
    }
    pub async fn get_view_transform(&self) -> Option<crate::pen_tools::ViewInfo> {
        // lock, clone, release asap
        let xform = match { self.document_transform.read().await.clone() } {
            crate::view_transform::DocumentTransform::Fit(f) => {
                let (pos, size) = *self.viewport.read();
                f.make_transform(
                    cgmath::Vector2 {
                        x: crate::DOCUMENT_DIMENSION as f32,
                        y: crate::DOCUMENT_DIMENSION as f32,
                    },
                    pos,
                    size,
                )?
            }
            crate::view_transform::DocumentTransform::Transform(t) => t,
        };
        let (pos, size) = self.get_viewport();

        Some(crate::pen_tools::ViewInfo {
            transform: xform,
            viewport_position: ultraviolet::Vec2 { x: pos.x, y: pos.y },
            viewport_size: ultraviolet::Vec2 {
                x: size.x,
                y: size.y,
            },
        })
    }
    pub fn insert_cursor(&self, new_cursor: Option<crate::gizmos::CursorOrInvisible>) {
        if let Some(mut cursor) = self.cursor.write().ok() {
            *cursor = new_cursor;
        }
    }
    pub fn insert_tool_render(&self, new_render_as: crate::pen_tools::RenderAs) {
        if let Some(mut render_as) = self.tool_render_as.write().ok() {
            *render_as = new_render_as;
        }
    }
    pub fn get_view_transform_sync(&self) -> Option<crate::view_transform::ViewTransform> {
        // lock, clone, release asap
        match { self.document_transform.blocking_read().clone() } {
            crate::view_transform::DocumentTransform::Fit(f) => {
                let (pos, size) = *self.viewport.read();
                f.make_transform(
                    cgmath::Vector2 {
                        x: crate::DOCUMENT_DIMENSION as f32,
                        y: crate::DOCUMENT_DIMENSION as f32,
                    },
                    pos,
                    size,
                )
            }
            crate::view_transform::DocumentTransform::Transform(t) => Some(t),
        }
    }
    pub fn get_viewport(&self) -> (cgmath::Point2<f32>, cgmath::Vector2<f32>) {
        *self.viewport.read()
    }
}
impl PreviewRenderProxy for DocumentViewportPreviewProxy {
    #[deny(unsafe_op_in_unsafe_fn)]
    unsafe fn render(
        &self,
        swapchain_image: Arc<vk::SwapchainImage>,
        swapchain_idx: u32,
    ) -> AnyResult<smallvec::SmallVec<[Arc<vk::PrimaryAutoCommandBuffer>; 2]>> {
        // Safety: contract forwarded to the contract of this fn.
        let image_idx = unsafe { self.read() };
        let read = self.surface_data.blocking_read();
        let commands = read.get_commands(swapchain_idx, image_idx)?;

        // Do we have anything to render?
        let tool_render_as = self
            .tool_render_as
            .read()
            .map_err(|_| anyhow::anyhow!("Poisoned"))?;
        let tool_buffer = if matches!(
            *tool_render_as,
            pen_tools::RenderAs::SharedGizmoCollection(..) | pen_tools::RenderAs::InlineGizmos(..)
        ) {
            let proj = crate::vk::projection::orthographic_vk(
                0.0,
                read.surface_dimensions[0] as f32,
                0.0,
                read.surface_dimensions[1] as f32,
                -1.0,
                1.0,
            );
            let proj: [[f32; 4]; 4] = proj.into();
            let proj: cgmath::Matrix4<f32> = proj.into();
            let mut visitor = self.gizmo_renderer.render_visit(
                swapchain_image,
                [
                    read.surface_dimensions[0] as f32,
                    read.surface_dimensions[1] as f32,
                ],
                self.get_view_transform_sync().unwrap(),
                proj,
            )?;
            match &*tool_render_as {
                pen_tools::RenderAs::SharedGizmoCollection(shared) => {
                    shared.blocking_read().visit_painter(&mut visitor);
                }
                pen_tools::RenderAs::InlineGizmos(gizmos) => {
                    for gizmo in gizmos.iter() {
                        gizmo.visit_painter(&mut visitor);
                    }
                }
                pen_tools::RenderAs::None => unreachable!(), // Guarded above
            }
            Some(visitor.build()?)
        } else {
            None
        };
        let mut vec = smallvec::SmallVec::with_capacity(2);
        vec.push(commands);
        if let Some(tool_buffer) = tool_buffer {
            vec.push(tool_buffer.into());
        }

        Ok(vec)
    }
    fn surface_changed(&self, render_surface: &render_device::RenderSurface) {
        let viewport = self.viewport.read().clone();
        let transform = self.document_transform.blocking_read().clone();

        let new = ProxySurfaceData::new(
            self.render_context.clone(),
            render_surface,
            self.render_pass.clone(),
            self.pipeline.clone(),
            &self.document_image_bindings,
            viewport.0,
            viewport.1,
            transform,
        );
        *self.surface_data.blocking_write() = new;
    }
    fn viewport_changed(&self, position: ultraviolet::Vec2, size: ultraviolet::Vec2) {
        let cg = (
            cgmath::Point2 {
                x: position.x,
                y: position.y,
            },
            cgmath::Vector2 {
                x: size.x,
                y: size.y,
            },
        );

        self.surface_data
            .blocking_write()
            .set_viewport_size(cg.0, cg.1);
        *self.viewport.write() = cg;
    }
    fn has_update(&self) -> bool {
        self.redraw_requested()
    }
    fn cursor(&self) -> Option<crate::gizmos::CursorOrInvisible> {
        self.cursor.read().ok().and_then(|read| *read)
    }
}

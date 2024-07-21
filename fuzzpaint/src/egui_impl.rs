use crate::render_device::RenderSurface;
use crate::vulkano_prelude::*;
use std::sync::Arc;

use egui_winit::{egui, winit};

/// Merge the textures data from one egui output into another. Useful for discarding Egui geomety
/// while maintaining its side-effects.
pub fn prepend_textures_delta(into: &mut egui::TexturesDelta, mut from: egui::TexturesDelta) {
    //Append into's data onto from, then copy the data back.
    //There is no convinient way to efficiently prepend a chunk of data, so this'll do :3
    from.free.reserve(into.free.len());
    from.free.extend(std::mem::take(&mut into.free));
    into.free = std::mem::take(&mut from.free);

    //Maybe duplicates work. Could optimize to discard redundant updates, but this probably
    //wont happen frequently
    from.set.reserve(into.set.len());
    from.set.extend(std::mem::take(&mut into.set));
    into.set = std::mem::take(&mut from.set);
}

pub struct Ctx {
    state: egui_winit::State,
    renderer: Render,

    redraw_this_frame: bool,
    redraw_next_frame: bool,
    full_output: Option<egui::FullOutput>,
    repaint_times: std::collections::VecDeque<std::time::Instant>,
}
impl Ctx {
    pub fn new(
        window: &winit::window::Window,
        render_surface: &RenderSurface,
    ) -> anyhow::Result<Self> {
        let mut renderer = Render::new(render_surface.context(), render_surface.format())?;
        renderer.gen_framebuffers(render_surface)?;

        let mut state = egui_winit::State::new(
            egui::Context::default(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
        );
        let properties = render_surface.context().physical_device().properties();
        let max_size = properties.max_image_dimension2_d;
        state.set_max_texture_side(max_size as usize);

        Ok(Self {
            state,
            renderer,
            redraw_this_frame: false,
            redraw_next_frame: true,
            full_output: None,
            repaint_times: std::collections::VecDeque::new(),
        })
    }
    pub fn wants_pointer_input(&self) -> bool {
        self.state.egui_ctx().wants_pointer_input()
    }
    pub fn replace_surface(&mut self, surface: &RenderSurface) -> anyhow::Result<()> {
        self.renderer.gen_framebuffers(surface)
    }
    pub fn push_winit_event(
        &mut self,
        window: &winit::window::Window,
        winit_event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        let response = self.state.on_window_event(window, winit_event);
        if response.repaint {
            self.redraw_this_frame = true;
        }
        response
    }
    /// Update the UI, regardless of if a frame was requested.
    pub fn update<T>(
        &'_ mut self,
        window: &winit::window::Window,
        f: impl FnOnce(&'_ egui::Context) -> T,
    ) -> T {
        let input = self.state.take_egui_input(window);

        let mut user_output = None;
        //Call into user code to draw
        let mut output = self
            .state
            .egui_ctx()
            .run(input, |ctx| user_output = Some(f(ctx)));

        // Schedule repaints. This doesn't handle multi-view.
        let now = std::time::Instant::now();
        output
            .viewport_output
            .iter()
            .for_each(|(_, viewport_output)| {
                let delay = viewport_output.repaint_delay;
                if delay.is_zero() {
                    // Wants immediate changes. mark next frame for redraw.
                    self.redraw_next_frame = true;
                } else if let Some(time) = now.checked_add(delay) {
                    // Egui gives absurd time delay if it doesn't want a repaint. Otherwise, enqueue it.
                    self.insert_repaint(time);
                }
            });

        //If there were outstanding deltas, accumulate those
        if let Some(old) = self.full_output.take() {
            prepend_textures_delta(&mut output.textures_delta, old.textures_delta);
        }

        self.state
            .handle_platform_output(window, output.platform_output.clone());
        //return platform outputs
        self.full_output = Some(output);

        // Closure always runs, this is not presented on a type level though.
        user_output.unwrap()
    }
    /// Peek the update flag without destroying it.
    pub fn peek_wants_update(&self) -> bool {
        let now = &std::time::Instant::now();
        self.redraw_this_frame || self.repaint_times.iter().any(|t| t <= now)
    }
    /// Wants to re-draw the screen. Check this after you've checked [`Self::wants_update`] and updated accordingly, but repaints may
    /// be requested even if an update is not. Check this frequently, but note that querying this destroys the flag.
    pub fn take_wants_update(&mut self) -> bool {
        // If redraw requests come from any source, return true.

        // Hey future self, I was and am very confused at how this is supposed to work. There is limited documentation for it
        // and I couldn't reverse-engineer the existing implementations.
        // For when you come back here to fix this crime (which may incur the odd extra render for no reason), here's some notes that
        // involved significant distress in order to discover:
        // * Re-running UI and re-rendering UI are *not* distinct events in egui and are referred to jointly as "repainting"
        // * a repaint delay of 0ms means "rerun ui logic again ASAP" (the most painful to figure out lmao)
        self.redraw_this_frame || self.take_past_repaints().is_some()
    }
    /// Insert a repaint time into the queue.
    fn insert_repaint(&mut self, when: std::time::Instant) {
        // Err means not found and gives where it *would* be.
        // Ignore OK, a redraw is already scheduled, despite how unlikely that case would be :P
        if let Err(idx) = self.repaint_times.binary_search(&when) {
            self.repaint_times.insert(idx, when);
        }
    }
    /// Take from the repaint time queue. All past times will be popped, and the latest will
    /// be returned, or None if no repaint times have passed.
    fn take_past_repaints(&mut self) -> Option<std::time::Instant> {
        let now = std::time::Instant::now();
        let first_in_future_idx = self.repaint_times.partition_point(|elem| elem <= &now);
        if first_in_future_idx != 0 {
            // Some are in the past!
            let last = self
                .repaint_times
                .get(first_in_future_idx - 1)
                .copied()
                .unwrap();
            // Delete all in the past
            self.repaint_times.drain(..first_in_future_idx);
            // Return the latest past time.
            Some(last)
        } else {
            None
        }
    }
    pub fn build_commands(
        &mut self,
        swapchain_idx: u32,
    ) -> Option<(
        Option<Arc<vk::PrimaryAutoCommandBuffer>>,
        Arc<vk::PrimaryAutoCommandBuffer>,
    )> {
        self.redraw_this_frame = std::mem::take(&mut self.redraw_next_frame);
        // Check if there's anything to draw!
        let output = self.full_output.take()?;

        let res: AnyResult<_> = try_block::try_block! {
            let transfer_commands = self.renderer.do_image_deltas(output.textures_delta).transpose()?;
            let tess_geom = self.state.egui_ctx().tessellate(output.shapes, output.pixels_per_point);
            let draw_commands = self.renderer.upload_and_render(output.pixels_per_point, swapchain_idx, &tess_geom)?;
            drop(tess_geom);

            Ok((transfer_commands, draw_commands))
        };

        Some(res.unwrap()) //also stinky
    }
}

use anyhow::Result as AnyResult;
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src:
        r"#version 460

        layout(binding = 0, set = 0) uniform sampler2D tex;

        layout(location = 0) in vec2 uv;
        layout(location = 1) in vec4 vertex_color;
        
        layout(location = 0) out vec4 out_color;

        vec3 toLinear(vec3 sRGB)
        {
            bvec3 cutoff = lessThan(sRGB, vec3(0.04045));
            vec3 higher = pow((sRGB + vec3(0.055))/vec3(1.055), vec3(2.4));
            vec3 lower = sRGB/vec3(12.92);
        
            return mix(higher, lower, cutoff);
        }

        void main() {
            //Texture is straight linear
            vec4 t = texture(tex, uv);

            //Color is premultiplied sRGB already, convert to straight linear
            vec3 c = vertex_color.a > 0.0 ? (vertex_color.rgb / vertex_color.a) : vec3(0.0);

            //sRGB to linear (needs to be slow + precise for color picker, unfortunately)
            //May be incorrect to do this in vertex shader,
            // due to linear interpolation for fragments. It is intuitively correct to do this here, but Egui
            // does not list the expected behavior.
            vec4 straight_vertex_color = vec4(toLinear(c), vertex_color.a);
            t *= straight_vertex_color;

            //Convert to premul linear
            t.rgb *= t.a;

            out_color = t;
        }",
    }
}
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src:
        r"#version 460

        layout(push_constant) uniform Matrix {
            mat4 ortho;
        } matrix;

        layout(location = 0) in vec2 pos;
        layout(location = 1) in vec4 color;
        layout(location = 2) in vec2 uv;

        layout(location = 0) out vec2 out_uv;
        layout(location = 1) out vec4 vertex_color;

        void main() {
            gl_Position = matrix.ortho * vec4(pos, 0.0, 1.0);
            out_uv = uv;
            vertex_color = color;
        }",
    }
}
#[derive(vk::BufferContents, vk::Vertex)]
#[repr(C)]
struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
}
impl From<egui::epaint::Vertex> for EguiVertex {
    fn from(value: egui::epaint::Vertex) -> Self {
        Self {
            pos: value.pos.into(),
            color: value.color.to_array(),
            uv: value.uv.into(),
        }
    }
}
struct Texture {
    image: Arc<vk::Image>,

    descriptor_set: Arc<vk::PersistentDescriptorSet>,
}
struct Render {
    remove_next_frame: Vec<egui::TextureId>,
    images: hashbrown::HashMap<egui::TextureId, Texture>,
    context: Arc<crate::render_device::RenderContext>,

    render_pass: Arc<vk::RenderPass>,
    pipeline: Arc<vk::GraphicsPipeline>,
    framebuffers: Vec<Arc<vk::Framebuffer>>,
}
impl Render {
    pub fn new(
        render_context: &Arc<crate::render_device::RenderContext>,
        surface_format: vk::Format,
    ) -> anyhow::Result<Self> {
        let device = render_context.device().clone();
        let renderpass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments : {
                swapchain_color : {
                    format: surface_format,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                },
            },
            pass: {
                color: [swapchain_color],
                depth_stencil: {},
            },
        )?;

        let matrix_push_constant = vk::PushConstantRange {
            offset: 0,
            stages: vk::ShaderStages::VERTEX,
            size: std::mem::size_of::<vs::Matrix>() as u32,
        };
        let image_sampler_layout = vk::DescriptorSetLayout::new(
            render_context.device().clone(),
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
        )?;

        let layout = vk::PipelineLayout::new(
            render_context.device().clone(),
            vk::PipelineLayoutCreateInfo {
                push_constant_ranges: vec![matrix_push_constant],
                set_layouts: vec![image_sampler_layout],
                ..Default::default()
            },
        )?;

        let fragment = fs::load(device.clone())?;
        let vertex = vs::load(device.clone())?;

        let fragment_entry = fragment.entry_point("main").unwrap();
        let vertex_entry = vertex.entry_point("main").unwrap();

        let fragment_stage = vk::PipelineShaderStageCreateInfo::new(fragment_entry);
        let vertex_stage = vk::PipelineShaderStageCreateInfo::new(vertex_entry.clone());

        let premul = {
            let premul = vk::AttachmentBlend {
                src_alpha_blend_factor: vk::BlendFactor::One,
                src_color_blend_factor: vk::BlendFactor::One,
                dst_alpha_blend_factor: vk::BlendFactor::OneMinusSrcAlpha,
                dst_color_blend_factor: vk::BlendFactor::OneMinusSrcAlpha,
                alpha_blend_op: vk::BlendOp::Add,
                color_blend_op: vk::BlendOp::Add,
            };
            let blend_states = vk::ColorBlendAttachmentState {
                blend: Some(premul),
                ..Default::default()
            };
            vk::ColorBlendState::with_attachment_states(1, blend_states)
        };

        let pipeline = vk::GraphicsPipeline::new(
            render_context.device().clone(),
            None,
            vk::GraphicsPipelineCreateInfo {
                color_blend_state: Some(premul),
                input_assembly_state: Some(vk::InputAssemblyState {
                    topology: vk::PrimitiveTopology::TriangleList,
                    primitive_restart_enable: false,
                    ..Default::default()
                }),
                multisample_state: Some(vk::MultisampleState::default()),
                rasterization_state: Some(vk::RasterizationState {
                    cull_mode: vk::CullMode::None,
                    ..Default::default()
                }),
                vertex_input_state: Some(
                    EguiVertex::per_vertex().definition(&vertex_entry.info().input_interface)?,
                ),
                // One dynamic viewport and scissor
                viewport_state: Some(vk::ViewportState::default()),
                dynamic_state: [vk::DynamicState::Viewport, vk::DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(renderpass.clone().first_subpass().into()),
                stages: smallvec::smallvec![vertex_stage, fragment_stage,],
                ..vk::GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;

        Ok(Self {
            remove_next_frame: Vec::new(),
            images: hashbrown::HashMap::default(),
            render_pass: renderpass,
            pipeline,
            context: render_context.clone(),
            framebuffers: Vec::new(),
        })
    }
    pub fn gen_framebuffers(
        &mut self,
        surface: &crate::render_device::RenderSurface,
    ) -> anyhow::Result<()> {
        let framebuffers: anyhow::Result<Vec<_>> = surface
            .swapchain_images()
            .iter()
            .map(|image| -> anyhow::Result<_> {
                let fb = vk::Framebuffer::new(
                    self.render_pass.clone(),
                    vk::FramebufferCreateInfo {
                        attachments: vec![vk::ImageView::new_default(image.clone())?],
                        ..Default::default()
                    },
                )?;

                Ok(fb)
            })
            .collect();

        //Treat error as fatal
        self.framebuffers = framebuffers?;

        Ok(())
    }
    pub fn upload_and_render(
        &self,
        scale_factor: f32,
        present_img_index: u32,
        tesselated_geom: &[egui::epaint::ClippedPrimitive],
    ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        let mut vert_buff_size = 0;
        let mut index_buff_size = 0;
        for clipped in tesselated_geom {
            match &clipped.primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    vert_buff_size += mesh.vertices.len();
                    index_buff_size += mesh.indices.len();
                }
                egui::epaint::Primitive::Callback(..) => {
                    //Todo. But I'm not sure I mind this feature being unimplemented :P
                    unimplemented!("Primitive Callback is not supported.");
                }
            }
        }

        if vert_buff_size == 0 || index_buff_size == 0 {
            let builder = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;
            return Ok(builder.build()?);
        }

        let mut vertex_vec = Vec::with_capacity(vert_buff_size);
        let mut index_vec = Vec::with_capacity(index_buff_size);

        for clipped in tesselated_geom {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                vertex_vec.extend(mesh.vertices.iter().copied().map(EguiVertex::from));
                index_vec.extend_from_slice(&mesh.indices);
            }
        }
        let vertices = vk::Buffer::from_iter(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertex_vec,
        )?;
        let indices = vk::Buffer::from_iter(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            index_vec,
        )?;

        let framebuffer = self
            .framebuffers
            .get(present_img_index as usize)
            .expect("Present image out-of-bounds.")
            .clone();

        let matrix = cgmath::ortho(
            0.0,
            framebuffer.extent()[0] as f32 / scale_factor,
            0.0,
            framebuffer.extent()[1] as f32 / scale_factor,
            -1.0,
            1.0,
        );

        let (texture_set_idx, _) = self.texture_set_layout();
        let pipeline_layout = self.pipeline.layout();

        let mut command_buffer_builder = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer_builder
            .begin_render_pass(
                vk::RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..vk::RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                vk::SubpassBeginInfo::default(),
            )?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_vertex_buffers(0, [vertices])?
            .bind_index_buffer(indices)?
            .set_viewport(
                0,
                smallvec::smallvec![vk::Viewport {
                    depth_range: 0.0..=1.0,
                    extent: framebuffer.extent().map(|dim| dim as f32),
                    offset: [0.0; 2],
                }],
            )?
            .push_constants(
                pipeline_layout.clone(),
                0,
                vs::Matrix {
                    ortho: matrix.into(),
                },
            )?;

        let mut start_vertex_buffer_offset: usize = 0;
        let mut start_index_buffer_offset: usize = 0;

        for clipped in tesselated_geom {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                // *Technically* it wants a float scissor rect. But.. oh well
                let offset = clipped.clip_rect.left_top();
                let offset = [
                    (offset.x.max(0.0) * scale_factor) as u32,
                    (offset.y.max(0.0) * scale_factor) as u32,
                ];

                let extent = clipped.clip_rect.size() * scale_factor;
                let extent = [extent.x as u32, extent.y as u32];

                command_buffer_builder
                    .set_scissor(0, smallvec::smallvec![vk::Scissor { offset, extent }])?
                    //Maybe there's a better way than rebinding every draw.
                    //shaderSampledImageArrayDynamicIndexing perhaps?
                    .bind_descriptor_sets(
                        self.pipeline.bind_point(),
                        pipeline_layout.clone(),
                        texture_set_idx,
                        self.images
                            .get(&mesh.texture_id)
                            .expect("Egui draw requested non-existent texture")
                            .descriptor_set
                            .clone(),
                    )?
                    .draw_indexed(
                        mesh.indices.len() as u32,
                        1,
                        start_index_buffer_offset as u32,
                        start_vertex_buffer_offset as i32,
                        0,
                    )?;
                start_index_buffer_offset += mesh.indices.len();
                start_vertex_buffer_offset += mesh.vertices.len();
            }
        }

        command_buffer_builder.end_render_pass(vk::SubpassEndInfo::default())?;
        let command_buffer = command_buffer_builder.build()?;

        Ok(command_buffer)
    }
    ///Get the descriptor set layout for the texture uniform. `(set_idx, layout)`
    fn texture_set_layout(&self) -> (u32, Arc<vk::DescriptorSetLayout>) {
        let pipe_layout = self.pipeline.layout();
        let layout = pipe_layout
            .set_layouts()
            .first()
            .expect("Egui shader needs a sampler!")
            .clone();
        (0, layout)
    }
    fn cleanup_textures(&mut self) {
        // Pending removals - clean up after last frame
        for texture in self.remove_next_frame.drain(..) {
            let _ = self.images.remove(&texture);
        }
    }
    /// Apply image deltas, optionally returning a command buffer filled with any
    /// transfers as needed.
    pub fn do_image_deltas(
        &mut self,
        deltas: egui::TexturesDelta,
    ) -> Option<anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>>> {
        // Deltas order of operations:
        // Set -> Draw -> Free

        // Clean up from last frame
        if !self.remove_next_frame.is_empty() {
            self.cleanup_textures();
        }

        // Queue up removals for next frame
        self.remove_next_frame.extend_from_slice(&deltas.free);

        // Perform changes
        if deltas.set.is_empty() {
            None
        } else {
            Some(self.do_image_deltas_set(deltas))
        }
    }
    fn do_image_deltas_set(
        &mut self,
        deltas: egui::TexturesDelta,
    ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        //Free is handled by do_image_deltas

        //Pre-allocate on the heap so we don't end up re-allocating a bunch as we populate
        let mut total_delta_size = 0;
        for (_, delta) in &deltas.set {
            total_delta_size += match &delta.image {
                egui::ImageData::Color(color) => color.width() * color.height() * 4,
                //We'll covert to 8bpp on upload
                egui::ImageData::Font(grey) => grey.width() * grey.height(),
            };
        }

        let mut data_vec = Vec::with_capacity(total_delta_size);
        for (_, delta) in &deltas.set {
            match &delta.image {
                egui::ImageData::Color(data) => {
                    data_vec.extend_from_slice(bytemuck::cast_slice(&data.pixels[..]));
                }
                egui::ImageData::Font(data) => {
                    //Convert f32 image to u8 unorm image
                    data_vec.extend(
                        data.pixels
                            .iter()
                            .map(|&f| (f * 255.0).clamp(0.0, 255.0) as u8),
                    );
                }
            }
        }

        let staging_buffer = vk::Buffer::from_iter(
            self.context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                sharing: vk::Sharing::Exclusive,
                usage: vk::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data_vec.into_iter(),
        )?;

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().transfer().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;

        //In case we need to allocate new textures.
        let (texture_set_idx, texture_set_layout) = self.texture_set_layout();

        let mut current_base_offset = 0;
        for (id, delta) in deltas.set {
            let entry = self.images.entry(id);
            //Generate if non-existent yet!
            let image: anyhow::Result<_> = match entry {
                hashbrown::hash_map::Entry::Vacant(v) => {
                    let format = match delta.image {
                        egui::ImageData::Color(_) => vk::Format::R8G8B8A8_UNORM,
                        egui::ImageData::Font(_) => vk::Format::R8_UNORM,
                    };
                    let extent = {
                        let mut extent = delta.pos.unwrap_or([0, 0]);
                        extent[0] += delta.image.width();
                        extent[1] += delta.image.height();

                        [extent[0] as u32, extent[1] as u32, 1]
                    };
                    let image = vk::Image::new(
                        self.context.allocators().memory().clone(),
                        vk::ImageCreateInfo {
                            array_layers: 1,
                            format,
                            extent,
                            usage: vk::ImageUsage::TRANSFER_DST | vk::ImageUsage::SAMPLED,
                            sharing: vk::Sharing::Exclusive,
                            ..Default::default()
                        },
                        vk::AllocationCreateInfo {
                            memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    )?;

                    let egui_to_vk_filter =
                        |egui_filter: egui::epaint::textures::TextureFilter| match egui_filter {
                            egui::TextureFilter::Linear => vk::Filter::Linear,
                            egui::TextureFilter::Nearest => vk::Filter::Nearest,
                        };

                    let mapping = if let egui::ImageData::Font(_) = delta.image {
                        //Font is one channel, representing percent coverage of white.
                        vk::ComponentMapping {
                            a: vk::ComponentSwizzle::Red,
                            r: vk::ComponentSwizzle::One,
                            g: vk::ComponentSwizzle::One,
                            b: vk::ComponentSwizzle::One,
                        }
                    } else {
                        vk::ComponentMapping::identity()
                    };

                    let view = vk::ImageView::new(
                        image.clone(),
                        vk::ImageViewCreateInfo {
                            component_mapping: mapping,
                            ..vk::ImageViewCreateInfo::from_image(&image)
                        },
                    )?;

                    //Could optimize here, re-using the four possible options of sampler.
                    let sampler = vk::Sampler::new(
                        self.context.device().clone(),
                        vk::SamplerCreateInfo {
                            mag_filter: egui_to_vk_filter(delta.options.magnification),
                            min_filter: egui_to_vk_filter(delta.options.minification),

                            ..Default::default()
                        },
                    )?;

                    let descriptor_set = vk::PersistentDescriptorSet::new(
                        self.context.allocators().descriptor_set(),
                        texture_set_layout.clone(),
                        [vk::WriteDescriptorSet::image_view_sampler(
                            texture_set_idx,
                            view.clone(),
                            sampler.clone(),
                        )],
                        [],
                    )?;
                    Ok(v.insert(Texture {
                        image,
                        descriptor_set,
                    })
                    .image
                    .clone())
                }
                hashbrown::hash_map::Entry::Occupied(o) => Ok(o.get().image.clone()),
            };
            let image = image?;

            let size = match &delta.image {
                egui::ImageData::Color(color) => color.width() * color.height() * 4,
                egui::ImageData::Font(grey) => grey.width() * grey.height(),
            };
            let start_offset = current_base_offset as u64;
            current_base_offset += size;

            let transfer_offset = delta.pos.unwrap_or([0, 0]);

            //Update regions according to delta
            let region = vk::BufferImageCopy {
                buffer_offset: start_offset,

                image_offset: [transfer_offset[0] as u32, transfer_offset[1] as u32, 0],
                buffer_image_height: delta.image.height() as u32,
                buffer_row_length: delta.image.width() as u32,
                image_extent: [delta.image.width() as u32, delta.image.height() as u32, 1],
                image_subresource: vk::ImageSubresourceLayers {
                    array_layers: 0..1,
                    aspects: vk::ImageAspects::COLOR,
                    mip_level: 0,
                },
                ..Default::default()
            };

            let transfer_info = vk::CopyBufferToImageInfo {
                regions: smallvec::smallvec![region],
                ..vk::CopyBufferToImageInfo::buffer_image(staging_buffer.clone(), image)
            };

            command_buffer.copy_buffer_to_image(transfer_info)?;
        }

        Ok(command_buffer.build()?)
    }
}

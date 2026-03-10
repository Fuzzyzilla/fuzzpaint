use crate::render_device::RenderSurface;
use crate::vulkano_prelude::*;
use std::sync::Arc;

use egui_winit::{egui, winit};

pub struct Callback {
    pub kind: CallbackKind,
}
pub enum CallbackKind {
    DocumentView { dummy_color: [u8; 4] },
}

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
    next_repaint: Option<std::time::Instant>,
}
impl Ctx {
    pub fn new(
        window: &winit::window::Window,
        render_surface: &RenderSurface,
    ) -> anyhow::Result<Self> {
        let mut renderer = Render::new(render_surface.context(), render_surface.format())?;
        renderer.gen_framebuffers(render_surface)?;

        let properties = render_surface.context().physical_device().properties();
        let max_size = properties.max_image_dimension2_d;

        let state = egui_winit::State::new(
            egui::Context::default(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            Some(max_size as usize),
        );
        {
            use egui::epaint::text;
            state.egui_ctx().add_font(egui::epaint::text::FontInsert {
                name: "Google Material Icons".to_owned(),
                data: egui::FontData {
                    font: std::borrow::Cow::Borrowed(material_icons::FONT),
                    index: 0,
                    tweak: egui::FontTweak::default(),
                },
                families: vec![text::InsertFontFamily {
                    family: text::FontFamily::Name("Google Material Icons".into()),
                    priority: text::FontPriority::Highest,
                }],
            });
        }

        Ok(Self {
            state,
            renderer,
            redraw_this_frame: false,
            redraw_next_frame: true,
            full_output: None,
            next_repaint: None,
        })
    }
    pub fn context(&self) -> &egui::Context {
        self.state.egui_ctx()
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
        mut f: impl FnMut(&'_ egui::Context) -> T,
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
        let now = std::time::Instant::now();
        self.redraw_this_frame
            || self
                .next_repaint
                .is_some_and(|next_repaint| now >= next_repaint)
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
        self.take_past_repaints().is_some() || self.redraw_this_frame
    }
    /// Insert a repaint time into the queue.
    fn insert_repaint(&mut self, when: std::time::Instant) {
        if let Some(next) = self.next_repaint.take() {
            if when < next {
                self.next_repaint = Some(when);
            }
        } else {
            self.next_repaint = Some(when);
        }
    }
    /// Take from the repaint time queue. All past times will be popped, and the latest will
    /// be returned, or None if no repaint times have passed.
    fn take_past_repaints(&mut self) -> Option<std::time::Instant> {
        let now = std::time::Instant::now();
        self.next_repaint.take_if(|next| now >= *next)
    }
    pub fn build_commands(
        &mut self,
        swapchain_idx: u32,
        clear: bool,
    ) -> Option<(
        Option<Arc<vk::PrimaryAutoCommandBuffer>>,
        Arc<vk::PrimaryAutoCommandBuffer>,
    )> {
        self.redraw_this_frame = std::mem::take(&mut self.redraw_next_frame);
        // Check if there's anything to draw!
        let output = self.full_output.take()?;

        let res: AnyResult<_> = try_block::try_block! {
            let transfer_commands = self.renderer.do_image_deltas(output.textures_delta)?;
            let tess_geom = self.state.egui_ctx().tessellate(output.shapes, output.pixels_per_point);
            let draw_commands = self.renderer.upload_and_render(output.pixels_per_point, swapchain_idx, &tess_geom, clear)?;
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
        layout(location = 1) in vec2 uv;
        layout(location = 2) in vec4 color;

        layout(location = 0) out vec2 out_uv;
        layout(location = 1) out vec4 vertex_color;

        void main() {
            gl_Position = matrix.ortho * vec4(pos, 0.0, 1.0);
            out_uv = uv;
            vertex_color = color;
        }",
    }
}
#[derive(vk::Vertex, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
}
impl EguiVertex {
    fn from_slice(egui: &[egui::epaint::Vertex]) -> &[Self] {
        // These are identical structs. :3
        bytemuck::cast_slice(egui)
    }
}
struct Texture {
    image: Arc<vk::Image>,

    descriptor_set: Arc<vk::PersistentDescriptorSet>,
}
struct Render {
    remove_next_frame: Vec<egui::TextureId>,
    samplers: hashbrown::HashMap<egui::epaint::textures::TextureOptions, Arc<vk::Sampler>>,
    images: hashbrown::HashMap<egui::TextureId, Texture>,
    context: Arc<crate::render_device::RenderContext>,

    render_pass: Arc<vk::RenderPass>,
    pipeline: Arc<vk::GraphicsPipeline>,
    framebuffers: Vec<Arc<vk::Framebuffer>>,

    vertex_index_arena: vulkano::buffer::allocator::SubbufferAllocator,
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

        let vertex_index_arena = vulkano::buffer::allocator::SubbufferAllocator::new(
            render_context.allocators().memory().clone(),
            vulkano::buffer::allocator::SubbufferAllocatorCreateInfo {
                // Usual per-frame usage is about 0x3_00_00
                arena_size: 0x4_00_00,
                buffer_usage: vk::BufferUsage::VERTEX_BUFFER | vk::BufferUsage::INDEX_BUFFER,
                // One write, one read. Staging is unnecessary, but we'd really
                // prefer it to be on the device if possible.
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        );

        Ok(Self {
            remove_next_frame: Vec::new(),
            samplers: hashbrown::HashMap::default(),
            images: hashbrown::HashMap::default(),
            render_pass: renderpass,
            pipeline,
            context: render_context.clone(),
            framebuffers: Vec::new(),

            vertex_index_arena,
        })
    }
    pub fn gen_framebuffers(
        &mut self,
        surface: &crate::render_device::RenderSurface,
    ) -> anyhow::Result<()> {
        let framebuffers: anyhow::Result<Vec<_>> = surface
            .swapchain_images()
            .unwrap()
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
        clear: bool,
    ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        let mut command_buffer_builder = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;

        let mut num_verts = 0;
        let mut num_indices = 0;
        for clipped in tesselated_geom {
            match &clipped.primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    num_verts += mesh.vertices.len();
                    num_indices += mesh.indices.len();
                }
                egui::epaint::Primitive::Callback(callback) => {
                    let Some(callback) = callback.callback.downcast_ref::<Callback>() else {
                        log::error!(
                            "unknown callback type {}",
                            std::any::type_name_of_val(callback.callback.as_ref())
                        );
                        continue;
                    };
                    let CallbackKind::DocumentView { dummy_color } = callback.kind;
                    log::debug!("{dummy_color:?}");
                }
            }
        }

        if num_verts == 0 || num_indices == 0 {
            // Nothing to do.
            return Ok(command_buffer_builder.build()?);
        }

        // It's possible to do these both in the same buffer, but its a lot less
        // convinient uwu.
        let vertices = self
            .vertex_index_arena
            .allocate_slice::<EguiVertex>(num_verts as _)?;
        let indices = self
            .vertex_index_arena
            .allocate_slice::<u32>(num_indices as _)?;
        {
            // Just allocated, no access conflicts possible.
            let mut vertices = &mut vertices.write().unwrap()[..];
            let mut indices = &mut indices.write().unwrap()[..];

            for clipped in tesselated_geom {
                if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                    vertices[..mesh.vertices.len()]
                        .copy_from_slice(EguiVertex::from_slice(&mesh.vertices));
                    vertices = &mut vertices[mesh.vertices.len()..];

                    indices[..mesh.indices.len()].copy_from_slice(&mesh.indices);
                    indices = &mut indices[mesh.indices.len()..];
                };
            }
        }

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

        if clear {
            command_buffer_builder.clear_color_image(vk::ClearColorImageInfo {
                clear_value: [0.0, 0.0, 0.0, 1.0].into(),
                regions: smallvec::smallvec![
                    framebuffer.attachments()[0].subresource_range().clone()
                ],
                ..vk::ClearColorImageInfo::image(framebuffer.attachments()[0].image().clone())
            })?;
        }
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

        let mut current_texture = None;
        for clipped in tesselated_geom {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                let top_left = clipped.clip_rect.left_top() * scale_factor;
                let top_left = [
                    top_left.x.max(0.0).floor() as u32,
                    top_left.y.max(0.0).floor() as u32,
                ];
                // Calculate the bottom right using ceiling rounding, and then
                // take the difference to find the extent. This is better than
                // just using the size of the rect, as it takes the differences
                // in how the two points got rounded into account. Otherwise,
                // there are single-pixel panel gaps!
                let bottom_right = clipped.clip_rect.right_bottom() * scale_factor;
                let bottom_right = [
                    bottom_right.x.min(u32::MAX as f32).ceil() as u32,
                    bottom_right.y.min(u32::MAX as f32).ceil() as u32,
                ];

                let extent = [bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]];

                // Egui tessellator dedups by scissor, no need to check for
                // changes.
                command_buffer_builder.set_scissor(
                    0,
                    smallvec::smallvec![vk::Scissor {
                        offset: top_left,
                        extent
                    }],
                )?;
                // Only bind texture on changes.
                if current_texture != Some(mesh.texture_id) {
                    //Maybe there's a better way than rebinding every draw.
                    //shaderSampledImageArrayDynamicIndexing perhaps?
                    command_buffer_builder.bind_descriptor_sets(
                        self.pipeline.bind_point(),
                        pipeline_layout.clone(),
                        texture_set_idx,
                        self.images
                            .get(&mesh.texture_id)
                            .expect("Egui draw requested non-existent texture")
                            .descriptor_set
                            .clone(),
                    )?;
                    current_texture = Some(mesh.texture_id);
                }
                command_buffer_builder.draw_indexed(
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
    /// Get or create the sampler for the given egui texture options. Mipmaps
    /// are not supported and mipmap mode is always treated as None.
    fn sampler_for(
        &mut self,
        mut options: egui::epaint::textures::TextureOptions,
    ) -> anyhow::Result<Arc<vk::Sampler>> {
        // We dont support mipmaps.
        options.mipmap_mode = None;
        match self.samplers.entry(options) {
            hashbrown::hash_map::Entry::Occupied(o) => Ok(o.get().clone()),
            hashbrown::hash_map::Entry::Vacant(v) => {
                use egui::epaint::textures;
                use vulkano::image::sampler::SamplerAddressMode;
                fn egui_to_vk_filter(egui_filter: textures::TextureFilter) -> vk::Filter {
                    match egui_filter {
                        textures::TextureFilter::Linear => vk::Filter::Linear,
                        textures::TextureFilter::Nearest => vk::Filter::Nearest,
                    }
                }

                let sampler = vk::Sampler::new(
                    self.context.device().clone(),
                    vk::SamplerCreateInfo {
                        mag_filter: egui_to_vk_filter(options.magnification),
                        min_filter: egui_to_vk_filter(options.minification),
                        address_mode: match options.wrap_mode {
                            egui::TextureWrapMode::ClampToEdge => {
                                [SamplerAddressMode::ClampToEdge; 3]
                            }
                            egui::TextureWrapMode::Repeat => [SamplerAddressMode::Repeat; 3],
                            egui::TextureWrapMode::MirroredRepeat => {
                                [SamplerAddressMode::MirroredRepeat; 3]
                            }
                        },
                        ..Default::default()
                    },
                )?;
                v.insert(sampler.clone());
                Ok(sampler)
            }
        }
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
    ) -> anyhow::Result<Option<Arc<vk::PrimaryAutoCommandBuffer>>> {
        // Deltas order of operations:
        // Set -> Draw -> Free

        // Clean up from last frame
        if !self.remove_next_frame.is_empty() {
            self.cleanup_textures();
        }

        // Queue up removals for next frame
        self.remove_next_frame.extend_from_slice(&deltas.free);

        // Perform Writes
        self.do_image_deltas_set(deltas)
    }
    fn do_image_deltas_set(
        &mut self,
        deltas: egui::TexturesDelta,
    ) -> anyhow::Result<Option<Arc<vk::PrimaryAutoCommandBuffer>>> {
        //Free is handled by do_image_deltas

        let staging_size_bytes = deltas
            .set
            .iter()
            .map(|(_, delta)| {
                [
                    u64::try_from(delta.image.bytes_per_pixel()),
                    delta.image.width().try_into(),
                    delta.image.height().try_into(),
                ]
                .into_iter()
                .product::<Result<u64, _>>()
            })
            .sum::<Result<u64, _>>()?;

        if staging_size_bytes == 0 {
            // Nothing to do.
            return Ok(None);
        }

        // Doesn't currently make sense to arena this, since (with my usage
        // pattern) it grows each time.
        let staging_buffer = vk::Buffer::new_slice::<u8>(
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
            staging_size_bytes,
        )?;
        {
            // Fill it up, in order. Just allocated, no access conflicts
            // possible.
            let mut staging_buffer = &mut staging_buffer.write().unwrap()[..];
            for (_, delta) in &deltas.set {
                let bytes = match &delta.image {
                    egui::ImageData::Color(c) => bytemuck::cast_slice(&c.pixels),
                };
                // Copy this chunk..
                staging_buffer[..bytes.len()].copy_from_slice(bytes);
                // then advance the starting pointer
                staging_buffer = &mut staging_buffer[bytes.len()..];
            }
        }

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().transfer().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;

        //In case we need to allocate new textures.
        let (texture_set_idx, texture_set_layout) = self.texture_set_layout();

        // Find and delete any images that are being upsized,
        // that way they can be reallocated from scratch.
        for (id, image) in &deltas.set {
            let hashbrown::hash_map::Entry::Occupied(entry) = self.images.entry(*id) else {
                continue;
            };
            let vk_image = &entry.get().image;
            let pos = image.pos.unwrap_or([0; 2]);
            let new_size = [pos[0] + image.image.width(), pos[1] + image.image.height()];

            if vk_image.extent()[0] < new_size[0] as u32
                || vk_image.extent()[1] < new_size[1] as u32
            {
                let _ = entry.remove();
            }
        }

        let mut current_base_offset = 0;
        for (id, delta) in deltas.set {
            // Create the sampler, given the egui sampling options.
            let sampler = self.sampler_for(delta.options)?;
            let entry = self.images.entry(id);

            let image = match entry {
                hashbrown::hash_map::Entry::Occupied(o) => o.get().image.clone(),
                hashbrown::hash_map::Entry::Vacant(v) => {
                    // Generate if non-existent yet!
                    let format = match delta.image {
                        egui::ImageData::Color(_) => vk::Format::R8G8B8A8_UNORM,
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

                    let view = vk::ImageView::new(
                        image.clone(),
                        vk::ImageViewCreateInfo {
                            ..vk::ImageViewCreateInfo::from_image(&image)
                        },
                    )?;

                    let descriptor_set = vk::PersistentDescriptorSet::new(
                        self.context.allocators().descriptor_set(),
                        texture_set_layout.clone(),
                        [vk::WriteDescriptorSet::image_view_sampler(
                            texture_set_idx,
                            view.clone(),
                            sampler,
                        )],
                        [],
                    )?;
                    v.insert(Texture {
                        image,
                        descriptor_set,
                    })
                    .image
                    .clone()
                }
            };

            let size = delta.image.bytes_per_pixel() * delta.image.width() * delta.image.height();

            let start_offset = current_base_offset as u64;
            current_base_offset += size;

            let transfer_offset = delta.pos.unwrap_or([0, 0]);
            //Update regions according to delta
            let region = vk::BufferImageCopy {
                buffer_offset: start_offset,

                image_offset: [transfer_offset[0] as u32, transfer_offset[1] as u32, 0],
                image_extent: [delta.image.width() as u32, delta.image.height() as u32, 1],
                image_subresource: vk::ImageSubresourceLayers {
                    array_layers: 0..1,
                    aspects: vk::ImageAspects::COLOR,
                    mip_level: 0,
                },
                // Packed.
                buffer_image_height: 0,
                buffer_row_length: 0,
                ..Default::default()
            };

            let transfer_info = vk::CopyBufferToImageInfo {
                regions: smallvec::smallvec![region],
                ..vk::CopyBufferToImageInfo::buffer_image(staging_buffer.clone(), image)
            };

            command_buffer.copy_buffer_to_image(transfer_info)?;
        }

        Ok(Some(command_buffer.build()?))
    }
}

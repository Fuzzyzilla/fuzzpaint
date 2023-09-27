use crate::render_device::*;
use crate::vulkano_prelude::*;
use std::sync::Arc;

/// Merge the textures data from one egui output into another. Useful for discarding Egui geomety
/// while maintaining its side-effects.
pub fn prepend_textures_delta(into: &mut egui::TexturesDelta, mut from: egui::TexturesDelta) {
    //Append into's data onto from, then copy the data back.
    //There is no convinient way to efficiently prepend a chunk of data, so this'll do :3
    from.free.reserve(into.free.len());
    from.free.extend(std::mem::take(&mut into.free).into_iter());
    into.free = std::mem::take(&mut from.free);

    //Maybe duplicates work. Could optimize to discard redundant updates, but this probably
    //wont happen frequently
    from.set.reserve(into.set.len());
    from.set.extend(std::mem::take(&mut into.set).into_iter());
    into.set = std::mem::take(&mut from.set);
}

pub struct EguiCtx {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: EguiRenderer,

    redraw_requested: bool,
    full_output: Option<egui::FullOutput>,
}
impl EguiCtx {
    pub fn new(
        window: &winit::window::Window,
        render_surface: &RenderSurface,
    ) -> anyhow::Result<Self> {
        let mut renderer = EguiRenderer::new(render_surface.context(), render_surface.format())?;
        renderer.gen_framebuffers(&render_surface)?;

        let mut state = egui_winit::State::new(window);
        state.set_pixels_per_point(egui_winit::native_pixels_per_point(window));
        let properties = render_surface.context().physical_device().properties();
        let max_size = properties.max_image_dimension2_d;
        state.set_max_texture_side(max_size as usize);

        Ok(Self {
            ctx: Default::default(),
            state,
            renderer,
            redraw_requested: true,
            full_output: None,
        })
    }
    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }
    pub fn replace_surface(&mut self, surface: &RenderSurface) -> anyhow::Result<()> {
        self.renderer.gen_framebuffers(surface)
    }
    pub fn push_winit_event<'e>(
        &mut self,
        winit_event: &winit::event::WindowEvent<'e>,
    ) -> egui_winit::EventResponse {
        let response = self.state.on_event(&self.ctx, winit_event);
        if response.repaint {
            self.redraw_requested = true;
        }
        response
    }
    pub fn update(
        &'_ mut self,
        window: &winit::window::Window,
        f: impl FnOnce(&'_ egui::Context) -> (),
    ) {
        //Call into user code to draw
        self.ctx.begin_frame(self.state.take_egui_input(window));
        f(&self.ctx);
        let mut output = self.ctx.end_frame();

        //If there were outstanding deltas, accumulate those
        if let Some(old) = self.full_output.take() {
            prepend_textures_delta(&mut output.textures_delta, old.textures_delta);
        }

        // handle repaint time
        if output.repaint_after.is_zero() {
            self.redraw_requested = true;
        } else {
            //Egui returns astronomically large number if it doesn't want a redraw - triggers overflow lol
            // let requested_instant = std::time::Instant::now().checked_add(output.repaint_after);
            // wawa, not implemented
        }

        self.state
            .handle_platform_output(window, &self.ctx, output.platform_output.clone());
        //return platform outputs
        self.full_output = Some(output);
    }
    pub fn needs_redraw(&self) -> bool {
        self.redraw_requested
    }
    pub fn build_commands(
        &mut self,
        swapchain_idx: u32,
    ) -> Option<(
        Option<vk::PrimaryAutoCommandBuffer>,
        vk::PrimaryAutoCommandBuffer,
    )> {
        self.redraw_requested = false;

        // Check if there's anything to draw!
        let Some(output) = self.full_output.take() else {
            return None;
        };

        let res: AnyResult<_> = try_block::try_block! {
            let transfer_commands = self.renderer.do_image_deltas(output.textures_delta).transpose()?;
            let tess_geom = self.ctx.tessellate(output.shapes);
            let draw_commands = self.renderer.upload_and_render(swapchain_idx, &tess_geom)?;
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
struct EguiTexture {
    image: Arc<vk::StorageImage>,

    descriptor_set: Arc<vk::PersistentDescriptorSet>,
}
struct EguiRenderer {
    remove_next_frame: Vec<egui::TextureId>,
    images: std::collections::HashMap<egui::TextureId, EguiTexture>,
    render_context: Arc<crate::render_device::RenderContext>,

    render_pass: Arc<vk::RenderPass>,
    pipeline: Arc<vk::GraphicsPipeline>,
    framebuffers: Vec<Arc<vk::Framebuffer>>,
}
impl EguiRenderer {
    pub fn new(
        render_context: &Arc<crate::render_device::RenderContext>,
        surface_format: vk::Format,
    ) -> anyhow::Result<Self> {
        let device = render_context.device().clone();
        let renderpass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments : {
                swapchain_color : {
                    load: Load,
                    store: Store,
                    format: surface_format,
                    samples: 1,
                },
            },
            pass: {
                color: [swapchain_color],
                depth_stencil: {},
            },
        )?;

        let fragment = fs::load(device.clone())?;
        let vertex = vs::load(device.clone())?;

        let fragment_entry = fragment.entry_point("main").unwrap();
        let vertex_entry = vertex.entry_point("main").unwrap();

        let mut blend_premul = vk::ColorBlendState::new(1);
        blend_premul.attachments[0].blend = Some(vk::AttachmentBlend {
            alpha_source: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
            color_source: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
            alpha_destination:
                vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
            color_destination:
                vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
            alpha_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
            color_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
        });

        let pipeline = vk::GraphicsPipeline::start()
            .vertex_shader(vertex_entry, vs::SpecializationConstants::default())
            .fragment_shader(fragment_entry, fs::SpecializationConstants::default())
            .vertex_input_state(EguiVertex::per_vertex())
            .render_pass(vk::Subpass::from(renderpass.clone(), 0).unwrap())
            .rasterization_state(vk::RasterizationState {
                cull_mode: vk::StateMode::Fixed(vk::CullMode::None),
                ..Default::default()
            })
            .input_assembly_state(vk::InputAssemblyState {
                topology: vk::PartialStateMode::Fixed(vk::PrimitiveTopology::TriangleList),
                primitive_restart_enable: vk::StateMode::Fixed(false),
            })
            .color_blend_state(blend_premul)
            .viewport_state(vk::ViewportState::Dynamic {
                count: 1,
                viewport_count_dynamic: false,
                scissor_count_dynamic: false,
            })
            .build(render_context.device().clone())?;

        Ok(Self {
            remove_next_frame: Vec::new(),
            images: Default::default(),
            render_pass: renderpass,
            pipeline,
            render_context: render_context.clone(),
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
        present_img_index: u32,
        tesselated_geom: &[egui::epaint::ClippedPrimitive],
    ) -> anyhow::Result<vk::PrimaryAutoCommandBuffer> {
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
                self.render_context.allocators().command_buffer(),
                self.render_context.queues().graphics().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;
            return Ok(builder.build()?);
        }

        let mut vertex_vec = Vec::with_capacity(vert_buff_size);
        let mut index_vec = Vec::with_capacity(index_buff_size);

        for clipped in tesselated_geom {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                vertex_vec.extend(mesh.vertices.iter().cloned().map(EguiVertex::from));
                index_vec.extend_from_slice(&mesh.indices);
            }
        }
        let vertices = vk::Buffer::from_iter(
            self.render_context.allocators().memory(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                usage: vk::MemoryUsage::Upload,
                ..Default::default()
            },
            vertex_vec,
        )?;
        let indices = vk::Buffer::from_iter(
            self.render_context.allocators().memory(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                usage: vk::MemoryUsage::Upload,
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
            framebuffer.extent()[0] as f32,
            0.0,
            framebuffer.extent()[1] as f32,
            -1.0,
            1.0,
        );

        let (texture_set_idx, _) = self.texture_set_layout();
        let pipeline_layout = self.pipeline.layout();

        let mut command_buffer_builder = vk::AutoCommandBufferBuilder::primary(
            self.render_context.allocators().command_buffer(),
            self.render_context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer_builder
            .begin_render_pass(
                vk::RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..vk::RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                vk::SubpassContents::Inline,
            )?
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_vertex_buffers(0, [vertices])
            .bind_index_buffer(indices)
            .set_viewport(
                0,
                [vk::Viewport {
                    depth_range: 0.0..1.0,
                    dimensions: framebuffer.extent().map(|dim| dim as f32),
                    origin: [0.0; 2],
                }],
            )
            .push_constants(
                pipeline_layout.clone(),
                0,
                vs::Matrix {
                    ortho: matrix.into(),
                },
            );

        let mut start_vertex_buffer_offset: usize = 0;
        let mut start_index_buffer_offset: usize = 0;

        for clipped in tesselated_geom {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                // *Technically* it wants a float scissor rect. But.. oh well
                let origin = clipped.clip_rect.left_top();
                let origin = [origin.x.max(0.0) as u32, origin.y.max(0.0) as u32];

                let dimensions = clipped.clip_rect.size();
                let dimensions = [dimensions.x as u32, dimensions.y as u32];

                command_buffer_builder
                    .set_scissor(0, [vk::Scissor { origin, dimensions }])
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
                    )
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

        command_buffer_builder.end_render_pass()?;
        let command_buffer = command_buffer_builder.build()?;

        Ok(command_buffer)
    }
    ///Get the descriptor set layout for the texture uniform. (set_idx, layout)
    fn texture_set_layout(&self) -> (u32, Arc<vk::DescriptorSetLayout>) {
        let pipe_layout = self.pipeline.layout();
        let layout = pipe_layout
            .set_layouts()
            .get(0)
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
    ) -> Option<anyhow::Result<vk::PrimaryAutoCommandBuffer>> {
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
    ) -> anyhow::Result<vk::PrimaryAutoCommandBuffer> {
        //Free is handled by do_image_deltas

        //Pre-allocate on the heap so we don't end up re-allocating a bunch as we populate
        let mut total_delta_size = 0;
        for (_, delta) in &deltas.set {
            total_delta_size += match &delta.image {
                egui::ImageData::Color(color) => color.width() * color.height() * 4,
                //We'll covert to 8bpp on upload
                egui::ImageData::Font(grey) => grey.width() * grey.height() * 1,
            };
        }

        let mut data_vec = Vec::with_capacity(total_delta_size);
        for (_, delta) in &deltas.set {
            match &delta.image {
                egui::ImageData::Color(data) => {
                    data_vec.extend_from_slice(bytemuck::cast_slice(&data.pixels[..]));
                }
                egui::ImageData::Font(data) => {
                    //Convert f32 image to u8 norm image
                    data_vec.extend(
                        data.pixels
                            .iter()
                            .map(|&f| (f * 255.0).clamp(0.0, 255.0) as u8),
                    );
                }
            }
        }

        //This is  dumb. Why can't i use the data directly? It's a slice of [u8]. Maybe (hopefully) it optimizes out?
        //TODO: Maybe mnually implement unsafe trait BufferContents to allow this without byte-by-byte iterator copying.
        let staging_buffer = vk::Buffer::from_iter(
            self.render_context.allocators().memory(),
            vk::BufferCreateInfo {
                sharing: vk::Sharing::Exclusive,
                usage: vk::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                usage: vk::MemoryUsage::Upload,
                ..Default::default()
            },
            data_vec.into_iter(),
        )?;

        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.render_context.allocators().command_buffer(),
            self.render_context.queues().transfer().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;

        //In case we need to allocate new textures.
        let (texture_set_idx, texture_set_layout) = self.texture_set_layout();

        let mut current_base_offset = 0;
        for (id, delta) in &deltas.set {
            let entry = self.images.entry(*id);
            //Generate if non-existent yet!
            let image: anyhow::Result<_> = match entry {
                std::collections::hash_map::Entry::Vacant(vacant) => {
                    let format = match delta.image {
                        egui::ImageData::Color(_) => vk::Format::R8G8B8A8_UNORM,
                        egui::ImageData::Font(_) => vk::Format::R8_UNORM,
                    };
                    let dimensions = {
                        let mut dimensions = delta.pos.unwrap_or([0, 0]);
                        dimensions[0] += delta.image.width();
                        dimensions[1] += delta.image.height();

                        vk::ImageDimensions::Dim2d {
                            width: dimensions[0] as u32,
                            height: dimensions[1] as u32,
                            array_layers: 1,
                        }
                    };
                    let image = vk::StorageImage::with_usage(
                        self.render_context.allocators().memory(),
                        dimensions,
                        format,
                        //We will not be using this StorageImage for storage :P
                        vk::ImageUsage::TRANSFER_DST | vk::ImageUsage::SAMPLED,
                        vk::ImageCreateFlags::empty(),
                        std::iter::empty(), //A puzzling difference in API from buffers - this just means Exclusive access.
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
                        self.render_context.device().clone(),
                        vk::SamplerCreateInfo {
                            mag_filter: egui_to_vk_filter(delta.options.magnification),
                            min_filter: egui_to_vk_filter(delta.options.minification),

                            ..Default::default()
                        },
                    )?;

                    let descriptor_set = vk::PersistentDescriptorSet::new(
                        self.render_context.allocators().descriptor_set(),
                        texture_set_layout.clone(),
                        [vk::WriteDescriptorSet::image_view_sampler(
                            texture_set_idx,
                            view.clone(),
                            sampler.clone(),
                        )],
                    )?;
                    Ok(vacant
                        .insert(EguiTexture {
                            image,
                            descriptor_set,
                        })
                        .image
                        .clone())
                }
                std::collections::hash_map::Entry::Occupied(occupied) => {
                    Ok(occupied.get().image.clone())
                }
            };
            let image = image?;

            let size = match &delta.image {
                egui::ImageData::Color(color) => color.width() * color.height() * 4,
                egui::ImageData::Font(grey) => grey.width() * grey.height() * 1,
            };
            let start_offset = current_base_offset as u64;
            current_base_offset += size;

            //The only way to get a struct of this is to call this method -
            //we need to redo many of the fields however.
            let transfer_info =
                vk::CopyBufferToImageInfo::buffer_image(staging_buffer.clone(), image);

            let transfer_offset = delta.pos.unwrap_or([0, 0]);

            command_buffer.copy_buffer_to_image(vk::CopyBufferToImageInfo {
                //Update regions according to delta
                regions: smallvec::smallvec![vk::BufferImageCopy {
                    buffer_offset: start_offset,
                    image_offset: [transfer_offset[0] as u32, transfer_offset[1] as u32, 0],
                    buffer_image_height: delta.image.height() as u32,
                    buffer_row_length: delta.image.width() as u32,
                    image_extent: [delta.image.width() as u32, delta.image.height() as u32, 1],
                    ..transfer_info.regions[0].clone()
                }],
                ..transfer_info
            })?;
        }

        Ok(command_buffer.build()?)
    }
}

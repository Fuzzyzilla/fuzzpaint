use std::sync::Arc;
use vulkano::sync::GpuFuture;

use anyhow::Result as AnyResult;

enum RemoteResource {
    Resident,
    Recoverable,
    Gone,
}

struct EguiEventAccumulator {
    events: Vec<egui::Event>,
    last_mouse_pos : Option<egui::Pos2>,
    last_modifiers : egui::Modifiers,
    //egui keys are 8-bit, so allocate 256 bools.
    held_keys : bitvec::array::BitArray<[u64; 4]>,
    has_focus : bool,
    hovered_files : Vec<egui::HoveredFile>,
    dropped_files : Vec<egui::DroppedFile>,
    screen_rect : Option<egui::Rect>,
    pixels_per_point: f32,

    is_empty: bool,
}
impl EguiEventAccumulator {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            last_mouse_pos: None,
            held_keys: bitvec::array::BitArray::ZERO,
            last_modifiers: egui::Modifiers::NONE,
            has_focus: true,
            hovered_files: Vec::new(),
            dropped_files: Vec::new(),
            screen_rect: None,
            pixels_per_point: 1.0,
            is_empty: false,
        }
    }
    pub fn accumulate(&mut self, event : &winit::event::Event<()>) {
        use egui::Event as GuiEvent;
        use winit::event::Event as SysEvent;
        //TODOS: Copy/Cut/Paste, IME, and Scroll + Zoom + MouseWheel confusion, Touch, AssistKit.
        match event {
            SysEvent::WindowEvent { event, .. } => {
                use winit::event::WindowEvent as WinEvent;
                match event {
                    WinEvent::Resized(size) => {
                        self.screen_rect = Some(egui::Rect{
                            min: egui::pos2(0.0, 0.0),
                            max: egui::pos2(size.width as f32, size.height as f32),
                        });
                        self.is_empty = false;
                    }
                    WinEvent::ScaleFactorChanged { scale_factor, .. } => {
                        self.pixels_per_point = *scale_factor as f32;
                        self.is_empty = false;
                    }
                    WinEvent::CursorLeft { .. } => {
                        self.last_mouse_pos = None;
                        self.events.push(
                            GuiEvent::PointerGone
                        );
                        self.is_empty = false;
                    }
                    WinEvent::CursorMoved { position, .. } => {
                        let position = egui::pos2(position.x as f32, position.y as f32);
                        self.last_mouse_pos = Some(position);
                        self.events.push(
                            GuiEvent::PointerMoved(position)
                        );
                        self.is_empty = false;
                    }
                    WinEvent::MouseInput { state, button, .. } => {
                        let Some(pos) = self.last_mouse_pos else {return};
                        let Some(button) = Self::winit_to_egui_mouse_button(*button) else {return};
                        self.events.push(
                            GuiEvent::PointerButton {
                                pos,
                                button,
                                pressed: if let winit::event::ElementState::Pressed = state {true} else {false},
                                modifiers: self.last_modifiers,
                            }
                        );
                        self.is_empty = false;
                    }
                    WinEvent::ModifiersChanged(state) => {
                        self.last_modifiers = egui::Modifiers{
                            alt: state.alt(),
                            command: state.ctrl(),
                            ctrl: state.ctrl(),
                            mac_cmd: false,
                            shift: state.shift(),
                        };
                        self.is_empty = false;
                    }
                    WinEvent::KeyboardInput { input, .. } => {
                        let Some(key) = input.virtual_keycode else {return};
                        let pressed = if let winit::event::ElementState::Pressed = input.state {true} else {false};
                        //Send a text event
                        if pressed {
                            if let Some(string) = Self::winit_key_to_text(key) {
                                self.events.push(
                                    GuiEvent::Text(string.to_string())
                                );
                            }
                            self.is_empty = false;
                        }

                        //If this is a key egui cares about, send it a key message as well.
                        let Some(key) = Self::winit_to_egui_key(key) else {return};

                        let prev_pressed = {
                            let mut key_state = self.held_keys.get_mut(key as u8 as usize).unwrap();
                            let prev_pressed = key_state.clone();
                            *key_state = pressed;
                            prev_pressed
                        };

                        self.events.push(
                            GuiEvent::Key {
                                key,
                                pressed,
                                repeat: prev_pressed && pressed,
                                modifiers: self.last_modifiers,
                            }
                        );
                        self.is_empty = false;
                    }
                    WinEvent::MouseWheel { delta, .. } => {
                        let (unit, delta) = match delta {
                            winit::event::MouseScrollDelta::LineDelta(x, y)
                                => (egui::MouseWheelUnit::Line, egui::vec2(*x, *y)),
                            winit::event::MouseScrollDelta::PixelDelta(delta)
                                => (egui::MouseWheelUnit::Point, egui::vec2(delta.x as f32, delta.y as f32)),
                        };
                        self.events.push(
                            GuiEvent::MouseWheel {
                                unit,
                                delta,
                                modifiers: self.last_modifiers,
                            }
                        );
                        self.is_empty = false;
                    }
                    WinEvent::TouchpadMagnify { delta, .. } => {
                        self.events.push(
                            GuiEvent::Zoom(*delta as f32)
                        );
                        self.is_empty = false;
                    }
                    WinEvent::Focused( has_focus ) => {
                        self.has_focus = *has_focus;
                        self.events.push(
                            GuiEvent::WindowFocused(self.has_focus)
                        );
                        self.is_empty = false;
                    }
                    WinEvent::HoveredFile(path) => {
                        self.hovered_files.push(
                            egui::HoveredFile{
                                mime: String::new(),
                                path: Some(path.clone()),
                            }
                        );
                        self.is_empty = false;
                    }
                    WinEvent::DroppedFile(path) => {
                        use std::io::Read;
                        let Ok(file) = std::fs::File::open(path) else {return};
                        //Surely there's a better way
                        let last_modified = if let Ok(Ok(modified)) = file.metadata().map(|md| md.modified()) {
                            Some(modified)
                        } else {
                            None
                        };

                        let bytes : Option<std::sync::Arc<[u8]>> = {
                            let mut reader = std::io::BufReader::new(file);
                            let mut data = Vec::new();

                            if let Ok(_) = reader.read_to_end(&mut data) {
                                Some(data.into())
                            } else {
                                None
                            }
                        };

                        self.dropped_files.push(
                            egui::DroppedFile{
                                name: path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                                bytes,
                                last_modified,
                                path: Some(path.clone()),
                            }
                        );
                        self.is_empty = false;
                    }
                    _ => ()
                }
            }
            _ => ()
        }
    }
    pub fn winit_to_egui_mouse_button(winit_button : winit::event::MouseButton) -> Option<egui::PointerButton> {
        use winit::event::MouseButton as WinitButton;
        use egui::PointerButton as EguiButton;
        match winit_button {
            WinitButton::Left => Some(EguiButton::Primary),
            WinitButton::Right => Some(EguiButton::Secondary),
            WinitButton::Middle => Some(EguiButton::Middle),
            WinitButton::Other(id) => {
                match id {
                    0 => Some(EguiButton::Extra1),
                    1 => Some(EguiButton::Extra2),
                    _ => None,
                }
            }
        }
    }
    pub fn winit_key_to_text(winit_key : winit::event::VirtualKeyCode) -> Option<&'static str> {
        use winit::event::VirtualKeyCode as Key;
        let char = match winit_key {
            Key::A => "A",
            Key::Apostrophe => "'",
            Key::At => "@",
            Key::B => "B",
            Key::Backslash => "\\",
            Key::C => "C",
            Key::Caret => "^",
            Key::Colon => ":",
            Key::D => "D",
            Key::E => "E",
            Key::F => "F",
            Key::G => "G",
            Key::Grave => "`",
            Key::H => "H",
            Key::I => "I",
            Key::J => "J",
            Key::K => "K",
            Key::L => "L",
            Key::LBracket => "[",
            Key::M => "M",
            Key::N => "N",
            Key::Key0 | Key::Numpad0 => "0",
            Key::Key1 | Key::Numpad1 => "1",
            Key::Key2 | Key::Numpad2 => "2",
            Key::Key3 | Key::Numpad3 => "3",
            Key::Key4 | Key::Numpad4 => "4",
            Key::Key5 | Key::Numpad5 => "5",
            Key::Key6 | Key::Numpad6 => "6",
            Key::Key7 | Key::Numpad7 => "7",
            Key::Key8 | Key::Numpad8 => "8",
            Key::Key9 | Key::Numpad9 => "9",
            Key::NumpadAdd | Key::Plus => "+",
            Key::NumpadMultiply | Key::Asterisk => "*",
            Key::NumpadSubtract | Key::Minus => "-",
            Key::NumpadDivide | Key::Slash => "/",
            Key::NumpadDecimal | Key::Period => ".",
            Key::Equals | Key::NumpadEquals => "=",
            Key::NumpadComma | Key::Comma => ",",
            Key::O => "O",
            Key::P => "P",
            Key::Q => "Q",
            Key::R => "R",
            Key::RBracket => "]",
            Key::S => "S",
            Key::Semicolon => ";",
            Key::Space => " ",
            Key::T => "T",
            Key::Tab => "\t",
            Key::U => "U",
            Key::V => "V",
            Key::W => "W",
            Key::X => "X",
            Key::Y => "Y",
            Key::Yen => "¥",
            Key::Z => "Z",
            _ => return None
        };
        Some(char)
    }
    pub fn winit_to_egui_key(winit_button : winit::event::VirtualKeyCode) -> Option<egui::Key> {
        use winit::event::VirtualKeyCode as winit_key;
        use egui::Key as egui_key;
        match winit_button {
            winit_key::Key0 | winit_key::Numpad0 => Some(egui_key::Num0),
            winit_key::Key1 | winit_key::Numpad1 => Some(egui_key::Num1),
            winit_key::Key2 | winit_key::Numpad2 => Some(egui_key::Num2),
            winit_key::Key3 | winit_key::Numpad3 => Some(egui_key::Num5),
            winit_key::Key4 | winit_key::Numpad4 => Some(egui_key::Num6),
            winit_key::Key5 | winit_key::Numpad5 => Some(egui_key::Num4),
            winit_key::Key6 | winit_key::Numpad6 => Some(egui_key::Num3),
            winit_key::Key7 | winit_key::Numpad7 => Some(egui_key::Num7),
            winit_key::Key8 | winit_key::Numpad8 => Some(egui_key::Num8),
            winit_key::Key9 | winit_key::Numpad9 => Some(egui_key::Num9),

            winit_key::Up => Some(egui_key::ArrowUp),
            winit_key::Down => Some(egui_key::ArrowDown),
            winit_key::Left => Some(egui_key::ArrowLeft),
            winit_key::Right => Some(egui_key::ArrowRight),

            winit_key::PageUp => Some(egui_key::PageUp),
            winit_key::PageDown => Some(egui_key::PageDown),

            winit_key::Home => Some(egui_key::Home),
            winit_key::End => Some(egui_key::End),

            winit_key::NumpadEnter | winit_key::Return => Some(egui_key::Enter),

            winit_key::Escape => Some(egui_key::Escape),

            winit_key::Space => Some(egui_key::Space),
            winit_key::Tab => Some(egui_key::Tab),

            winit_key::Delete => Some(egui_key::Delete),
            winit_key::Back => Some(egui_key::Backspace),

            winit_key::Insert => Some(egui_key::Insert),

            //Help
            winit_key::A => Some(egui_key::A),
            winit_key::B => Some(egui_key::B),
            winit_key::C => Some(egui_key::C),
            winit_key::D => Some(egui_key::D),
            winit_key::E => Some(egui_key::E),
            winit_key::F => Some(egui_key::F),
            winit_key::G => Some(egui_key::G),
            winit_key::H => Some(egui_key::H),
            winit_key::I => Some(egui_key::I),
            winit_key::J => Some(egui_key::J),
            winit_key::K => Some(egui_key::K),
            winit_key::L => Some(egui_key::L),
            winit_key::M => Some(egui_key::M),
            winit_key::N => Some(egui_key::N),
            winit_key::O => Some(egui_key::O),
            winit_key::P => Some(egui_key::P),
            winit_key::Q => Some(egui_key::Q),
            winit_key::R => Some(egui_key::R),
            winit_key::S => Some(egui_key::S),
            winit_key::T => Some(egui_key::T),
            winit_key::U => Some(egui_key::U),
            winit_key::V => Some(egui_key::V),
            winit_key::W => Some(egui_key::W),
            winit_key::X => Some(egui_key::X),
            winit_key::Y => Some(egui_key::Y),
            winit_key::Z => Some(egui_key::Z),
            
            _ => {
                eprintln!("Unimplemented Key {winit_button:?}");
                None
            },
        }
    }
    fn is_empty(&self) -> bool {
        self.is_empty
    }
    fn take_raw_input(&mut self) -> egui::RawInput {
        self.is_empty = true;
        egui::RawInput {
            modifiers : self.last_modifiers,
            events: std::mem::take(&mut self.events),
            focused: self.has_focus,
            //Unclear whether this should be taken or cloned.
            hovered_files: std::mem::take(&mut self.hovered_files),
            dropped_files: std::mem::take(&mut self.dropped_files),

            predicted_dt: 1.0 / 60.0,
            time: None,

            screen_rect: self.screen_rect,
            pixels_per_point: Some(self.pixels_per_point),
            //We cannot know yet!
            ..Default::default()
        }
    }
}
impl Default for EguiEventAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

fn egui_to_winit_cursor(cursor : egui::CursorIcon) -> Option<winit::window::CursorIcon> {
    use egui::CursorIcon as GuiCursor;
    use winit::window::CursorIcon as WinCursor;
    match cursor {
        GuiCursor::Alias => Some(WinCursor::Alias),
        GuiCursor::AllScroll => Some(WinCursor::AllScroll),
        GuiCursor::Cell => Some(WinCursor::Cell),
        GuiCursor::ContextMenu => Some(WinCursor::ContextMenu),
        GuiCursor::Copy => Some(WinCursor::Copy),
        GuiCursor::Crosshair => Some(WinCursor::Crosshair),
        GuiCursor::Default => Some(WinCursor::Default),
        GuiCursor::Grab => Some(WinCursor::Grab),
        GuiCursor::Grabbing => Some(WinCursor::Grabbing),
        GuiCursor::Help => Some(WinCursor::Help),
        GuiCursor::Move => Some(WinCursor::Move),
        GuiCursor::NoDrop => Some(WinCursor::NoDrop),
        GuiCursor::None => None,
        GuiCursor::NotAllowed => Some(WinCursor::NotAllowed),
        GuiCursor::PointingHand => Some(WinCursor::Hand),
        GuiCursor::Progress => Some(WinCursor::Progress),
        GuiCursor::ResizeColumn => Some(WinCursor::ColResize),
        GuiCursor::ResizeEast => Some(WinCursor::EResize),
        GuiCursor::ResizeHorizontal => Some(WinCursor::EwResize),
        GuiCursor::ResizeNeSw => Some(WinCursor::NeswResize),
        GuiCursor::ResizeNorth => Some(WinCursor::NResize),
        GuiCursor::ResizeNorthEast => Some(WinCursor::NeResize),
        GuiCursor::ResizeNorthWest => Some(WinCursor::NwResize),
        GuiCursor::ResizeNwSe => Some(WinCursor::NwseResize),
        GuiCursor::ResizeRow => Some(WinCursor::RowResize),
        GuiCursor::ResizeSouth => Some(WinCursor::SResize),
        GuiCursor::ResizeSouthEast => Some(WinCursor::SeResize),
        GuiCursor::ResizeSouthWest => Some(WinCursor::SwResize),
        GuiCursor::ResizeVertical => Some(WinCursor::NsResize),
        GuiCursor::ResizeWest => Some(WinCursor::WResize),
        GuiCursor::Text => Some(WinCursor::Text),
        GuiCursor::VerticalText => Some(WinCursor::VerticalText),
        GuiCursor::Wait => Some(WinCursor::Wait),
        GuiCursor::ZoomIn => Some(WinCursor::ZoomIn),
        GuiCursor::ZoomOut => Some(WinCursor::ZoomOut),
    }
}

mod EguiRenderer {
    use std::sync::Arc;
    use anyhow::{Result as AnyResult, Context};
    use vulkano::pipeline::{graphics::vertex_input::Vertex, Pipeline};
    mod fs {
        vulkano_shaders::shader!{
            ty: "fragment",
            src:
            r"#version 460

            layout(binding = 0, set = 0) uniform sampler2D tex;

            layout(location = 0) in vec2 uv;
            layout(location = 1) in vec4 vertex_color;
            layout(location = 0) out vec4 out_color;

            void main() {
                out_color = vertex_color * texture(tex, uv);
            }",
        }
    }
    mod vs {
        vulkano_shaders::shader!{
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
    #[derive(vulkano::buffer::BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex)]
    #[repr(C)]
    struct EguiVertex {
        #[format(R32G32_SFLOAT)]
        pos : [f32; 2],
        #[format(R8G8B8A8_UNORM)]
        color : [u8; 4],
        #[format(R32G32_SFLOAT)]
        uv : [f32; 2],
    }
    impl From<egui::epaint::Vertex> for EguiVertex {
        fn from(value: egui::epaint::Vertex) -> Self {
            Self {
                pos : value.pos.into(),
                color: value.color.to_array(),
                uv: value.uv.into()
            }
        }
    }
    struct EguiTexture {
        image : Arc<vulkano::image::StorageImage>,
        view : Arc<vulkano::image::view::ImageView<vulkano::image::StorageImage>>,
        sampler: Arc<vulkano::sampler::Sampler>,

        descriptor_set: Arc<vulkano::descriptor_set::PersistentDescriptorSet>,
    }
    pub struct EguiRenderer {
        images : std::collections::HashMap<egui::TextureId, EguiTexture>,
        render_context : Arc<super::RenderContext>,

        render_pass : Arc<vulkano::render_pass::RenderPass>,
        pipeline: Arc<vulkano::pipeline::graphics::GraphicsPipeline>,
        framebuffers: Vec<Arc<vulkano::render_pass::Framebuffer>>,
    }
    impl EguiRenderer {
        pub fn new(render_context: Arc<super::RenderContext>, surface_format: vulkano::format::Format) -> AnyResult<Self> {
            let device = render_context.device.clone();
            let renderpass = vulkano::single_pass_renderpass!(
                device.clone(),
                attachments : {
                    swapchain_color : {
                        load: Clear,
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

            let pipeline = vulkano::pipeline::graphics::GraphicsPipeline::start()
                .vertex_shader(vertex_entry, vs::SpecializationConstants::default())
                .fragment_shader(fragment_entry, fs::SpecializationConstants::default())
                .vertex_input_state(EguiVertex::per_vertex())
                .render_pass(vulkano::render_pass::Subpass::from(renderpass.clone(), 0).unwrap())
                .rasterization_state(
                    vulkano::pipeline::graphics::rasterization::RasterizationState{
                        cull_mode: vulkano::pipeline::StateMode::Fixed(vulkano::pipeline::graphics::rasterization::CullMode::None),
                        polygon_mode: vulkano::pipeline::graphics::rasterization::PolygonMode::Fill,
                        ..Default::default()
                    }
                )
                .input_assembly_state(
                    vulkano::pipeline::graphics::input_assembly::InputAssemblyState {
                        topology: vulkano::pipeline::PartialStateMode::Fixed(vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::TriangleList),
                        primitive_restart_enable: vulkano::pipeline::StateMode::Fixed(false),
                    }
                )
                .color_blend_state(
                     vulkano::pipeline::graphics::color_blend::ColorBlendState::new(1).blend_alpha()
                )
                .viewport_state(
                    vulkano::pipeline::graphics::viewport::ViewportState::Dynamic {
                        count: 1,
                        viewport_count_dynamic: false,
                        scissor_count_dynamic: false,
                    }
                )
                .build(render_context.device.clone())?;

            Ok(
                Self {
                    images: Default::default(),
                    render_pass: renderpass,
                    pipeline,
                    render_context: render_context.clone(),
                    framebuffers: Vec::new(),
                }
            )
        }
        pub fn gen_framebuffers(&mut self, surface: &super::RenderSurface) -> AnyResult<()> {
            let framebuffers : AnyResult<Vec<_>> =
                surface.swapchain_images
                .iter()
                .map(|image| -> AnyResult<_> {
                    let fb = vulkano::render_pass::Framebuffer::new(
                        self.render_pass.clone(),
                        vulkano::render_pass::FramebufferCreateInfo {
                            attachments: vec![
                                vulkano::image::view::ImageView::new_default(image.clone())?
                            ],
                            ..Default::default()
                        }
                    )?;

                    Ok(fb)
                }).collect();
            
            self.framebuffers = framebuffers?;

            Ok(())
        }
        pub fn upload_and_render(
            &self,
            present_img_index: u32,
            tesselated_geom: &[egui::epaint::ClippedPrimitive],
        ) -> AnyResult<vulkano::command_buffer::PrimaryAutoCommandBuffer> {
            let mut vert_buff_size = 0;
            let mut index_buff_size = 0;
            for clipped in tesselated_geom {
                match &clipped.primitive {
                    egui::epaint::Primitive::Mesh(mesh) => {
                        vert_buff_size += mesh.vertices.len();
                        index_buff_size += mesh.indices.len();
                    },
                    egui::epaint::Primitive::Callback(..) => {
                        //Todo. But I'm not sure I mind this feature being unimplemented :P
                        unimplemented!("Primitive Callback is not supported.");
                    },
                }
            }

            if vert_buff_size == 0 || index_buff_size == 0 {
                let builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
                    &self.render_context.command_buffer_alloc,
                    self.render_context.queues.graphics().idx(),
                    vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit
                )?;
                return Ok(
                    builder.build()?
                )
            }

            let mut vertex_vec = Vec::with_capacity(vert_buff_size);
            let mut index_vec = Vec::with_capacity(index_buff_size);


            for clipped in tesselated_geom {
                if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                    vertex_vec.extend(
                        mesh.vertices.iter()
                        .cloned()
                        .map(EguiVertex::from)
                    );
                    index_vec.extend_from_slice(&mesh.indices);
                }
            }

            let vertices = vulkano::buffer::Buffer::from_iter(
                &self.render_context.memory_alloc,
                vulkano::buffer::BufferCreateInfo {
                    usage: vulkano::buffer::BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                vulkano::memory::allocator::AllocationCreateInfo {
                    usage: vulkano::memory::allocator::MemoryUsage::Upload,
                    ..Default::default()
                },
                vertex_vec
            )?;
            let indices = vulkano::buffer::Buffer::from_iter(
                &self.render_context.memory_alloc,
                vulkano::buffer::BufferCreateInfo {
                    usage: vulkano::buffer::BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                vulkano::memory::allocator::AllocationCreateInfo {
                    usage: vulkano::memory::allocator::MemoryUsage::Upload,
                    ..Default::default()
                },
                index_vec
            )?;

            let framebuffer = self.framebuffers.get(present_img_index as usize).expect("Present image out-of-bounds.").clone();

            let matrix = cgmath::ortho(0.0, framebuffer.extent()[0] as f32, 0.0, framebuffer.extent()[1] as f32, -1.0, 1.0);

            let (texture_set_idx, _) = self.texture_set_layout();
            let pipeline_layout = self.pipeline.layout();

            let mut command_buffer_builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
                    &self.render_context.command_buffer_alloc,
                    self.render_context.queues.graphics().idx(),
                    vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit
                )?;
            command_buffer_builder
                .begin_render_pass(
                    vulkano::command_buffer::RenderPassBeginInfo{
                        clear_values: vec![
                            Some(
                                vulkano::format::ClearValue::Float([0.2, 0.2, 0.2, 1.0])
                            )
                        ],
                        ..vulkano::command_buffer::RenderPassBeginInfo::framebuffer(
                            framebuffer.clone()
                        )
                    },
                    vulkano::command_buffer::SubpassContents::Inline
                )?
                .bind_pipeline_graphics(self.pipeline.clone())
                .bind_vertex_buffers(0, [vertices])
                .bind_index_buffer(indices)
                .set_viewport(
                    0,
                    [vulkano::pipeline::graphics::viewport::Viewport{
                        depth_range: 0.0..1.0,
                        dimensions: framebuffer.extent().map(|dim| dim as f32),
                        origin: [0.0; 2],
                    }]
                )
                .push_constants(pipeline_layout.clone(), 0, vs::Matrix{
                    ortho: matrix.into()
                });

            let mut start_vertex_buffer_offset : usize = 0;
            let mut start_index_buffer_offset : usize = 0;


            for clipped in tesselated_geom {
                if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                    // *Technically* it wants a float scissor rect. But.. oh well
                    let origin = clipped.clip_rect.left_top();
                    let origin = [
                        origin.x.max(0.0) as u32,
                        origin.y.max(0.0) as u32
                    ];

                    let dimensions = clipped.clip_rect.size();
                    let dimensions = [
                        dimensions.x as u32,
                        dimensions.y as u32
                    ];

                    command_buffer_builder
                        .set_scissor(
                            0,
                            [
                                vulkano::pipeline::graphics::viewport::Scissor{
                                    origin,
                                    dimensions
                                }
                            ]
                        )
                        //Maybe there's a better way than rebinding every draw.
                        //shaderSampledImageArrayDynamicIndexing perhaps?
                        .bind_descriptor_sets(
                            self.pipeline.bind_point(),
                            pipeline_layout.clone(),
                            texture_set_idx,
                            self.images.get(&mesh.texture_id)
                                .expect("Egui draw requested non-existent texture")
                                .descriptor_set.clone()
                        )
                        .draw_indexed(
                            mesh.indices.len() as u32,
                            1,
                            start_index_buffer_offset as u32,
                            start_vertex_buffer_offset as i32,
                            0
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
        fn texture_set_layout(&self) -> (u32, Arc<vulkano::descriptor_set::layout::DescriptorSetLayout>) {
            let pipe_layout = self.pipeline.layout();
            let layout = pipe_layout.set_layouts().get(0).expect("Egui shader needs a sampler!").clone();
            (0, layout)
        }
        /// Apply image deltas, optionally returning a command buffer filled with any
        /// transfers as needed.
        pub fn do_image_deltas(
            &mut self,
            deltas : egui::TexturesDelta
        )  -> Option<AnyResult<vulkano::command_buffer::PrimaryAutoCommandBuffer>> {
            for free in deltas.free.iter() {
                self.images.remove(&free).unwrap();
            }

            if deltas.set.is_empty() {
                None
            } else {
                Some(
                    self.do_image_deltas_set(deltas)
                )
            }
        }
        fn do_image_deltas_set(
            &mut self,
            deltas : egui::TexturesDelta,
        ) -> AnyResult<vulkano::command_buffer::PrimaryAutoCommandBuffer> {
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
                            data.pixels.iter()
                                .map(|&f| {
                                    (f * 255.0).clamp(0.0, 255.0) as u8
                                })
                        );
                    }
                }
            }

            //This is  dumb. Why can't i use the data directly? It's a slice of [u8]. Maybe (hopefully) it optimizes out?
            //TODO: Maybe mnually implement unsafe trait BufferContents to allow this without byte-by-byte iterator copying.
            let staging_buffer = vulkano::buffer::Buffer::from_iter(
                &self.render_context.memory_alloc,
                vulkano::buffer::BufferCreateInfo {
                    sharing: vulkano::sync::Sharing::Exclusive,
                    usage: vulkano::buffer::BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                vulkano::memory::allocator::AllocationCreateInfo {
                    usage: vulkano::memory::allocator::MemoryUsage::Upload,
                    ..Default::default()
                },
                data_vec.into_iter()
            )?;

            let mut command_buffer =
                vulkano::command_buffer::AutoCommandBufferBuilder::primary(
                    &self.render_context.command_buffer_alloc,
                    self.render_context.queues.transfer().idx(),
                    vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit
                )?;
            
            //In case we need to allocate new textures.
            let (texture_set_idx, texture_set_layout) = self.texture_set_layout();

            let mut current_base_offset = 0;
            for (id, delta) in &deltas.set {
                let entry = self.images.entry(*id);
                //Generate if non-existent yet!
                let image : AnyResult<_> = match entry {
                    std::collections::hash_map::Entry::Vacant(vacant) => {
                        let format = match delta.image {
                            egui::ImageData::Color(_) => vulkano::format::Format::R8G8B8A8_UNORM,
                            egui::ImageData::Font(_) => vulkano::format::Format::R8_UNORM,
                        };
                        let dimensions = {
                            let mut dimensions = delta.pos.unwrap_or([0, 0]);
                            dimensions[0] += delta.image.width();
                            dimensions[1] += delta.image.height();
    
                            vulkano::image::ImageDimensions::Dim2d {
                                width: dimensions[0] as u32,
                                height: dimensions[1] as u32,
                                array_layers: 1
                            }
                        };
                        let image = vulkano::image::StorageImage::with_usage(
                            &self.render_context.memory_alloc,
                            dimensions,
                            format,
                            //We will not be using this StorageImage for storage :P
                            vulkano::image::ImageUsage::TRANSFER_DST | vulkano::image::ImageUsage::SAMPLED,
                            vulkano::image::ImageCreateFlags::empty(),
                            std::iter::empty() //A puzzling difference in API from buffers - this just means Exclusive access.
                        )?;
    
                        let egui_to_vulkano_filter = |egui_filter : egui::epaint::textures::TextureFilter| {
                            match egui_filter {
                                egui::TextureFilter::Linear => vulkano::sampler::Filter::Linear,
                                egui::TextureFilter::Nearest => vulkano::sampler::Filter::Nearest,
                            }
                        };
    
                        let mapping = if let egui::ImageData::Font(_) = delta.image {
                            //Font is one channel, representing percent coverage of white.
                            vulkano::sampler::ComponentMapping {
                                a: vulkano::sampler::ComponentSwizzle::Red,
                                r: vulkano::sampler::ComponentSwizzle::One,
                                g: vulkano::sampler::ComponentSwizzle::One,
                                b: vulkano::sampler::ComponentSwizzle::One,
                            }
                        } else {
                            vulkano::sampler::ComponentMapping::identity()
                        };

                        let view = vulkano::image::view::ImageView::new(
                            image.clone(),
                            vulkano::image::view::ImageViewCreateInfo {
                                component_mapping: mapping,
                                ..vulkano::image::view::ImageViewCreateInfo::from_image(&image)
                            }
                        )?;

                        //Could optimize here, re-using the four possible options of sampler.
                        let sampler = vulkano::sampler::Sampler::new(
                            self.render_context.device.clone(),
                            vulkano::sampler::SamplerCreateInfo {
                                mag_filter: egui_to_vulkano_filter(delta.options.magnification),
                                min_filter: egui_to_vulkano_filter(delta.options.minification),

                                ..Default::default()
                            }
                        )?;

                        let descriptor_set = vulkano::descriptor_set::persistent::PersistentDescriptorSet::new(
                            &self.render_context.descriptor_set_alloc,
                            texture_set_layout.clone(), 
                            [
                                vulkano::descriptor_set::WriteDescriptorSet::image_view_sampler(
                                    texture_set_idx, view.clone(), sampler.clone()
                                )
                            ]
                        )?;
                        Ok(
                            vacant.insert(         
                                EguiTexture {
                                    image,
                                    view,
                                    sampler,
                                    descriptor_set
                                }
                            ).image.clone()
                        )
                    },
                    std::collections::hash_map::Entry::Occupied(occupied) => {
                        Ok(occupied.get().image.clone())
                    }
                };
                let image = image.context("Failed to allocate Egui texture")?;

                let size = match &delta.image {
                    egui::ImageData::Color(color) => color.width() * color.height() * 4,
                    egui::ImageData::Font(grey) => grey.width() * grey.height() * 1,
                };
                let start_offset = current_base_offset as u64;
                current_base_offset += size;

                //The only way to get a struct of this is to call this method -
                //we need to redo many of the fields however.
                let transfer_info = 
                    vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                        staging_buffer.clone(),
                        image
                    );
                
                let transfer_offset = delta.pos.unwrap_or([0, 0]);

                command_buffer
                    .copy_buffer_to_image(
                        vulkano::command_buffer::CopyBufferToImageInfo {
                            //Update regions according to delta
                            regions: smallvec::smallvec![
                                vulkano::command_buffer::BufferImageCopy {
                                    buffer_offset: start_offset,
                                    image_offset: [
                                        transfer_offset[0] as u32,
                                        transfer_offset[1] as u32,
                                        0
                                    ],
                                    buffer_image_height: delta.image.height() as u32,
                                    buffer_row_length: delta.image.width() as u32,
                                    image_extent: [
                                        delta.image.width() as u32,
                                        delta.image.height() as u32,
                                        1
                                    ],
                                    ..transfer_info.regions[0].clone()
                                }
                            ],
                            ..transfer_info
                        }
                    )?;
            }

            Ok(
                command_buffer.build()?
            )
        }
    }
}


pub struct WindowSurface {
    event_loop : winit::event_loop::EventLoop<()>,
    win : Arc<winit::window::Window>,
}
impl WindowSurface {
    pub fn new() -> AnyResult<Self> {
        let event_loop = winit::event_loop::EventLoopBuilder::default().build();
        let win = winit::window::WindowBuilder::default()
            .with_transparent(false)
            .build(&event_loop)?;

        Ok(Self {
            event_loop: event_loop,
            win: Arc::new(win),
        })
    }
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    pub fn with_render_surface(self, render_surface: RenderSurface, render_context: Arc<RenderContext>)
        -> AnyResult<WindowRenderer> {
        
        let mut egui_renderer = EguiRenderer::EguiRenderer::new(render_context.clone(), render_surface.format().clone())?;
        egui_renderer.gen_framebuffers(&render_surface)?;
        Ok(
            WindowRenderer {
                egui_renderer,
                win: self.win,
                render_surface: Some(render_surface),
                render_context,
                event_loop: Some(self.event_loop),
                egui_ctx: Default::default(),
                egui_events: Default::default(),
                requested_redraw_time: Some(std::time::Instant::now())
            }
        )
    }
}

/// Merge the textures data from one egui output into another. Useful for discarding Egui out geomety
/// while maintaining it's side-effects.
pub fn append_textures_delta(into : &mut egui::TexturesDelta, from: egui::TexturesDelta) {
    into.free.reserve(from.free.len());
    for free in from.free.into_iter() {
        into.free.push(free);
    }

    //Maybe duplicates work. Could optimize to discard redundant updates, but this probably
    //wont happen frequently
    into.set.reserve(from.set.len());
    for set in from.set.into_iter() {
        into.set.push(set);
    }
}

pub struct WindowRenderer {
    event_loop : Option<winit::event_loop::EventLoop<()>>,
    win : Arc<winit::window::Window>,
    /// Always Some. This is to allow it to be take-able to be remade.
    /// Could None represent a temporary loss of surface that can be recovered from?
    render_surface : Option<RenderSurface>,
    render_context: Arc<RenderContext>,
    egui_ctx : egui::Context,
    egui_events : EguiEventAccumulator,
    egui_renderer: EguiRenderer::EguiRenderer,

    requested_redraw_time : Option<std::time::Instant>,
}
impl WindowRenderer {
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    /*
    pub fn gen_framebuffers(&mut self) {
        self.swapchain_framebuffers = Vec::with_capacity(self.render_surface.swapchain_images.len());

        self.swapchain_framebuffers.extend(
            self.render_surface.swapchain_images.iter()
                .map(|image| {
                    vulkano::render_pass::Framebuffer::
                })
        )
    }*/
    pub fn render_surface(&self) -> &RenderSurface {
        //this will ALWAYS be Some. The option is for taking from a mutable reference for recreation.
        &self.render_surface.as_ref().unwrap()
    }
    /// Recreate surface after loss or out-of-date. Todo: This only handles out-of-date and resize.
    pub fn recreate_surface(&mut self) -> AnyResult<()> {
        self.render_surface =
            Some(
                self.render_surface
                .take()
                .unwrap()
                .recreate(Some(self.window().inner_size().into()))?
            );

        self.egui_renderer.gen_framebuffers(self.render_surface.as_ref().unwrap())?;

        Ok(())
    }
    fn apply_platform_output(&mut self, out: &mut egui::PlatformOutput) {
        //Todo: Copied text
        if let Some(url) = out.open_url.take() {
            //Todo: x-platform lol
            let out = std::process::Command::new("xdg-open").arg(url.url).spawn();
            if let Err(e) = out {
                eprintln!("Failed to open url: {e:?}");
            }
        }

        if let Some(cursor) = egui_to_winit_cursor(out.cursor_icon) {
            self.win.set_cursor_icon(cursor);
            self.win.set_cursor_visible(true);
        } else {
            self.win.set_cursor_visible(false);
        }
    }
    pub fn run(mut self) -> ! {
        //There WILL be an event loop if we got here
        let event_loop = self.event_loop.take().unwrap();
        let mut egui_out = None::<egui::FullOutput>;

        let mut edit_str = String::new();

        event_loop.run(move |event, _, control_flow|{
            use winit::event::{Event, WindowEvent};

            self.egui_events.accumulate(&event);

            match event {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = winit::event_loop::ControlFlow::Exit;
                            return;
                        }
                        WindowEvent::Resized(..) => {
                            self.recreate_surface().expect("Failed to rebuild surface");
                        }
                        WindowEvent::ThemeChanged(t) => {
                            println!("Theme :0")
                        }
                        _ => ()
                    }
                }
                Event::MainEventsCleared => {
                    //No inputs, redraw (if any) is in the future. Skip!
                    //This doesn't work!
                    if self.egui_events.is_empty() && self.requested_redraw_time.map_or(true, |time| time > std::time::Instant::now()) {
                        return;
                    }

                    let raw_input = self.egui_events.take_raw_input();
                    self.egui_ctx.begin_frame(raw_input);

                    egui::Window::new("🦈 Baa")
                        .show(&self.egui_ctx, |ui| {
                            ui.label("Thing's wrong with it wha'ya mean");
                            ui.text_edit_multiline(&mut edit_str);
                        });

                    //Mutable so that we can take from it
                    let mut out = self.egui_ctx.end_frame();
                    if out.repaint_after.is_zero() {
                        //Repaint immediately!
                        self.requested_redraw_time = None;

                        self.window().request_redraw()
                    } else {
                        //Egui returns astronomically large number if it doesn't want a redraw - triggers overflow lol
                        let requested_instant = std::time::Instant::now().checked_add(out.repaint_after);
                        //Choose the minimum of the current and the requested.
                        //Should we queue all requests instead?
                        self.requested_redraw_time = self.requested_redraw_time.min(requested_instant);
                    }

                    //Requested time period is up, redraw!
                    if let Some(time) = self.requested_redraw_time {
                        if time <= std::time::Instant::now() {
                            self.window().request_redraw();
                            self.requested_redraw_time = None
                        }
                    }

                    self.apply_platform_output(&mut out.platform_output);

                    //There was old data, make sure we don't lose the texture updates when we replace it.
                    if let Some(old_output) = egui_out.take() {
                        append_textures_delta(&mut out.textures_delta, old_output.textures_delta);
                    }
                    egui_out = Some(out);
                }
                Event::RedrawRequested(..) => {
                    //Ensure there is actually data for us to draw
                    let Some(out) = egui_out.take() else {return};

                    let (idx, suboptimal, image_future) =
                        match vulkano::swapchain::acquire_next_image(
                            self.render_surface().swapchain.clone(),
                            None
                        ) {
                            Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                                eprintln!("Swapchain unusable.. Recreating");
                                //We cannot draw on this surface as-is. Recreate and request another try next frame.
                                self.recreate_surface().expect("Failed to recreate render surface after it went out-of-date.");
                                self.window().request_redraw();
                                return;
                            }
                            Err(_) => {
                                //Todo. Many of these errors are recoverable!
                                panic!("Surface image acquire failed!");
                            }
                            Ok(r) => r
                        };
                    let res : AnyResult<()> = try_block::try_block! {
                        let transfer_commands = self.egui_renderer.do_image_deltas(out.textures_delta);

                        let transfer_queue = self.render_context.queues.transfer().queue();
                        let render_queue = self.render_context.queues.graphics().queue();
                        let mut future : Box<dyn vulkano::sync::GpuFuture> = self.render_context.now().boxed();
                        if let Some(transfer_commands) = transfer_commands {
                            let buffer = transfer_commands?;
    
                            //Flush transfer commands, while we tesselate Egui output.
                            future = future.then_execute(transfer_queue.clone(), buffer)?
                                .boxed();
                        }
                        let tess_geom = self.egui_ctx.tessellate(out.shapes);
                        let draw_commands = self.egui_renderer.upload_and_render(idx, &tess_geom)?;
                        drop(tess_geom);

                        future.then_execute(render_queue.clone(),draw_commands)?
                            .join(image_future)
                            .then_swapchain_present(
                                self.render_context.queues.present().unwrap().queue().clone(),
                                vulkano::swapchain::SwapchainPresentInfo::swapchain_image_index(self.render_surface().swapchain.clone(), idx),
                            )
                            .then_signal_fence_and_flush()?
                            .wait(None)?;

                       // let draw_commands = self.egui_renderer.upload_and_render(idx, tesselated_geom);
                        Ok(())
                    };
                    if suboptimal {
                        self.recreate_surface().expect("Recreating suboptimal swapchain failed spectacularly");
                    }
                }
                _ => ()
            }
        });
    }
}

struct Queue {
    queue : Arc<vulkano::device::Queue>,
    family_idx : u32,
}
impl Queue {
    pub fn idx(&self) -> u32 {
        self.family_idx
    }
    pub fn queue(&self) -> &Arc<vulkano::device::Queue> {
        &self.queue
    }
}

enum QueueSrc {
    UseGraphics,
    Queue(Queue),
}

struct QueueIndices {
    graphics : u32,
    present : Option<u32>,
    compute : u32,
}
struct Queues {
    graphics_queue : Queue,
    present_queue : Option<QueueSrc>,
    compute_queue : QueueSrc,
}
impl Queues {
    pub fn present(&self) -> Option<&Queue> {
        match &self.present_queue {
            None => None,
            Some(QueueSrc::UseGraphics) => Some(self.graphics()),
            Some(QueueSrc::Queue(q)) => Some(q),
        }
    }
    pub fn graphics(&self) -> &Queue {
        &self.graphics_queue
    }
    pub fn compute(&self) -> &Queue {
        match &self.compute_queue {
            QueueSrc::UseGraphics => self.graphics(),
            QueueSrc::Queue(q) => q
        }
    }
    //No transfer queues yet, just use Graphics.
    pub fn transfer(&self) -> &Queue {
        self.graphics()
    }
    pub fn has_unique_compute(&self) -> bool {
        match &self.compute_queue {
            QueueSrc::UseGraphics => false,
            QueueSrc::Queue(..) => true
        }
    }
}

pub struct RenderSurface {
    swapchain: Arc<vulkano::swapchain::Swapchain>,
    surface: Arc<vulkano::swapchain::Surface>,
    swapchain_images: Vec<Arc<vulkano::image::SwapchainImage>>,
    //A future for each image, representing the time at which it has been presented and can start to be redrawn.
    fences: Vec<Option<Box<dyn vulkano::sync::GpuFuture>>>,

    swapchain_create_info: vulkano::swapchain::SwapchainCreateInfo,
}
impl RenderSurface {
    pub fn format(&self) -> vulkano::format::Format {
        self.swapchain_create_info.image_format.unwrap()
    }
    fn new(
        physical_device : Arc<vulkano::device::physical::PhysicalDevice>,
        device : Arc<vulkano::device::Device>,
        surface: Arc<vulkano::swapchain::Surface>,
        size: [u32; 2],
    ) -> AnyResult<Self> {
        
        let surface_info = vulkano::swapchain::SurfaceInfo::default();
        let capabilies = physical_device.surface_capabilities(&surface, surface_info.clone())?;

        let Some(&(format, color_space)) = physical_device.surface_formats(&surface, surface_info)?.first()
            else {return Err(anyhow::anyhow!("Device reported no valid surface formats."))};

        //Use mailbox for low-latency, if supported. Otherwise, FIFO is always supported.
        /*
        let present_mode =
            physical_device.surface_present_modes(&surface)
            .map(|mut modes| {
                if let Some(_) = modes.find(|mode| *mode == vulkano::swapchain::PresentMode::Mailbox) {
                    vulkano::swapchain::PresentMode::Mailbox
                } else {
                    vulkano::swapchain::PresentMode::Fifo
                }
            }).unwrap_or(vulkano::swapchain::PresentMode::Fifo);
        */
        let present_mode = vulkano::swapchain::PresentMode::Fifo;
        let image_count = {
            //Get one more then minimum, if maximum allows
            let min_image_count = capabilies.min_image_count + 1;
            if let Some(max_count) = capabilies.max_image_count {
                min_image_count.min(max_count)
            } else {
                min_image_count
            }
        };

        // We don't care!
        let alpha_mode = capabilies.supported_composite_alpha.clone().into_iter().next()
            .expect("Device provided no alpha modes");

        let swapchain_create_info = 
            vulkano::swapchain::SwapchainCreateInfo {
                min_image_count: image_count,
                image_format: Some(format),
                image_color_space: color_space,
                image_extent: size,
                image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: alpha_mode,
                present_mode,
                clipped: true, // We wont read the framebuffer.
                ..Default::default()
            };

        let (swapchain, images) = vulkano::swapchain::Swapchain::new(
                device.clone(),
                surface.clone(),
                swapchain_create_info.clone(),
        )?;

        Ok(Self {
            swapchain,
            surface: surface.clone(),
            swapchain_images: images,
            swapchain_create_info,
            fences: Vec::new(),
        })
    }
    pub fn recreate(self, new_size: Option<[u32; 2]>) -> AnyResult<Self> {
        let mut new_info = self.swapchain_create_info;
        if let Some(new_size) = new_size {
            new_info.image_extent = new_size;
        }
        let (swapchain, swapchain_images) = self.swapchain.recreate(new_info.clone())?;

        Ok(
            Self {
                swapchain,
                swapchain_images,
                swapchain_create_info: new_info,
                ..self
            }
        )
    }
    fn gen_framebuffers(&mut self) {
        let framebuffers = 
            self.swapchain_images.iter()
            .map(|image| {
                //vulkano::render_pass::Framebuffer::n
                ()
            });
    }
}
pub struct RenderContext {
    library : Arc<vulkano::VulkanLibrary>,
    instance : Arc<vulkano::instance::Instance>,
    physical_device : Arc<vulkano::device::physical::PhysicalDevice>,
    device : Arc<vulkano::device::Device>,
    queues: Queues,

    command_buffer_alloc : vulkano::command_buffer::allocator::StandardCommandBufferAllocator,
    memory_alloc: vulkano::memory::allocator::StandardMemoryAllocator,
    descriptor_set_alloc : vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator,
}


impl RenderContext {
    pub fn new_headless() -> AnyResult<Self> {
        unimplemented!()
    }
    pub fn new_with_window_surface(win: &WindowSurface) -> AnyResult<(Self, RenderSurface)> {
        let library = vulkano::VulkanLibrary::new()?;
        let required_instance_extensions = vulkano_win::required_extensions(&library);

        let instance = vulkano::instance::Instance::new(
            library.clone(),
            vulkano::instance::InstanceCreateInfo{
                application_name: Some("Fuzzpaint-vk".to_string()),
                application_version: vulkano::Version { major: 0, minor: 1, patch: 0 },
                enabled_extensions: required_instance_extensions,
                ..Default::default()
            }
        )?;

        let surface = vulkano_win::create_surface_from_winit(win.window(), instance.clone())?;
        let required_device_extensions = vulkano::device::DeviceExtensions {
            khr_swapchain : true,
            ..Default::default()
        };

        let Some((physical_device, queue_indices)) =
            Self::choose_physical_device(instance.clone(), required_device_extensions, Some(surface.clone()))?
            else {return Err(anyhow::anyhow!("Failed to find a suitable Vulkan device."))};

        println!("Chose physical device {} ({:?})", physical_device.properties().device_name, physical_device.properties().driver_info);

        let (device, queues) = Self::create_device(physical_device.clone(), queue_indices, required_device_extensions)?;

        println!("Got device :3");

        // We have a device! Now to create the swapchain..
        let image_size = win.window().inner_size();

        let render_head = RenderSurface::new(physical_device.clone(), device.clone(), surface.clone(), image_size.into())?;
        Ok(
            (
                Self {
                    command_buffer_alloc: vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(device.clone(), Default::default()),
                    memory_alloc: vulkano::memory::allocator::StandardMemoryAllocator::new_default(device.clone()),
                    descriptor_set_alloc: vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(device.clone()),
                    library,
                    instance,
                    device,
                    physical_device,
                    queues,
                },
                render_head
            )
        )
    }
    fn create_device(physical_device : Arc<vulkano::device::physical::PhysicalDevice>, queue_indices : QueueIndices, extensions: vulkano::device::DeviceExtensions)
        -> AnyResult<(Arc<vulkano::device::Device>, Queues)>{
        //Need a graphics queue.
        let mut graphics_queue_info =
            vulkano::device::QueueCreateInfo{
                queue_family_index: queue_indices.graphics,
                queues: vec![0.5],
                ..Default::default()
            };
        
        //Todo: what if compute and present end up in the same family, aside from graphics? Unlikely :)
        let (has_compute_queue, compute_queue_info) = {
            if queue_indices.compute == queue_indices.graphics {
                let capacity = physical_device.queue_family_properties()[graphics_queue_info.queue_family_index as usize].queue_count;
                //Is there room for another queue?
                if capacity as usize > graphics_queue_info.queues.len() {
                    graphics_queue_info.queues.push(0.5);

                    //In the graphics queue family. No queue info.
                    (true, None)
                } else {
                    //Share a queue with graphics.
                    (false, None)
                }
            } else {
                (true, Some(
                    vulkano::device::QueueCreateInfo{
                        queue_family_index: queue_indices.compute,
                        queues: vec![0.5],
                        ..Default::default()
                    }
                ))
            }
        };

        let present_queue_info = queue_indices.present.map( |present| {
            if present == queue_indices.graphics {
                let capacity = physical_device.queue_family_properties()[graphics_queue_info.queue_family_index as usize].queue_count;
                //Is there room for another queue?
                if capacity as usize > graphics_queue_info.queues.len() {
                    graphics_queue_info.queues.push(0.5);

                    //In the graphics queue family. No queue info.
                    (true, None)
                } else {
                    //Share a queue with graphics.
                    (false, None)
                }
            } else {
                (true, Some(
                    vulkano::device::QueueCreateInfo{
                        queue_family_index: present,
                        queues: vec![0.5],
                        ..Default::default()
                    }
                ))
            }
        });

        let mut create_infos = vec![graphics_queue_info];

        if let Some(compute_create_info) = compute_queue_info {
            create_infos.push(compute_create_info);
        }
        if let Some((_, Some(ref present_create_info))) = present_queue_info {
            create_infos.push(present_create_info.clone());
        }

        let (device, mut queues) = vulkano::device::Device::new(
            physical_device,
            vulkano::device::DeviceCreateInfo{
                enabled_extensions: extensions,
                enabled_features: vulkano::device::Features::empty(),
                queue_create_infos: create_infos,
                ..Default::default()
            }
        )?;

        let graphics_queue = Queue{
            queue: queues.next().unwrap(),
            family_idx: queue_indices.graphics,
        };
        //Todo: Are these indices correct?
        let compute_queue = if has_compute_queue {
            QueueSrc::Queue(
                Queue{
                    queue: queues.next().unwrap(),
                    family_idx: queue_indices.compute,
                }
            )
        } else {
            QueueSrc::UseGraphics
        };
        let present_queue = present_queue_info.map(|(has_present, _)| {
            if has_present {
                QueueSrc::Queue(
                    Queue{
                        queue: queues.next().unwrap(),
                        family_idx: queue_indices.present.unwrap(),
                    })
            } else {
                QueueSrc::UseGraphics
            }
        });
        assert!(queues.next().is_none());

        Ok(
            (
                device,
                Queues{
                    graphics_queue,
                    compute_queue,
                    present_queue
                }
            )
        )
    }
    /// Find a device that fits our needs, including the ability to present to the surface if in non-headless mode.
    /// Horrible signature - Returns Ok(None) if no device found, Ok(Some((device, queue indices))) if suitable device found.
    fn choose_physical_device(instance: Arc<vulkano::instance::Instance>, required_extensions: vulkano::device::DeviceExtensions, compatible_surface: Option<Arc<vulkano::swapchain::Surface>>)
        -> AnyResult<Option<(Arc<vulkano::device::physical::PhysicalDevice>, QueueIndices)>> {
        
        //TODO: does not respect queue family max queue counts. This will need to be redone in some sort of 
        //multi-pass shenanigan to properly find a good queue setup. Also requires that graphics and compute queues be transfer as well.
        let res = instance.enumerate_physical_devices()?
            .filter_map(|device| {
                use vulkano::device::QueueFlags;

                //Make sure it has what we need
                if !device.supported_extensions().contains(&required_extensions) {
                    return None;
                }

                let families = device.queue_family_properties();

                //Find a queue that supports the requested surface, if any
                let present_queue = compatible_surface
                    .clone()
                    .and_then(|surface| {
                        families.iter().enumerate().find(
                            |(family_idx, _)| {
                                //Assume error is false. Todo?
                                device.surface_support(*family_idx as u32, surface.as_ref()).unwrap_or(false)
                            }
                        )
                    });
                
                //We needed a present queue, but none was found. Disqualify this device!
                if compatible_surface.is_some() && present_queue.is_none() {
                    return None;
                }

                //We need a graphics queue, always! Otherwise, disqualify.
                let Some(graphics_queue) = families
                    .iter()
                    .enumerate()
                    .find(|q| {
                        q.1.queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::TRANSFER)
                    }) else {return None};
                
                //We need a compute queue. This can be the same as graphics, but preferably not.
                let graphics_supports_compute = graphics_queue.1.queue_flags.contains(QueueFlags::COMPUTE);

                //Find a different queue that supports compute
                let compute_queue = families
                    .iter()
                    .enumerate()
                    //Ignore the family we chose for graphics
                    .filter(|&(idx, _)| idx != graphics_queue.0)
                    .find(|q| {
                        q.1.queue_flags.contains(QueueFlags::COMPUTE | QueueFlags::TRANSFER)
                    });
                
                //Failed to find compute queue, shared or otherwise. Disqualify!
                if !graphics_supports_compute && compute_queue.is_none() {
                    return None;
                }

                Some(
                    (
                        device.clone(),
                        QueueIndices {
                            compute: compute_queue.unwrap_or(graphics_queue).0 as u32,
                            graphics: graphics_queue.0 as u32,
                            present: present_queue.map(|(idx, _)| idx as u32),

                        }
                    )
                )
            })
            .min_by_key(|(device, _)| {
                use vulkano::device::physical::PhysicalDeviceType;
                match device.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,

                    _ => 3
                }
            });
        
        Ok(res)
    }
    pub fn now(&self) -> vulkano::sync::future::NowFuture {
        vulkano::sync::now(self.device.clone())
    }
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    let window_surface = WindowSurface::new()?;
    let (render_context, render_surface) = RenderContext::new_with_window_surface(&window_surface)?;
    let render_context = Arc::new(render_context);
    let window_renderer = window_surface.with_render_surface(render_surface, render_context)?;
    println!("Made render context!");

    window_renderer.run();
}

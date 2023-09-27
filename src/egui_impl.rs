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
    events: EguiEventAccumulator,
    renderer: EguiRenderer,

    requested_redraw_times: std::collections::VecDeque<std::time::Instant>,
    immediate_redraw: bool,
    full_output: Option<egui::FullOutput>,
}
impl EguiCtx {
    pub fn new(render_surface: &RenderSurface) -> anyhow::Result<Self> {
        let mut renderer = EguiRenderer::new(render_surface.context(), render_surface.format())?;
        renderer.gen_framebuffers(&render_surface)?;

        Ok(Self {
            ctx: Default::default(),
            events: Default::default(),
            renderer,
            immediate_redraw: true,
            requested_redraw_times: std::collections::VecDeque::from_iter(std::iter::once(
                std::time::Instant::now(),
            )),
            full_output: None,
        })
    }
    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }
    pub fn replace_surface(&mut self, surface: &RenderSurface) -> anyhow::Result<()> {
        self.renderer.gen_framebuffers(surface)
    }
    pub fn push_winit_event(&mut self, winit_event: &winit::event::Event<'static, ()>) {
        self.events.accumulate(winit_event)
    }
    pub fn update(
        &'_ mut self,
        f: impl FnOnce(&'_ egui::Context) -> (),
    ) -> Option<egui::PlatformOutput> {
        if self.needs_refresh() {
            //Call into user code to draw
            self.ctx.begin_frame(self.events.take_raw_input());
            f(&self.ctx);
            let mut output = self.ctx.end_frame();

            //If there were outstanding deltas, accumulate those
            if let Some(old) = self.full_output.take() {
                prepend_textures_delta(&mut output.textures_delta, old.textures_delta);
            }

            // handle repaint time
            if output.repaint_after.is_zero() {
                self.immediate_redraw = true;
            } else {
                //Egui returns astronomically large number if it doesn't want a redraw - triggers overflow lol
                let requested_instant = std::time::Instant::now().checked_add(output.repaint_after);

                if let Some(instant) = requested_instant {
                    //Insert sorted
                    match self.requested_redraw_times.binary_search(&instant) {
                        Ok(..) => (), //A redraw is already scheduled for this exact instant
                        Err(pos) => self.requested_redraw_times.insert(pos, instant),
                    }
                }
            }

            //return platform outputs
            let platform_output = output.platform_output.take();
            self.full_output = Some(output);
            Some(platform_output)
        } else {
            None
        }
    }
    pub fn needs_redraw(&self) -> bool {
        self.immediate_redraw
            || self
                .requested_redraw_times
                .front()
                .map_or(false, |&time| time < std::time::Instant::now())
    }
    pub fn needs_refresh(&self) -> bool {
        let redraw_is_past = self
            .requested_redraw_times
            .front()
            .map_or(false, |&time| time < std::time::Instant::now());

        !self.events.is_empty() || redraw_is_past
    }
    pub fn build_commands(
        &mut self,
        swapchain_idx: u32,
    ) -> Option<(
        Option<vk::PrimaryAutoCommandBuffer>,
        vk::PrimaryAutoCommandBuffer,
    )> {
        self.immediate_redraw = false;
        let now = std::time::Instant::now();
        //Remove past redraw requests.
        self.requested_redraw_times.retain(|&time| time > now);

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

struct EguiEventAccumulator {
    events: Vec<egui::Event>,
    last_mouse_pos: Option<egui::Pos2>,
    last_modifiers: egui::Modifiers,
    //egui keys are 8-bit, so allocate 256 bools.
    held_keys: bitvec::array::BitArray<[u64; 4]>,
    has_focus: bool,
    hovered_files: Vec<egui::HoveredFile>,
    dropped_files: Vec<egui::DroppedFile>,
    screen_rect: Option<egui::Rect>,
    pixels_per_point: f32,

    last_taken: std::time::Instant,
    start_time: std::time::Instant,

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

            start_time: std::time::Instant::now(),
            last_taken: std::time::Instant::now(),
        }
    }
    pub fn accumulate(&mut self, event: &winit::event::Event<()>) {
        use egui::Event as GuiEvent;
        use winit::event::Event as SysEvent;
        //TODOS: Copy/Cut/Paste, IME, Touch, AssistKit.
        match event {
            SysEvent::WindowEvent { event, .. } => {
                use winit::event::WindowEvent as WinEvent;
                match event {
                    WinEvent::Resized(size) => {
                        self.screen_rect = Some(egui::Rect {
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
                        self.events.push(GuiEvent::PointerGone);
                        self.is_empty = false;
                    }
                    WinEvent::CursorMoved { position, .. } => {
                        let position = egui::pos2(position.x as f32, position.y as f32);
                        self.last_mouse_pos = Some(position);
                        self.events.push(GuiEvent::PointerMoved(position));
                        self.is_empty = false;
                    }
                    WinEvent::MouseInput { state, button, .. } => {
                        let Some(pos) = self.last_mouse_pos else {
                            return;
                        };
                        let Some(button) = Self::winit_to_egui_mouse_button(*button) else {
                            return;
                        };
                        self.events.push(GuiEvent::PointerButton {
                            pos,
                            button,
                            pressed: if let winit::event::ElementState::Pressed = state {
                                true
                            } else {
                                false
                            },
                            modifiers: self.last_modifiers,
                        });
                        self.is_empty = false;
                    }
                    WinEvent::ModifiersChanged(state) => {
                        self.last_modifiers = egui::Modifiers {
                            alt: state.alt(),
                            command: state.ctrl(),
                            ctrl: state.ctrl(),
                            mac_cmd: false,
                            shift: state.shift(),
                        };
                        self.is_empty = false;
                    }
                    WinEvent::ReceivedCharacter(ch) => {
                        //Various ascii codes that winit emits which break Egui
                        if ('\x00'..'\x20').contains(ch) || *ch == '\x7F' {
                            return;
                        };
                        self.events.push(GuiEvent::Text(ch.to_string()));
                        self.is_empty = false;
                    }
                    WinEvent::KeyboardInput { input, .. } => {
                        let Some(key) = input.virtual_keycode.and_then(Self::winit_to_egui_key)
                        else {
                            return;
                        };
                        let pressed = if let winit::event::ElementState::Pressed = input.state {
                            true
                        } else {
                            false
                        };

                        let prev_pressed = {
                            let mut key_state = self.held_keys.get_mut(key as u8 as usize).unwrap();
                            let prev_pressed = key_state.clone();
                            *key_state = pressed;
                            prev_pressed
                        };

                        self.events.push(GuiEvent::Key {
                            key,
                            pressed,
                            repeat: prev_pressed && pressed,
                            modifiers: self.last_modifiers,
                        });
                        self.is_empty = false;
                    }
                    WinEvent::MouseWheel { delta, .. } => {
                        let (unit, delta, pix_delta) = match delta {
                            winit::event::MouseScrollDelta::LineDelta(x, y) => {
                                (
                                    egui::MouseWheelUnit::Line,
                                    egui::vec2(*x, *y),
                                    //TODO: This 10.0 constant should come from the OS-defined
                                    //line size, for accessibility.
                                    egui::vec2(*x, *y) * 10.0,
                                )
                            }
                            winit::event::MouseScrollDelta::PixelDelta(delta) => (
                                egui::MouseWheelUnit::Point,
                                egui::vec2(delta.x as f32, delta.y as f32),
                                egui::vec2(delta.x as f32, delta.y as f32),
                            ),
                        };
                        self.events.push(GuiEvent::MouseWheel {
                            unit,
                            delta,
                            modifiers: self.last_modifiers,
                        });

                        //Emit scroll event as well.
                        {
                            let mut delta = pix_delta;

                            if self.last_modifiers.shift {
                                // Transpose scroll delta
                                std::mem::swap(&mut delta.x, &mut delta.y);
                            }

                            self.events.push(GuiEvent::Scroll(delta));
                        }
                        self.is_empty = false;
                    }
                    WinEvent::TouchpadMagnify { delta, .. } => {
                        self.events.push(GuiEvent::Zoom(*delta as f32));
                        self.is_empty = false;
                    }
                    WinEvent::Focused(has_focus) => {
                        self.has_focus = *has_focus;
                        self.events.push(GuiEvent::WindowFocused(self.has_focus));
                        self.is_empty = false;
                    }
                    WinEvent::HoveredFile(path) => {
                        self.hovered_files.push(egui::HoveredFile {
                            mime: String::new(),
                            path: Some(path.clone()),
                        });
                        self.is_empty = false;
                    }
                    WinEvent::DroppedFile(path) => {
                        use std::io::Read;
                        let Ok(file) = std::fs::File::open(path) else {
                            return;
                        };
                        //Surely there's a better way
                        let last_modified =
                            if let Ok(Ok(modified)) = file.metadata().map(|md| md.modified()) {
                                Some(modified)
                            } else {
                                None
                            };

                        let bytes: Option<std::sync::Arc<[u8]>> = {
                            let mut reader = std::io::BufReader::new(file);
                            let mut data = Vec::new();

                            if let Ok(_) = reader.read_to_end(&mut data) {
                                Some(data.into())
                            } else {
                                None
                            }
                        };

                        self.dropped_files.push(egui::DroppedFile {
                            name: path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .into_owned(),
                            bytes,
                            last_modified,
                            path: Some(path.clone()),
                        });
                        self.is_empty = false;
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
    pub fn winit_to_egui_mouse_button(
        winit_button: winit::event::MouseButton,
    ) -> Option<egui::PointerButton> {
        use egui::PointerButton as EguiButton;
        use winit::event::MouseButton as WinitButton;
        match winit_button {
            WinitButton::Left => Some(EguiButton::Primary),
            WinitButton::Right => Some(EguiButton::Secondary),
            WinitButton::Middle => Some(EguiButton::Middle),
            WinitButton::Other(id) => match id {
                0 => Some(EguiButton::Extra1),
                1 => Some(EguiButton::Extra2),
                _ => None,
            },
        }
    }
    pub fn winit_to_egui_key(winit_button: winit::event::VirtualKeyCode) -> Option<egui::Key> {
        use egui::Key as egui_key;
        use winit::event::VirtualKeyCode as winit_key;
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

            _ => None,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.is_empty
    }
    pub fn take_raw_input(&mut self) -> egui::RawInput {
        self.is_empty = true;

        // Take the old time, update it, and find the delta
        let old_time = std::mem::replace(&mut self.last_taken, std::time::Instant::now());
        let time_delta = self.last_taken - old_time;

        egui::RawInput {
            modifiers: self.last_modifiers,
            events: std::mem::take(&mut self.events),
            focused: self.has_focus,
            //Unclear whether this should be taken or cloned.
            hovered_files: std::mem::take(&mut self.hovered_files),
            dropped_files: std::mem::take(&mut self.dropped_files),

            predicted_dt: time_delta.as_secs_f32(),
            //Time since app launch.
            time: Some((self.last_taken - self.start_time).as_secs_f64()),

            screen_rect: self.screen_rect,
            pixels_per_point: Some(self.pixels_per_point),
            max_texture_side: Some(4096),
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

pub fn egui_to_winit_cursor(cursor: egui::CursorIcon) -> Option<winit::window::CursorIcon> {
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

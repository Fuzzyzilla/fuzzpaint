use std::sync::Arc;

use anyhow::Result as AnyResult;

struct EguiEventAccumulator {
    events: Vec<egui::Event>,
    last_mouse_pos : Option<egui::Pos2>,
    last_modifiers : egui::Modifiers,
    //TODO: Probably a much more efficient datastructure can be used here ;)
    held_keys : std::collections::BTreeMap<egui::Key, bool>,
    has_focus : bool,
    hovered_files : Vec<egui::HoveredFile>,
    dropped_files : Vec<egui::DroppedFile>,
    screen_rect : Option<egui::Rect>,
    pixels_per_point: f32,
}
impl EguiEventAccumulator {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            last_mouse_pos: None,
            held_keys: Default::default(),
            last_modifiers: egui::Modifiers::NONE,
            has_focus: true,
            hovered_files: Vec::new(),
            dropped_files: Vec::new(),
            screen_rect: None,
            pixels_per_point: 1.0,
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
                    }
                    WinEvent::ScaleFactorChanged { scale_factor, .. } => {
                        self.pixels_per_point = *scale_factor as f32;
                    }
                    WinEvent::CursorLeft { .. } => {
                        self.last_mouse_pos = None;
                        self.events.push(
                            GuiEvent::PointerGone
                        );
                    }
                    WinEvent::CursorMoved { position, .. } => {
                        let position = egui::pos2(position.x as f32, position.y as f32);
                        self.last_mouse_pos = Some(position);
                        self.events.push(
                            GuiEvent::PointerMoved(position)
                        );
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
                    }
                    WinEvent::ModifiersChanged(state) => {
                        self.last_modifiers = egui::Modifiers{
                            alt: state.alt(),
                            command: state.ctrl(),
                            ctrl: state.ctrl(),
                            mac_cmd: false,
                            shift: state.shift(),
                        };
                    }
                    WinEvent::KeyboardInput { input, .. } => {
                        let Some(key) = input.virtual_keycode.and_then(Self::winit_to_egui_key) else {return};
                        let pressed = if let winit::event::ElementState::Pressed = input.state {true} else {false};

                        let prev_pressed = {
                            let retained_pressed = self.held_keys.entry(key).or_insert(false);
                            let prev_pressed = retained_pressed.clone();
                            *retained_pressed = pressed;
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
                    }
                    WinEvent::TouchpadMagnify { delta, .. } => {
                        self.events.push(
                            GuiEvent::Zoom(*delta as f32)
                        );
                    }
                    WinEvent::Focused( has_focus ) => {
                        self.has_focus = *has_focus;
                        self.events.push(
                            GuiEvent::WindowFocused(self.has_focus)
                        );
                    }
                    WinEvent::HoveredFile(path) => {
                        self.hovered_files.push(
                            egui::HoveredFile{
                                mime: String::new(),
                                path: Some(path.clone()),
                            }
                        )
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
                        )
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
    fn take_raw_input(&mut self) -> egui::RawInput {
        egui::RawInput {
            modifiers : self.last_modifiers,
            events: std::mem::take(&mut self.events),
            focused: self.has_focus,
            //Unclear whether this should be taken or cloned.
            hovered_files: std::mem::take(&mut self.hovered_files),
            dropped_files: std::mem::take(&mut self.dropped_files),

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

pub struct WindowSurface {
    event_loop : winit::event_loop::EventLoop<()>,
    win : Arc<winit::window::Window>,
}
impl WindowSurface {
    pub fn new() -> AnyResult<Self> {
        let event_loop = winit::event_loop::EventLoopBuilder::default().build();
        let win = winit::window::WindowBuilder::default()
            .build(&event_loop)?;

        Ok(Self {
            event_loop: event_loop,
            win: Arc::new(win),
        })
    }
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    pub fn with_render_surface(self, render_surface: RenderSurface) -> WindowRenderer {
        WindowRenderer {
            win: self.win,
            render_surface,
            event_loop: Some(self.event_loop),
            egui_ctx: Default::default(),
            egui_events: Default::default(),
        }
    }
}

pub struct WindowRenderer {
    event_loop : Option<winit::event_loop::EventLoop<()>>,
    win : Arc<winit::window::Window>,
    render_surface : RenderSurface,
    egui_ctx : egui::Context,
    egui_events : EguiEventAccumulator,
}
impl WindowRenderer {
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
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
                        _ => ()
                    }
                }
                Event::MainEventsCleared => {
                    let raw_input = self.egui_events.take_raw_input();
                    self.egui_ctx.begin_frame(raw_input);

                    egui::CentralPanel::default()
                        .show(&self.egui_ctx, |ui| {
                            ui.hyperlink("https://www.youtube.com/");
                        });

                    //Mutable so that we can take from it
                    let mut out = self.egui_ctx.end_frame();
                    self.apply_platform_output(&mut out.platform_output);
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
    pub fn queue(&self) -> &vulkano::device::Queue {
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

    swapchain_create_info: vulkano::swapchain::SwapchainCreateInfo,
}
impl RenderSurface {
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
        let present_mode =
            physical_device.surface_present_modes(&surface)
            .map(|mut modes| {
                if let Some(_) = modes.find(|mode| *mode == vulkano::swapchain::PresentMode::Mailbox) {
                    vulkano::swapchain::PresentMode::Mailbox
                } else {
                    vulkano::swapchain::PresentMode::Fifo
                }
            }).unwrap_or(vulkano::swapchain::PresentMode::Fifo);

        let image_count = {
            //Get one more then minimum, if maximum allows
            let min_image_count = capabilies.min_image_count + 1;
            if let Some(max_count) = capabilies.max_image_count {
                min_image_count.min(max_count)
            } else {
                min_image_count
            }
        };

        let swapchain_create_info = 
            vulkano::swapchain::SwapchainCreateInfo {
                min_image_count: image_count,
                image_format: Some(format),
                image_color_space: color_space,
                image_extent: size,
                image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: vulkano::swapchain::CompositeAlpha::Inherit,
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
        })
    }
}
struct RenderContext {
    library : Arc<vulkano::VulkanLibrary>,
    instance : Arc<vulkano::instance::Instance>,
    physical_device : Arc<vulkano::device::physical::PhysicalDevice>,
    device : Arc<vulkano::device::Device>,
    queues: Queues,

    command_buffer_alloc : vulkano::command_buffer::allocator::StandardCommandBufferAllocator,
    memory_alloc: vulkano::memory::allocator::StandardMemoryAllocator,
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
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    let window_surface = WindowSurface::new()?;
    let (render_context, render_surface) = RenderContext::new_with_window_surface(&window_surface)?;
    let window_renderer = window_surface.with_render_surface(render_surface);
    println!("Made render context!");

    window_renderer.run();
}

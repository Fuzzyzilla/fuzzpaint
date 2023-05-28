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
                        // -w- glorious
                        let Some(Some(key)) = input.virtual_keycode.map(Self::winit_to_egui_key) else {return};
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

pub struct Head {
    event_loop : Option<winit::event_loop::EventLoop<()>>,
    win : winit::window::Window,
    egui_ctx : egui::Context,
    egui_events : EguiEventAccumulator,
}
impl Head {
    pub fn new() -> AnyResult<Self> {
        let event_loop = winit::event_loop::EventLoopBuilder::default().build();
        let win = winit::window::WindowBuilder::default()
            .build(&event_loop)?;

        let egui_ctx = egui::Context::default();

        Ok(Self {
            egui_ctx,
            egui_events: EguiEventAccumulator::new(),
            event_loop: Some(event_loop),
            win,
        })
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

fn main() {
    Head::new().unwrap().run();
}

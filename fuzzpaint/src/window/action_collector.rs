use crate::actions::{
    self,
    hotkeys::{KeyboardHotkey, Modifiers},
};

#[derive(Default)]
pub struct ActionCollector {
    /// Maps keys to the number of times they are shadowed.
    current_hotkeys: hashbrown::HashMap<KeyboardHotkey, usize>,
    currently_pressed: Vec<winit::keyboard::KeyCode>,
    modifiers: Modifiers,
}
impl ActionCollector {
    pub fn push_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::WindowEvent;

        let hotkeys = crate::global::hotkeys::Hotkeys::read();
        match event {
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } => {
                let winit::keyboard::PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                match event.state {
                    winit::event::ElementState::Pressed => {
                        if !self.currently_pressed.contains(&code) {
                            self.currently_pressed.push(code);
                        }
                    }
                    winit::event::ElementState::Released => {
                        if let Some(idx) = self
                            .currently_pressed
                            .iter()
                            .enumerate()
                            .find_map(|(i, &key)| (key == code).then_some(i))
                        {
                            self.currently_pressed.remove(idx);
                        }
                    }
                }
            }
            WindowEvent::Focused(false) => {
                self.currently_pressed = Vec::new();
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = Modifiers::ctrl_alt_shift(
                    m.state()
                        .intersects(winit::keyboard::ModifiersState::CONTROL),
                    m.state().intersects(winit::keyboard::ModifiersState::ALT),
                    m.state().intersects(winit::keyboard::ModifiersState::SHIFT),
                );
                self.update();
            }
            _ => (),
        }
    }
    pub fn update(&mut self) {}
}

use super::hotkeys::HotkeyShadow;

pub struct WinitKeyboardActionCollector {
    /// Maps keys to the number of times they are shadowed.
    current_hotkeys: std::collections::HashMap<super::hotkeys::KeyboardHotkey, usize>,
    currently_pressed: std::collections::HashSet<winit::event::VirtualKeyCode>,
    ctrl: bool,
    shift: bool,
    alt: bool,
}
impl WinitKeyboardActionCollector {
    pub fn push_event<'a>(&mut self, event: &winit::event::WindowEvent) {
        let hotkeys = crate::GlobalHotkeys::get();

        use winit::event::WindowEvent;
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                let Some(key) = input.virtual_keycode else {
                    return;
                };

                let was_pressed = self.currently_pressed.contains(&key);
                let is_pressed = input.state == winit::event::ElementState::Pressed;

                // Update currently_pressed set accordingly:
                if is_pressed {
                    self.currently_pressed.remove(&key);
                } else if !was_pressed {
                    self.currently_pressed.insert(key);
                }

                // Depending on the status of ctrl, shift, and alt, this key
                // event could correspond to eight different actions. Check
                // them all!

                // Copy so that the iter does not borrow self.
                let ctrl = self.ctrl;
                let shift = self.shift;
                let alt = self.alt;
                let possible_keys = (0u8..(1
                    << (ctrl as u8 + shift as u8 + alt as u8)))
                    .into_iter()
                    .map(|mut bits| {
                        // Generates all unique combos of each flag where self.<flag> is set.
                        // Or false if not set.
                        let mut consume = |condition: bool| {
                            if condition {
                                let bit = bits & 1 == 1;
                                bits >>= 1;
                                bit
                            } else {
                                false
                            }
                        };
                        super::hotkeys::KeyboardHotkey {
                            key,
                            alt: consume(alt),
                            shift: consume(shift),
                            ctrl: consume(ctrl),
                        }
                    })
                    .filter(|key| {
                        // only check keys that correspond to an action
                        hotkeys.keys_to_actions.contains(key.clone())
                    });

                match (was_pressed, is_pressed) {
                    // Just pressed
                    (false, true) => possible_keys.for_each(|key| self.push_key(key)),
                    // OS key repeat
                    (true, true) => (),
                    // Just released
                    (_, false) => possible_keys.for_each(|key| self.pop_key(key)),
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                let alt_changed = self.alt != m.alt();
                let ctrl_changed = self.ctrl != m.ctrl();
                let shift_changed = self.shift != m.shift();
                self.alt ^= alt_changed;
                self.ctrl ^= ctrl_changed;
                self.shift ^= shift_changed;
                // For every held key, re-evaluate their meaning w.r.t new
                // modifiers.
                // Holy moly that sounds like a lot of work -w-;;
                todo!()
            }
            _ => (),
        }
    }
    /// A hotkey was detected, apply it. Will go through and shadow any
    /// hotkeys this one overrides, and potentially shadow this hotkey
    /// immediately if it's shadowed by an existing key.
    fn push_key(&mut self, new: super::hotkeys::KeyboardHotkey) {
        let mut shadows_on_new = 0;
        for (old_key, shadows) in self.current_hotkeys.iter_mut() {
            if new.shadows(old_key) {
                if *shadows == 0 {
                    // <emit shadow>
                }
                *shadows += 1;
            } else {
                // Todo - does the lack of asymmetry break this logic?
                if old_key.shadows(&new) {
                    shadows_on_new += 1;
                }
            }
        }
        self.current_hotkeys.insert(new, shadows_on_new);
        
        // <emit press>
        if shadows_on_new != 0 {
            // <emit shadow>
        }
    }
    /// A hotkey was ended, discard it. Will go through and unshadow any
    /// hotkeys this one overrode, provided they are not shadowed by another.
    fn pop_key(&mut self, remove: super::hotkeys::KeyboardHotkey) {
        self.current_hotkeys.remove(&remove);
        // <emit release>

        for (old_key, shadows) in self.current_hotkeys.iter_mut() {
            if remove.shadows(old_key) {
                *shadows = shadows.checked_sub(1).unwrap_or_else(|| {
                    // Not confident that this isn't possible - and if it happens all hotkeys will
                    // be in an unknown state.
                    // Nothing will be outright broken, but actions may stop making sense until all
                    // keys are released....
                    log::warn!("{old_key:?} unshadowed too many times while removing {remove:?}!");
                    0
                });
                if *shadows == 0 {
                    // <emit unshadow>
                }
            }
        }
    }
}

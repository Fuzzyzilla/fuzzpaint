use super::hotkeys::HotkeyShadow;

pub struct WinitKeyboardActionCollector {
    /// Maps keys to the number of times they are shadowed.
    current_hotkeys: std::collections::HashMap<super::hotkeys::KeyboardHotkey, usize>,
    currently_pressed: std::collections::HashSet<winit::event::VirtualKeyCode>,
    ctrl: bool,
    shift: bool,
    alt: bool,

    sender: super::ActionSender,
}
impl WinitKeyboardActionCollector {
    pub fn new(sender: super::ActionSender) -> Self {
        Self {
            ctrl: false,
            alt: false,
            shift: false,
            current_hotkeys: Default::default(),
            currently_pressed: Default::default(),

            sender,
        }
    }
    pub fn push_event<'a>(&mut self, event: &winit::event::WindowEvent) {
        let hotkeys = crate::GlobalHotkeys::get();

        use winit::event::WindowEvent;
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                let Some(code) = input.virtual_keycode else {
                    return;
                };

                let was_pressed = self.currently_pressed.contains(&code);
                let is_pressed = winit::event::ElementState::Pressed == input.state;

                // Update currently_pressed set accordingly:
                if is_pressed && !was_pressed {
                    self.currently_pressed.insert(code);
                } else if !is_pressed {
                    self.currently_pressed.remove(&code);
                }

                // Depending on the status of ctrl, shift, and alt, this key
                // event could correspond to eight different actions. Check
                // them all!

                // Copy so that the iter does not borrow self.
                let ctrl = self.ctrl;
                let shift = self.shift;
                let alt = self.alt;
                let possible_keys = (0u8..(1 << (ctrl as u8 + shift as u8 + alt as u8)))
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
                            key: code,
                            alt: consume(alt),
                            shift: consume(shift),
                            ctrl: consume(ctrl),
                        }
                    })
                    .filter_map(|key| {
                        // find the action of each key, or skip if none.
                        Some((hotkeys.keys_to_actions.action_of(key.clone())?, key))
                    });

                match (was_pressed, is_pressed) {
                    // Just pressed
                    (false, true) => {
                        possible_keys.for_each(|(action, key)| self.push_key(action, key))
                    }
                    // OS key repeat
                    (true, true) => possible_keys.for_each(|(action, _)| {
                        // No bookkeeping to do, just emit directly
                        self.sender.repeat(action);
                    }),
                    // Just released
                    (_, false) => possible_keys.for_each(|(action, key)| self.pop_key(action, key)),
                }

                // Shouldn't need to happen but it's not working and i'm getting tired of debugging TwT
                self.cull();
            }
            WindowEvent::ModifiersChanged(m) => {
                self.alt = m.alt();
                self.ctrl = m.ctrl();
                self.shift = m.shift();
                // Original plan:
                // For every held key, re-evaluate their meaning w.r.t new
                // modifiers.
                // Holy moly that sounds like a lot of work -w-;;

                // However, upon testing, it feels great with no logic
                // in here. I'll work on plumbing this logic in with the
                // rest of the app, and I'll revisit this logic if the need
                // arises!

                // Clear any hotkeys that stopped due to any modifiers releasing.
                self.cull();
            }
            _ => (),
        }
    }
    /// Release any events that have stopped being relavent.
    fn cull(&mut self) {
        let mut to_remove = Vec::<super::hotkeys::KeyboardHotkey>::new();

        for (hotkey, _) in self.current_hotkeys.iter() {
            let no_longer_applies = (hotkey.alt && !self.alt)
                || (hotkey.shift && !self.shift)
                || (hotkey.ctrl && !self.ctrl)
                || !self.currently_pressed.contains(&hotkey.key);

            if no_longer_applies {
                to_remove.push(hotkey.clone())
            }
        }

        let hotkeys = crate::GlobalHotkeys::get();
        for hotkey in to_remove.into_iter() {
            if let Some(action) = hotkeys.keys_to_actions.action_of(hotkey.clone()) {
                self.pop_key(action, hotkey);
            }
        }
    }
    /// A hotkey was detected, apply it. Will go through and shadow any
    /// hotkeys this one overrides, and potentially shadow this hotkey
    /// immediately if it's shadowed by an existing key.
    fn push_key(&mut self, action: super::Action, new: super::hotkeys::KeyboardHotkey) {
        // Already pressed, skip to avoid breaking shadow counters
        if self.current_hotkeys.contains_key(&new) {
            return;
        }

        let hotkeys = crate::GlobalHotkeys::get();

        let mut shadows_on_new = 0;
        for (old_key, shadows) in self.current_hotkeys.iter_mut() {
            if new.shadows(old_key) {
                if *shadows == 0 {
                    if let Some(old_action) = hotkeys.keys_to_actions.action_of(old_key.clone()) {
                        self.sender.shadow(old_action);
                    }
                }
                *shadows += 1;
            } else {
                // Todo - does the lack of asymmetry break this logic?
                if old_key.shadows(&new) {
                    shadows_on_new += 1;
                }
            }
        }
        self.sender.press(action);
        if shadows_on_new != 0 {
            self.sender.shadow(action);
        }

        self.current_hotkeys.insert(new.clone(), shadows_on_new);
    }
    /// A hotkey was ended, discard it. Will go through and unshadow any
    /// hotkeys this one overrode, provided they are not shadowed by another.
    fn pop_key(&mut self, action: super::Action, remove: super::hotkeys::KeyboardHotkey) {
        // Early return if the hotkey wasn't previously detected as pressed,
        // to avoid committing chaos to the shadow counters.
        if self.current_hotkeys.remove(&remove).is_none() {
            return;
        };
        self.sender.release(action);

        let hotkeys = crate::GlobalHotkeys::get();
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
                    if let Some(old_action) = hotkeys.keys_to_actions.action_of(old_key.clone()) {
                        self.sender.unshadow(old_action);
                    }
                }
            }
        }
    }
}

pub struct Settings {
    // LoadBlockError is !Clone (and can't be, oopsie) so use a string.
    hotkeys_error: Option<String>,
    hotkeys: crate::actions::hotkeys::ActionsToKeys,
    /// When adding a new hotkey, remember exactly where we're adding it.
    new_hotkey: Option<NewHotkeyState>,
    pane: Pane,
}
impl Default for Settings {
    fn default() -> Self {
        let hotkeys = crate::global::hotkeys::Hotkeys::read();
        Self {
            hotkeys_error: hotkeys.load_blocker().map(ToString::to_string),
            hotkeys: hotkeys.actions_to_keys.clone(),
            new_hotkey: None,
            pane: Pane::default(),
        }
    }
}

impl Settings {
    /// Request the settings to reload from disk, and re-sync UI state with it.
    fn hard_reload(&mut self) {
        let mut write = crate::global::hotkeys::Hotkeys::write();
        *write = crate::global::hotkeys::Hotkeys::from_default_file();
        self.hotkeys_error = write.load_blocker().map(ToString::to_string);
        self.hotkeys = write.actions_to_keys.clone();
    }
    /// Save the settings to disk, *regardless of internal error state*. Ensure the user is Ok with this.
    fn save(&mut self) {
        let try_save = || -> Result<(), String> {
            let new_hotkeys = crate::global::hotkeys::Hotkeys::try_from(self.hotkeys.clone())
                .map_err(|e| e.to_string())?;
            let mut write = crate::global::hotkeys::Hotkeys::write();
            *write = new_hotkeys;
            write.save().map_err(|e| e.to_string())
        };

        if let Err(e) = try_save() {
            self.hotkeys_error = Some(e);
        }
    }
    fn hotkey_ui(
        &mut self,
        ui: &mut egui::Ui,
    ) -> super::modal::Response<(), (), std::convert::Infallible> {
        // Show an error banner.
        if let Some(error) = self.hotkeys_error.clone() {
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::Min).with_main_justify(true).with_main_wrap(true),
                |ui| {
                    ui.label(
                        egui::RichText::new("An error occured reading the settings file. Defaults have been used. To prevent data loss, the file will not be overwritten.")
                            .color(ui.style().visuals.error_fg_color),
                    );
                    ui.end_row();
                    // Use monospace as toml errors use ascii art
                    ui.label(egui::RichText::new(error).monospace());
                    ui.end_row();
                    if ui.button("Retry").on_hover_text("Try loading the file again.").clicked() {
                        self.hard_reload();
                    }
                    if ui.button("⚠ Allow overwrite").on_hover_text("Ignore the error, allowing replacing the erroneous file with new values. Existing settings data will be lost!").clicked() {
                        // Just clear the error. This removes the banner and signifies that the file is writable again.
                        self.hotkeys_error = None;
                    }
                },
            );
            ui.separator();
        }
        // Show the main hotkey edit area!
        egui::ScrollArea::vertical()
            // Something is hecked, the scroll area explodes to infinity if not explicitly limited.
            .max_height(400.0)
            .show(ui, |ui| {
                egui::Grid::new("hotkeys")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        for action in <crate::actions::Action as strum::IntoEnumIterator>::iter() {
                            // First column, with the name of the action being assigned.
                            ui.label(action.as_ref());
                            // Second column, with a bunch of buttons for changing existing binds and adding new ones
                            ui.with_layout(
                                egui::Layout::top_down_justified(egui::Align::Min),
                                |ui| {
                                    // Show a number of buttons, to allow each existing hotkey to be modified.
                                    let hotkeys = self.hotkeys.0.get_mut(&action);
                                    let after_end_idx = hotkeys
                                        .as_ref()
                                        .map_or(0, |hotkeys| hotkeys.keyboard.len());

                                    // There exist some hotkeys already, modify em!
                                    if let Some(hotkeys) = hotkeys {
                                        for (index, key) in hotkeys.keyboard.iter_mut().enumerate()
                                        {
                                            // This is the hotkey we're actively changing!
                                            if self.new_hotkey
                                                == Some(NewHotkeyState { action, index })
                                            {
                                                match clicked_hotkey(ui) {
                                                    ClickedHotkeyResponse::None => (),
                                                    ClickedHotkeyResponse::Cancel => {
                                                        self.new_hotkey = None;
                                                    }
                                                    ClickedHotkeyResponse::Finished(new_key) => {
                                                        self.new_hotkey = None;
                                                        // Re-assign the existing key.
                                                        *key = new_key;
                                                    }
                                                }
                                            } else {
                                                // Not being changed, show as normal.
                                                if ui.button(key.to_string()).clicked() {
                                                    // Clicked the button, start changin'!
                                                    self.new_hotkey =
                                                        Some(NewHotkeyState { action, index });
                                                }
                                            }
                                        }
                                    }

                                    // Add more hotkeys, at the bottom.
                                    if self.new_hotkey
                                        == Some(NewHotkeyState {
                                            action,
                                            index: after_end_idx,
                                        })
                                    {
                                        // We're in the process of adding one, show UI for that...
                                        match clicked_hotkey(ui) {
                                            ClickedHotkeyResponse::None => (),
                                            ClickedHotkeyResponse::Cancel => {
                                                self.new_hotkey = None;
                                            }
                                            ClickedHotkeyResponse::Finished(key) => {
                                                self.new_hotkey = None;
                                                // Insert the new key into the collection, making the collection in the process if need be.
                                                let keys =
                                                    self.hotkeys.0.entry(action).or_default();
                                                keys.keyboard.push(key);
                                            }
                                        }
                                    } else {
                                        // Show button to initiate new addition.
                                        if ui.button(super::PLUS_ICON.to_string()).clicked() {
                                            // Start adding a new hotkey at the end.
                                            self.new_hotkey = Some(NewHotkeyState {
                                                action,
                                                index: after_end_idx,
                                            });
                                        };
                                    }
                                    // Add an extra item worth of space, for hrule.
                                    ui.add_space(0.0);
                                    // Hack: Egui doesn't show lines between grid cells - this is genuinely a readability issue, especially
                                    // given that the striping doesn't work inexplicably. Do it ourselves :P
                                    // Just draw a line across the entire drawing area, at the y level of current cursor.
                                    let painter = ui.painter();
                                    painter.hline(
                                        ui.clip_rect().x_range(),
                                        ui.next_widget_position().y,
                                        ui.style().visuals.window_stroke,
                                    );
                                },
                            );
                            ui.end_row();
                        }
                    });
            });

        // show the file path for custom editting/syntax error fixing.
        if let Some(path) = crate::global::hotkeys::Hotkeys::default_file_location() {
            ui.label(egui::RichText::new(path.to_string_lossy()).weak());
        }

        // Ok and cancel buttons at the bottom of the window
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.hotkeys_error.is_none(), egui::Button::new("Ok"))
                .on_disabled_hover_text("Cannot write settings while an error is present.")
                .clicked()
            {
                // No error, safe to save!
                self.save();
                return super::modal::Response::Confirm(());
            }
            if ui.button("Close").clicked() {
                return super::modal::Response::Cancel(());
            }
            super::modal::Response::Continue
        })
        .inner
    }
}

impl super::Modal for Settings {
    const NAME: &'static str = "Settings";
    type Cancel = ();
    type Confirm = ();
    type Error = std::convert::Infallible;
    fn do_ui(
        &mut self,
        ui: &mut egui::Ui,
    ) -> super::modal::Response<Self::Cancel, Self::Confirm, Self::Error> {
        match self.pane {
            Pane::Hotkeys => self.hotkey_ui(ui),
        }
    }
}

#[derive(PartialEq, Eq)]
struct NewHotkeyState {
    action: crate::actions::Action,
    index: usize,
}

#[derive(Clone, Copy, Default)]
enum Pane {
    #[default]
    Hotkeys,
}

fn egui_key_to_winit_key(key: egui::Key) -> winit::keyboard::KeyCode {
    use egui::Key as EKey;
    use winit::keyboard::KeyCode as WKey;
    // Adapted from egui_winit, nightmare match statement!
    match key {
        EKey::Tab => WKey::Tab,
        EKey::ArrowDown => WKey::ArrowDown,
        EKey::ArrowLeft => WKey::ArrowLeft,
        EKey::ArrowRight => WKey::ArrowRight,
        EKey::ArrowUp => WKey::ArrowUp,
        EKey::End => WKey::End,
        EKey::Home => WKey::Home,
        EKey::PageDown => WKey::PageDown,
        EKey::PageUp => WKey::PageUp,
        EKey::Backspace => WKey::Backspace,
        EKey::Delete => WKey::Delete,
        EKey::Insert => WKey::Insert,
        EKey::Escape => WKey::Escape,
        EKey::Cut => WKey::Cut,
        EKey::Copy => WKey::Copy,
        EKey::Paste => WKey::Paste,
        EKey::Space => WKey::Space,
        EKey::Enter => WKey::Enter,
        EKey::Comma => WKey::Comma,
        EKey::Period => WKey::Period,

        EKey::Colon | EKey::Semicolon => WKey::Semicolon,
        EKey::Pipe | EKey::Backslash => WKey::Backslash,
        EKey::Questionmark | EKey::Slash => WKey::Slash,

        EKey::OpenBracket => WKey::BracketLeft,
        EKey::CloseBracket => WKey::BracketRight,
        EKey::Backtick => WKey::Backquote,
        EKey::Minus => WKey::Minus,
        EKey::Plus => WKey::NumpadAdd,
        EKey::Equals => WKey::Equal,
        EKey::Num0 => WKey::Digit0,
        EKey::Num1 => WKey::Digit1,
        EKey::Num2 => WKey::Digit2,
        EKey::Num3 => WKey::Digit3,
        EKey::Num4 => WKey::Digit4,
        EKey::Num5 => WKey::Digit5,
        EKey::Num6 => WKey::Digit6,
        EKey::Num7 => WKey::Digit7,
        EKey::Num8 => WKey::Digit8,
        EKey::Num9 => WKey::Digit9,
        EKey::A => WKey::KeyA,
        EKey::B => WKey::KeyB,
        EKey::C => WKey::KeyC,
        EKey::D => WKey::KeyD,
        EKey::E => WKey::KeyE,
        EKey::F => WKey::KeyF,
        EKey::G => WKey::KeyG,
        EKey::H => WKey::KeyH,
        EKey::I => WKey::KeyI,
        EKey::J => WKey::KeyJ,
        EKey::K => WKey::KeyK,
        EKey::L => WKey::KeyL,
        EKey::M => WKey::KeyM,
        EKey::N => WKey::KeyN,
        EKey::O => WKey::KeyO,
        EKey::P => WKey::KeyP,
        EKey::Q => WKey::KeyQ,
        EKey::R => WKey::KeyR,
        EKey::S => WKey::KeyS,
        EKey::T => WKey::KeyT,
        EKey::U => WKey::KeyU,
        EKey::V => WKey::KeyV,
        EKey::W => WKey::KeyW,
        EKey::X => WKey::KeyX,
        EKey::Y => WKey::KeyY,
        EKey::Z => WKey::KeyZ,
        EKey::F1 => WKey::F1,
        EKey::F2 => WKey::F2,
        EKey::F3 => WKey::F3,
        EKey::F4 => WKey::F4,
        EKey::F5 => WKey::F5,
        EKey::F6 => WKey::F6,
        EKey::F7 => WKey::F7,
        EKey::F8 => WKey::F8,
        EKey::F9 => WKey::F9,
        EKey::F10 => WKey::F10,
        EKey::F11 => WKey::F11,
        EKey::F12 => WKey::F12,
        EKey::F13 => WKey::F13,
        EKey::F14 => WKey::F14,
        EKey::F15 => WKey::F15,
        EKey::F16 => WKey::F16,
        EKey::F17 => WKey::F17,
        EKey::F18 => WKey::F18,
        EKey::F19 => WKey::F19,
        EKey::F20 => WKey::F20,
    }
}

enum ClickedHotkeyResponse {
    None,
    Cancel,
    Finished(crate::actions::hotkeys::KeyboardHotkey),
}

fn clicked_hotkey(ui: &mut egui::Ui) -> ClickedHotkeyResponse {
    let response = ui.input(|input| {
        // If escape was pressed, exit.
        if input.key_pressed(egui::Key::Escape) {
            ClickedHotkeyResponse::Cancel
        } else if let Some((&key, &modifiers)) = input.events.iter().find_map(|event| {
            // Find the first key press and it's modifiers this frame.
            if let egui::Event::Key {
                pressed: true,
                key,
                modifiers,
                ..
            } = event
            {
                Some((key, modifiers))
            } else {
                None
            }
        }) {
            // Return it as a hotkey!
            let key = egui_key_to_winit_key(key);
            ClickedHotkeyResponse::Finished(crate::actions::hotkeys::KeyboardHotkey {
                alt: modifiers.alt,
                ctrl: modifiers.ctrl,
                shift: modifiers.shift,
                key,
            })
        } else {
            // No clicks, continue to listen.
            ClickedHotkeyResponse::None
        }
    });

    if matches!(&response, ClickedHotkeyResponse::None) {
        // Show a button to indicate listening. If clicked, stop the operation.
        if ui.button("press a hotkey...").clicked() {
            return ClickedHotkeyResponse::Cancel;
        };
    }
    response
}

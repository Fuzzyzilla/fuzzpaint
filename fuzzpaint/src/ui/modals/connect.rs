const DISCLAIMER: &str =
    "Warning: Neither the identity of the server nor the users on that server \
are verified. Communication with the server is not encrypted.

The following may be shared, unencrypted:
• Your IP address and port.
• The server's address and port.
• Your user details including name, profile image, etc.
• Any action you or other users perform on the canvas, including typed text, \
drawings, etc.
• Any resources you download, upload, or use, explicitly or implicitly by the \
software, including images, fonts, documents, etc.
This is an incomplete list provided only as an example.

No guarantees or warranty is made, express or implied. This isn't a contract, \
agreement, nor a EULA, I literally just thought I'd let you know. Please don't \
sue me.";

#[derive(Default)]
enum State {
    #[default]
    Input,
    Waiting(crate::connections::NewConnectionStatus),
}
#[derive(Default)]
pub struct Modal {
    // Validation of this locally (within the modal) seems extremely complex,
    // considering it could be a DNS lookup, so we don't bother.
    address_text: String,
    state: State,
    agree: bool,
}
impl super::Modal for Modal {
    fn do_ui(
        &mut self,
        _id: egui::Id,
        ui: &mut egui::Ui,
        _state: &mut crate::ui::MainUI,
        interface: &mut crate::ui::interface::Interface,
    ) -> super::Response {
        let enable_input = matches!(&self.state, State::Input);
        ui.scope(|ui| {
            // Scope so we can logically disable the whole panel while
            // attempting to connect.
            if !enable_input {
                ui.disable();
            }
            super::title(ui, "Connect");
            ui.label(
                "Enter the address and port of the remote server. The port \
                will need to be forwarded manually by the remote.",
            );
            let response = egui::text_edit::TextEdit::singleline(&mut self.address_text)
                .hint_text("example.com:1234, 127.0.0.1:1234, [::1]:1234...")
                .show(ui)
                .response;

            let enter_pressed_on_text =
                response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));

            ui.label(DISCLAIMER);
            ui.label(egui::RichText::new("Connect at your own risk!").strong());
            let response = ui.checkbox(&mut self.agree, "I understand.");

            // Enter pressed and not agreed, highlight agree checkbox.
            if enter_pressed_on_text && !self.agree {
                response.request_focus();
            }
        });
        ui.horizontal_centered(|ui| {
            let connect = ui
                // Only enable if agreed, not empty, and not actively waiting
                // for it to connect.
                .add_enabled(
                    self.agree && enable_input && !self.address_text.is_empty(),
                    egui::Button::new("Connect"),
                )
                .on_disabled_hover_text("Please read the disclaimer!")
                .clicked();

            if ui
                .add_enabled(enable_input, egui::Button::new("Cancel"))
                .clicked()
            {
                return super::Response::Close;
            }

            if connect {
                self.state = State::Waiting(interface.connect(self.address_text.clone()));
            }

            if let State::Waiting(waiting) = &self.state {
                ui.add(egui::Spinner::new());
                // Finished?
                if !waiting.is_pending() {
                    // Try again!
                    if waiting.has_failed() {
                        self.state = State::Input;
                    } else {
                        // Succeeded, close.
                        return super::Response::Close;
                    }
                }
            }
            super::Response::Retain
        })
        .inner
    }
}

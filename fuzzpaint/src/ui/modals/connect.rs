const DISCLAIMER: &str = "\
Warning: Neither the identity nor authenticity of the server, the users \
on that server, nor the contents of resources retrieved from that server are \
verified. Encryption is performed on a best-effort basis and the security \
thereof is not guaranteed.

The following will be shared to the network, and may not be secured:
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
    Waiting(crate::connections::NewConnectionStatusReciever),
    CompareHash(String),
}
#[derive(Default)]
pub struct Modal {
    // Validation of this locally (within the modal) seems extremely complex,
    // considering it could be a DNS lookup, so we don't bother.
    address_text: String,
    last_err: Option<String>,
    state: State,
    agree: bool,
}
impl Modal {
    fn input_ui(
        &mut self,
        ui: &mut egui::Ui,
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
            let text_response = ui
                .horizontal(|ui| {
                    let text_response =
                        egui::text_edit::TextEdit::singleline(&mut self.address_text)
                            .hint_text("example.com:1234, 127.0.0.1:1234, [::1]:1234...")
                            .show(ui)
                            .response;
                    if let Some(err) = self.last_err.as_deref() {
                        ui.separator();
                        ui.label(egui::RichText::new(err).strong());
                    }
                    text_response
                })
                .inner;

            let enter_pressed_on_text =
                text_response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));

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
                self.last_err = None;
            }

            if let State::Waiting(waiting) = std::mem::take(&mut self.state) {
                ui.add(egui::Spinner::new());
                match waiting.poll() {
                    Ok(Ok(success)) => {
                        self.state = State::CompareHash(success.session_id);
                    }
                    Ok(Err(failed)) => {
                        self.state = State::Input;
                        self.last_err = Some(failed.to_string());
                    }
                    Err(not_ready) => {
                        self.state = State::Waiting(not_ready);
                    }
                }
            }
            super::Response::Retain
        })
        .inner
    }
    fn compare_hash_ui(
        &mut self,
        ui: &mut egui::Ui,
        _interface: &mut crate::ui::interface::Interface,
    ) -> super::Response {
        super::title(ui, "Verify");
        // Gaurded externally
        let State::CompareHash(hash) = &self.state else {
            unreachable!();
        };
        let hash = hash.clone();

        ui.label(
            egui::RichText::new("Compare the code below with the host of the server.").strong(),
        );
        ui.label(
            "If the code differs, the integrity and confidentiality of the \
        the connection may be broken and should not be trusted.",
        );
        ui.label(
            "As there is no central authority, the identity of the server \
        cannot be verified automatically. Use an external, trusted \
        communication channel to compare this with the code the server \
        recieved when you joined. You must do this every time you connect.",
        );
        ui.label(
            egui::RichText::new("Do not accept codes communicated through Fuzzpaint.").strong(),
        );
        ui.label("This code is not sensitive information and may be shared publicly.");
        ui.separator();
        ui.vertical_centered(|ui| {
            let len_4 = hash.len() / 4;
            #[allow(clippy::erasing_op)]
            #[allow(clippy::identity_op)]
            for section in [
                &hash[0 * len_4..1 * len_4],
                &hash[1 * len_4..2 * len_4],
                &hash[2 * len_4..3 * len_4],
                // Might not be divisible by four, so take the rest.
                &hash[3 * len_4..],
            ] {
                let response = ui
                    .label(egui::RichText::new(section).monospace())
                    .on_hover_text("Click to copy");
                if response.clicked() {
                    ui.ctx().copy_text(hash.clone());
                    // FIXME: A popup would be nicer.
                    log::info!("Verification code copied to clipboard.");
                }
            }
            ui.separator();
            ui.horizontal(|ui| {
                let response = if ui.button("Looks good!").clicked() {
                    super::Response::Close
                } else {
                    super::Response::Retain
                };
                if ui.button("That's not right...").clicked() {
                    // Disconnect.
                    // FIXME
                    log::error!("unimplimented! you gotta disconnect manually! Sorry!!");
                    self.state = State::Input;
                }
                response
            })
            .inner
        })
        .inner
    }
}
impl super::Modal for Modal {
    fn do_ui(
        &mut self,
        _id: egui::Id,
        ui: &mut egui::Ui,
        _state: &mut crate::ui::MainUI,
        interface: &mut crate::ui::interface::Interface,
    ) -> super::Response {
        match &self.state {
            State::Input | State::Waiting(_) => self.input_ui(ui, interface),
            State::CompareHash(_) => self.compare_hash_ui(ui, interface),
        }
    }
    fn close_requested(
        &mut self,
        _id: egui::Id,
        _ctx: &egui::Context,
        _state: &mut crate::ui::MainUI,
        _interface: &mut crate::ui::interface::Interface,
    ) -> super::Response {
        match &self.state {
            // Simple cancel
            State::Input => super::Response::Close,
            // We need a yes or no, not a "cancel". Reject the request.
            State::Waiting(_) | State::CompareHash(_) => super::Response::Retain,
        }
    }
}

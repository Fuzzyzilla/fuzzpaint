#[derive(Default)]
pub struct Modal;
impl super::Modal for Modal {
    fn do_ui(
        &mut self,
        _id: egui::Id,
        ui: &mut egui::Ui,
        state: &mut crate::ui::MainUI,
        _interface: &mut crate::ui::interface::Interface,
    ) -> super::Response {
        ui.label("There are unsaved documents. Do you really want to exit?");
        ui.horizontal(|ui| {
            // On first run-thru it would be nice for this cancel button to auto-focus itself.
            if ui.button("Cancel").highlight().clicked() {
                return super::Response::Close;
            }
            if ui.button("Exit").clicked() {
                state.window_action = Some(crate::ui::WindowAction::Close);
                return super::Response::Close;
            }
            super::Response::Retain
        })
        .inner
    }
    fn close(
        &mut self,
        _: egui::Id,
        _: &egui::Context,
        state: &mut crate::ui::MainUI,
        _: &mut crate::ui::interface::Interface,
    ) {
        // Tell the main UI the close modal is gone.
        state.app_close_modal_shown = false;
    }
}

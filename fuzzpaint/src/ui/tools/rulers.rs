pub struct Rulers {
    unit: fuzzpaint_types::dpi::Unit,
}
impl Rulers {
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        view_transform: &fuzzpaint_types::similarity::Similarity,
        rect: &fuzzpaint_types::dpi::Rect,
        dpi: fuzzpaint_types::dpi::Dpi,
    ) {
        let ui_rect = ui.max_rect();
    }
}

#[derive(Default)]
pub struct Snapper {}
impl Snapper {
    pub fn snap(&self, point: egui::Pos2) -> egui::Pos2 {
        point
    }
}

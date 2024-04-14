use fuzzpaint_core::brush::{Brush, Texture, UniqueID};

pub struct CreationOutput {
    pub texture_data: Option<Vec<u8>>,
    pub brush: Brush,
}
pub struct CreationModal {
    // Imported texture + handle, egui tex frees on drop!
    texture_data: Option<Vec<u8>>,
    texture: Option<egui::TextureHandle>,
}
impl Default for CreationModal {
    fn default() -> Self {
        let () = ();
        Self {
            texture_data: None,
            texture: None,
        }
    }
}
impl super::Modal for CreationModal {
    type Cancel = ();
    type Confirm = CreationOutput;
    type Error = ();
    const NAME: &'static str = "Create Brush";
    fn do_ui(
        &mut self,
        ui: &mut egui::Ui,
    ) -> super::modal::Response<Self::Cancel, Self::Confirm, Self::Error> {
        ui.label("OwO");
        if ui.button("Leave me be, foul beeste.").clicked() {
            super::modal::Response::Cancel(())
        } else {
            super::modal::Response::Continue
        }
    }
}

pub struct Preloaded {
    id: UniqueID,
    texture: egui::TextureHandle,
    vertices: egui::Mesh,
}

/// Provides a brush selection drawer with many brushes loaded dynamically.
pub struct Bin {}

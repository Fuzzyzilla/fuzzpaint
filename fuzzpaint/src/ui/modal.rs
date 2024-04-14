pub enum Response<Cancel, Confirm, Error> {
    Cancel(Cancel),
    Confirm(Confirm),
    Error(Error),
    Continue,
}
pub trait Modal {
    type Cancel;
    type Confirm;
    type Error;
    const NAME: &'static str;
    fn do_ui(&mut self, ui: &mut egui::Ui) -> Response<Self::Cancel, Self::Confirm, Self::Error>;
}

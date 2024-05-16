pub enum Response<Cancel, Confirm, Error> {
    Cancel(Cancel),
    Confirm(Confirm),
    Error(Error),
    Continue,
}
impl<Cancel, Confirm, Error> Response<Cancel, Confirm, Error> {
    /// Returns true if the response is not [`Response::Continue`]
    pub fn closed(&self) -> bool {
        !matches!(self, Self::Continue)
    }
}
pub trait Modal {
    type Cancel;
    type Confirm;
    type Error;
    const NAME: &'static str;
    fn do_ui(&mut self, ui: &mut egui::Ui) -> Response<Self::Cancel, Self::Confirm, Self::Error>;
}

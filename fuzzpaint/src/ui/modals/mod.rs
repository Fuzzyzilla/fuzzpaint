pub mod connect;
pub mod exit;
pub mod new_doc;
pub mod settings;

fn title(ui: &mut egui::Ui, title: &str) {
    ui.heading(title);
    ui.separator();
}

static MODAL_CONTAINER: std::sync::Mutex<ModalContainer> = std::sync::Mutex::new(ModalContainer {
    last_used_id: egui::Id::NULL,
    modals: Vec::new(),
});

/// Create a modal. One of the modals created this frame will be on top.
pub fn spawn(modal: impl Modal + 'static) {
    let modal = Box::new(modal);
    ModalContainer::with_mut(move |this| {
        // Advance the modal id by something "random enough." Addr does not
        // carry provenance, this is what we want. We just want the randomish
        // numeric address (even if it's not random, this works.)

        // We can't re-use IDs because the sizing pass only gets run once, so
        // modals end up inheriting each other's sizes and it looks real bad.
        // FIXME: does this cause a memory leak within egui?
        this.last_used_id = this.last_used_id.with(std::ptr::from_ref(&modal).addr());
        this.modals.push(DynModal {
            id: this.last_used_id,
            inner: modal,
        });
    });
}
/// Show the modals.
// When the clippy yaps at me for something that's SEMANTICALLY CORRECT >:O
#[allow(clippy::semicolon_if_nothing_returned)]
pub fn show(
    state: &mut super::MainUI,
    ui: &mut egui::Ui,
    interface: &mut super::interface::Interface,
) {
    ModalContainer::with_mut(|this| this.show_inner(state, ui, interface))
}

/// Maintains and polls all modals. A singleton instance of this is kept
/// internally, use one of the associated functions to interact.
// The reason this is put in a static mutex instead of living inside of
// `super::MainUI` is so that modals can accept the `MainUI` as a `&mut`
// parameter. If this container lived in there, then borrow issues abound.
struct ModalContainer {
    last_used_id: egui::Id,
    modals: Vec<DynModal>,
}
impl ModalContainer {
    pub fn with_mut<R>(f: impl FnOnce(&mut ModalContainer) -> R) -> R {
        f(&mut MODAL_CONTAINER.lock().unwrap())
    }
    pub fn add(modal: impl Modal + 'static) {
        let modal = Box::new(modal);
        Self::with_mut(move |this| {
            // Advance the modal id by something "random enough." Addr does not
            // carry provenance, this is what we want. We just want the
            // randomish numeric address (even if it's not random, this works.)

            // We can't re-use IDs because the sizing pass only gets run once,
            // so modals end up inheriting each other's sizes and it looks real
            // bad. FIXME: does this cause a memory leak within egui?
            this.last_used_id = this.last_used_id.with(std::ptr::from_ref(&modal).addr());
            this.modals.push(DynModal {
                id: this.last_used_id,
                inner: modal,
            });
        })
    }
    /// Show the modals.
    pub fn show(
        state: &mut super::MainUI,
        ui: &mut egui::Ui,
        interface: &mut super::interface::Interface,
    ) {
        Self::with_mut(|this| this.show_inner(state, ui, interface))
    }
    fn show_inner(
        &mut self,
        state: &mut super::MainUI,
        ui: &mut egui::Ui,
        interface: &mut super::interface::Interface,
    ) {
        let mut close_indices = smallvec::SmallVec::<[usize; 1]>::new();
        for (i, modal) in self.modals.iter_mut().enumerate() {
            let response = egui::Modal::new(modal.id)
                .show(ui, |ui| modal.inner.do_ui(modal.id, ui, state, interface));

            let mut manually_requested_close = response.inner == Response::Close;
            // Only if externally closed (esc or clicked outside) we ask the
            // modal if it's allowed.
            if !manually_requested_close && response.should_close() {
                manually_requested_close =
                    modal.inner.close_requested(modal.id, ui, state, interface) == Response::Close;
            }
            // Close only if the modal requested it or agreed to the external
            // request.
            if manually_requested_close {
                close_indices.push(i);
            }
        }
        let mut i = 0;
        // Notify and remove all the modals that requested a close
        self.modals.retain_mut(|modal| {
            let i = {
                let temp = i;
                i += 1;
                temp
            };
            if !close_indices.contains(&i) {
                // not closed, do retain
                return true;
            }
            modal.inner.close(modal.id, ui, state, interface);
            // Delete.
            false
        });
    }
}
struct DynModal {
    id: egui::Id,
    inner: Box<dyn Modal>,
}

#[derive(PartialEq, Eq)]
pub enum Response {
    Retain,
    Close,
}
pub trait Modal: Send {
    /// Show the modal.
    fn do_ui(
        &mut self,
        id: egui::Id,
        ui: &mut egui::Ui,
        state: &mut super::MainUI,
        interface: &mut super::interface::Interface,
    ) -> Response;
    /// Called if the modal is clicked outside of, or if escape is pressed.
    ///
    /// Default implimentation unconditionally accepts the close request.
    #[allow(unused_variables)]
    fn close_requested(
        &mut self,
        id: egui::Id,
        ui: &mut egui::Ui,
        state: &mut super::MainUI,
        interface: &mut super::interface::Interface,
    ) -> Response {
        Response::Close
    }
    /// The modal is closing, because one of the other two methods returned
    /// [`Response::Close`]
    ///
    /// Default implimentation does nothing.
    #[allow(unused_variables)]
    fn close(
        &mut self,
        id: egui::Id,
        ui: &mut egui::Ui,
        state: &mut super::MainUI,
        interface: &mut super::interface::Interface,
    ) {
    }
}

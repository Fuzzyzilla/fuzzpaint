const TRACE: char = '📈';
const DEBUG: char = '🐞';
const INFO: char = 'ℹ';
const WARN: char = '⚠';
const ERROR: char = '❌';
const CONNECTION: char = '🔌';

#[derive(Default)]
pub struct ErrorDisplay {
    records: Vec<fuzzpaint_logger::Record>,
}
impl ErrorDisplay {
    /// Display error popup messages from the crate's [logger](log).
    ///
    /// This should be called at the end of the frame, to catch potential logs
    /// made earlier in the UI stack.
    pub fn show(&mut self, ctx: &egui::Context, collector: &fuzzpaint_logger::CollectLogger) {
        self.records.extend(
            collector
                .take()
                .into_iter()
                .filter(|item| item.level <= log::Level::Info),
        );
        let max_width = ctx.available_rect().width();
        egui::Window::new("debug").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for item in &self.records {
                    ui.label(&item.text);
                }
            })
        });
    }
}

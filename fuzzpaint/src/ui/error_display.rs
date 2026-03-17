const TRACE: char = '📈';
const DEBUG: char = '🐞';
const INFO: char = 'ℹ';
const WARN: char = '⚠';
const ERROR: char = '❌';
const CONNECTION: char = '🔌';

const WINDOW_AUTO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

struct Message {
    level: log::Level,
    kind: MessageKind,
    contents: String,
}

// What is the source of the message? Different messages may display
// differently, or be individually configured.
enum MessageKind {
    /// Comes from a remote server, inherently untrustworthy~
    UntrustedRemoteLog,
    /// Comes from [`log`]
    LocalLog,
    /// A directly asserted message into the error display from other parts of
    /// the UI.
    Ui,
    /// A vulkan debugger message.
    Vulkan,
}
struct Window {
    record_idx: usize,
    id: egui::Id,
    expires: std::time::Instant,
    bottom_spacing: f32,
    hovered: bool,
}
#[derive(Default)]
pub struct ErrorDisplay {
    has_run: bool,
    records: Vec<fuzzpaint_logger::Record>,
    windows: Vec<Window>,
    show_list: bool,
    show_which_record_idx: Option<usize>,
}
impl ErrorDisplay {
    /// Enable the main list.
    pub fn show_list(&mut self) {
        self.show_list = true;
    }
    /// Display error popup messages from the crate's [logger](log).
    ///
    /// This should be called at the end of the frame, to catch potential logs
    /// made earlier in the UI stack.
    pub fn show(&mut self, ctx: &egui::Context, collector: &fuzzpaint_logger::CollectLogger) {
        // It is very important that nothing related to this logs, otherwise we
        // could end up in a death spiral!
        let len_before = self.records.len();
        self.records.extend(
            collector
                .take()
                .into_iter()
                .filter(|item| item.level <= log::Level::Info),
        );
        let len_after = self.records.len();
        if !self.has_run {
            // Skip creating new popups on the first run through,
            self.has_run = true;
            return;
        }

        let now = std::time::Instant::now();
        let new_window_expiry = now + WINDOW_AUTO_TIMEOUT;
        for i in len_before..len_after {
            self.windows.push(Window {
                record_idx: i,
                id: egui::Id::new(i),
                expires: new_window_expiry,
                bottom_spacing: 0.0,
                hovered: false,
            });
        }
        let mut remove_windows = Vec::new();
        for (i, window) in self.windows.iter_mut().enumerate() {
            let hoveredness = ctx.animate_bool(window.id, window.hovered);
            let response = egui::Window::new("")
                .id(window.id)
                .anchor(egui::Align2::RIGHT_BOTTOM, [0.0, -window.bottom_spacing])
                .title_bar(false)
                .constrain_to(ctx.available_rect())
                .collapsible(false)
                .fade_in(true)
                .default_height(0.0)
                .max_width(250.0)
                .resizable(false)
                // Cant fade in. :3
                .frame(
                    egui::Frame::window(&ctx.style())
                        .multiply_with_opacity(hoveredness * 0.5 + 0.5),
                )
                .show(ctx, |ui| {
                    // Time left / timout time
                    let time_left = window.expires - now;
                    let completion_ratio =
                        1.0 - (time_left.as_secs_f32() / WINDOW_AUTO_TIMEOUT.as_secs_f32());
                    if completion_ratio >= 1.0 {
                        remove_windows.push(i);
                    }
                    let record = &self.records[window.record_idx];

                    // Draw a progress bar by re
                    let painter = ui.painter();
                    let mut rect = ui.clip_rect();
                    rect.set_width(rect.width() * (1.0 - completion_ratio));
                    painter.rect_filled(
                        rect,
                        ui.style().visuals.window_corner_radius,
                        ui.style().visuals.window_fill.gamma_multiply_u8(127),
                    );

                    ui.horizontal_centered(|ui| {
                        ui.label(WARN.to_string());
                        ui.label(&record.text);
                    });
                })
                // Always some, since this window is not collapsible.
                .unwrap()
                .response;

            // Response.hovered is false when the text is hovered. We instead
            // want if the pointer is anywhere on the window.
            window.hovered = ctx.rect_contains_pointer(response.layer_id, response.rect);
            if window.hovered {
                window.expires = new_window_expiry;
            } else {
                // We animate while not hovered :3
                ctx.request_repaint();
            }

            // This sucks in the same way as above, but unlike that I can't
            // figure out a solution.
            if response.clicked() {
                remove_windows.push(i);
                self.show_which_record_idx = Some(window.record_idx);
            }
        }
        // Close the windows that requested removal.
        let mut i = 0;
        self.windows.retain(|_| {
            let retain = !remove_windows.contains(&i);
            i += 1;
            retain
        });

        // If clicked on a window to highlight the log, show the window.
        if self.show_which_record_idx.is_some() {
            self.show_list = true;
        }
        let max_width = ctx.available_rect().width();
        egui::Window::new("log-list")
            .open(&mut self.show_list)
            .fade_in(true)
            .fade_out(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let highlight = self.show_which_record_idx.take();
                    for (i, item) in self.records.iter().enumerate() {
                        let response = ui.label(&item.text);
                        if highlight == Some(i) {
                            // Fixme: visual highlight?
                            response.scroll_to_me(None);
                        }
                    }
                })
            });
    }
}

#![deny(unsafe_op_in_unsafe_fn)]
#![feature(portable_simd)]
#![feature(once_cell_try)]
#![feature(write_all_vectored)]
#![warn(clippy::pedantic)]
// I know it's bad but while working on a focus it's not really something I wanna be bugged about
// with a full screen of yellow lol.
#![allow(clippy::too_many_lines)]

use std::sync::Arc;
mod egui_impl;
pub mod renderer;
pub mod vulkano_prelude;
pub mod window;
use vulkano_prelude::*;
pub mod actions;
pub mod document_viewport_proxy;
pub mod gizmos;
pub mod global;
pub mod pen_tools;
pub mod picker;
pub mod render_device;
pub mod stylus_events;
pub mod text;
pub mod ui;
pub mod view_transform;

use fuzzpaint_core::id::FuzzID;

#[cfg(feature = "dhat_heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// Use jemalloc global if enabled + supported playform + not mem profiling
#[cfg(all(
    not(any(feature = "dhat_heap", target_env = "msvc")),
    feature = "jemallocator"
))]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Obviously will be user specified on a per-document basis, but for now...
const DOCUMENT_DIMENSION: u32 = 1080;
/// Premultiplied RGBA16F for interesting effects (negative + overbright colors and alpha) with
/// more than 11bit per channel precision in the \[0,1\] range.
/// Will it be user specified in the future?
const DOCUMENT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

use anyhow::Result as AnyResult;

/// FIXME! This is temp until I can see that everything is working :3
/// There still needs to be a way to intercommunicate between UI selections, Pen actions, and renderer preview.
#[derive(Clone)]
pub struct AdHocGlobals {
    pub document: fuzzpaint_core::state::document::ID,
    pub brush: fuzzpaint_core::state::StrokeBrushSettings,
    pub node: Option<fuzzpaint_core::state::graph::AnyID>,
}
impl AdHocGlobals {
    #[must_use]
    pub fn get() -> &'static parking_lot::RwLock<Option<AdHocGlobals>> {
        static ONCE: std::sync::OnceLock<parking_lot::RwLock<Option<AdHocGlobals>>> =
            std::sync::OnceLock::new();

        ONCE.get_or_init(parking_lot::RwLock::default)
    }
    #[must_use]
    pub fn read_clone() -> Option<Self> {
        Self::get().read().clone()
    }
}

async fn stylus_event_collector(
    mut event_stream: tokio::sync::broadcast::Receiver<stylus_events::StylusEventFrame>,
    ui_requests: crossbeam::channel::Receiver<ui::requests::UiRequest>,
    _: tokio::sync::mpsc::Sender<renderer::requests::RenderRequest>,
    mut action_listener: actions::ActionListener,
    mut tools: pen_tools::ToolState,
    document_preview: Arc<document_viewport_proxy::Proxy>,
) -> AnyResult<()> {
    loop {
        match event_stream.recv().await {
            Ok(stylus_frame) => {
                // We need a transform in order to do any of our work!
                let Some(transform) = document_preview.get_view_transform().await else {
                    continue;
                };

                // Get the actions, returning if stream closed.
                let action_frame = match action_listener.frame() {
                    Ok(frame) => frame,
                    Err(e) => match e {
                        actions::ListenError::Closed => return Ok(()),
                        // Todo: this is recoverable!
                        actions::ListenError::Poisoned => todo!(),
                    },
                };

                let render = tools
                    .process(&transform, stylus_frame, &action_frame, &ui_requests)
                    .await;

                if let Some(transform) = render.set_view {
                    document_preview.insert_document_transform(transform).await;
                }
                document_preview.insert_cursor(render.cursor);
                document_preview.insert_tool_render(render.render_as);
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(num)) => {
                log::warn!("Lost {num} stylus frames!");
            }
            // Stream closed, no more data to handle - we're done here!
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

fn main() -> AnyResult<()> {
    let has_term = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let default_log_level = if cfg!(debug_assertions) {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    // Log to a terminal, if available. Else, log to "log.out" in the working directory.
    if has_term {
        env_logger::Builder::new()
            .filter_level(default_log_level)
            .parse_default_env()
            .init();
    } else {
        let _ = simple_logging::log_to_file("log.out", default_log_level);
    }
    #[cfg(feature = "dhat_heap")]
    let _profiler = {
        log::trace!("Installed dhat");
        dhat::Profiler::new_heap()
    };

    let loading_succeeded = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        // Args are a simple list of paths to open at startup.
        // Paths are OSStrings, let the system handle character encoding restrictions.
        // Todo: Expand glob patterns on windows (on unix this is handled by shell)
        let paths: Vec<std::path::PathBuf> = std::env::args_os().skip(1).map(Into::into).collect();
        // Did we have at least one success? No paths is a success.
        let had_success: std::sync::atomic::AtomicBool = paths.is_empty().into();
        let repo = crate::global::points();
        paths.into_par_iter().for_each(|path| {
            let try_block =
                || -> Result<fuzzpaint_core::queue::DocumentCommandQueue, std::io::Error> {
                    fuzzpaint_core::io::read_path(&path, repo)
                };

            match try_block() {
                Err(e) => {
                    log::error!("failed to open file {}: {e:#}", path.display());
                }
                Ok(queue) => {
                    // We don't care when it's stored, so long as it gets there eventually.
                    had_success.store(true, std::sync::atomic::Ordering::Relaxed);
                    // Defaulted ID, can't fail
                    let _ = global::provider().insert(queue);
                }
            }
        });

        had_success.into_inner()
    };
    // False if every file failed.
    // This should abort the startup if ran from commandline, or give a visual warning and continue
    // if using a GUI.
    if !loading_succeeded {
        log::warn!("Failed to load any provided document.");
    }

    let mut application = window::Application::new();
    let recievers = application.take_renderer_reciever().unwrap();

    std::thread::Builder::new()
        .name("Stylus+Render worker".to_owned())
        .spawn(move || {
            let Ok(receivers) = recievers.recv() else {
                log::error!("Didn't recieve a renderer. Exiting.");
                return;
            };

            let result: Result<((), ()), anyhow::Error> = 'block: {
                let tools = match pen_tools::ToolState::new_from_renderer(&receivers.render_context)
                {
                    Ok(tools) => tools,
                    Err(e) => break 'block Err(e),
                };

                let (send, recv) = tokio::sync::mpsc::channel(4);

                let runtime = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                // between current_thread runtime and try_join, these tasks are
                // not actually run in parallel, just interleaved. This is preferable
                // for now, just a note for future self UwU
                runtime.block_on(async {
                    tokio::try_join!(
                        renderer::render_worker(
                            receivers.render_context,
                            recv,
                            receivers.document_view.clone(),
                        ),
                        stylus_event_collector(
                            receivers.stylus_events,
                            receivers.ui_actions,
                            send,
                            receivers.actions,
                            tools,
                            receivers.document_view,
                        ),
                    )
                })
            };
            if let Err(e) = result {
                log::error!("Helper task exited with err, runtime terminated:\n{e:?}");
            }
        })
        .unwrap();

    application.run()
}

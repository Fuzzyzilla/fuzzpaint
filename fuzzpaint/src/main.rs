#![deny(unsafe_op_in_unsafe_fn)]
#![feature(portable_simd)]
#![feature(once_cell_try)]
#![feature(write_all_vectored)]
#![warn(clippy::pedantic)]
// I know it's bad but while working on a focus it's not really something I wanna be bugged about
// with a full screen of yellow lol.
#![allow(clippy::too_many_lines)]

use std::sync::Arc;
pub mod connections;
mod egui_impl;
pub mod renderer;
pub mod vulkano_prelude;
pub mod window;
use vulkano_prelude::*;
pub mod actions;
pub mod document_viewport_proxy;
pub mod gizmos;
pub mod global;
pub mod my_futures;
pub mod pen_tools;
pub mod picker;
pub mod render_device;
pub mod text;
pub mod ui;
pub mod view_transform;

use fuzzpaint_core::id::FuzzID;

const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

#[cfg(feature = "dhat_heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// Use jemalloc global if enabled + supported playform + not mem profiling
#[cfg(all(
    not(any(feature = "dhat_heap", target_os = "windows")),
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
    mut event_stream: tokio::sync::broadcast::Receiver<window::stylus_events::StylusEventFrame>,
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

struct InitialConnection {
    // lazy: bool,
    ty: InitialConnectionType,
}
enum InitialConnectionType {
    Tcp(std::net::SocketAddr),
    // IPC(),
    // Inprocess,
}

fn client(connection: InitialConnection) -> AnyResult<()> {
    let mut application = window::Application::new()?;
    match connection.ty {
        InitialConnectionType::Tcp(addr) => {
            application.connections().connect_tcp(addr);
        }
    }

    let recievers = application.take_renderer_reciever().unwrap();
    let render_context = application.render_context().clone();

    std::thread::Builder::new()
        .name("Stylus+Render worker".to_owned())
        .spawn(move || {
            let Ok(receivers) = recievers.recv() else {
                log::error!("Didn't recieve a renderer. Exiting.");
                return;
            };

            let result: Result<((), ()), anyhow::Error> = 'block: {
                let tools = match pen_tools::ToolState::new_from_renderer(&render_context) {
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
                            render_context,
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
fn server() -> AnyResult<()> {
    use fuzzpaint_connection::{
        server::{ClientConnection, Connection},
        tcp::server::Client,
    };
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()?;
    // Tokio globals *weeps*
    let _guard = executor.enter();
    let server = fuzzpaint_connection::tcp::server::Builder::default()
        .allow_loopback_nodelay(true)
        .oneshot(true)
        .bind_local();
    let mut server = executor.block_on(server)?;
    let addr = server.local_addr()?;

    // Crappy cross-platform `fork()`
    let mut child = tokio::process::Command::new(std::env::current_exe()?)
        .arg("--client")
        .arg("--tcp")
        .arg(format!("{addr}"))
        .spawn()?;

    let (new_connections, mut recv_new_connections) = tokio::sync::mpsc::channel(1);

    // wait_client is not cancel safe, so it cant be part of the main race-loop.
    // The channel acts as a not-cancel-safe -> cancel-safe bridge.
    let new_client_loop = async {
        loop {
            match server.wait_client().await {
                Ok(client) => {
                    if new_connections.send(client).await.is_err() {
                        break Ok(());
                    }
                }
                Err(e) => break Err(e),
            }
        }
    };
    let client_poll = async {
        use fuzzpaint_connection::{client_msg, server_msg};
        let mut clients = Vec::<Client>::new();
        let mut messages = Vec::new();

        async fn broadcast(
            clients: &mut Vec<Client>,
            message: &server_msg::Message<'_>,
        ) -> std::io::Result<()> {
            if clients.is_empty() {
                return Ok(());
            }
            let results = futures_util::future::join_all(
                clients.iter_mut().map(|client| client.send(message)),
            )
            .await;
            for result in results {
                if let Err(e) = result {
                    return Err(e);
                }
            }
            Ok(())
        }

        loop {
            if let Ok(Some(_)) | Err(_) = child.try_wait() {
                return;
            }
            let recv_any = clients
                .iter_mut()
                .map(|client| async { client.recv().await.expect("todo") })
                .collect::<Vec<_>>();

            let recv_any = my_futures::race(recv_any);
            let await_new_client = recv_new_connections.recv();
            let mut new_client = None;

            tokio::select! {
                biased;
                Some(message) = recv_any => {
                    match message {
                        client_msg::Message::ClientMessage(client_msg::ClientMessage{message}) => messages.push(message.to_owned()),
                    }
                },
                // If returns None, this branch is decarded and does not
                // participate in the selection.
                Some(client) = await_new_client => {
                    new_client = Some(client);
                }
            }
            if let Some(new_client) = new_client {
                clients.push(new_client);
            }
            if !messages.is_empty() {
                for message in messages.drain(..) {
                    let message = server_msg::Message {
                        last_processed: (),
                        message: server_msg::MessageKind::ServerMessage(
                            server_msg::ServerMessage {
                                user_id: Some(()),
                                message: &message,
                            },
                        ),
                    };
                    for client in &mut clients {
                        client.defer_send(&message).unwrap();
                    }
                }
                futures_util::future::join_all(
                    clients
                        .iter_mut()
                        .map(|client| async { client.flush().await.expect("todo") }),
                )
                .await;
            }
        }
    };
    let res = executor.block_on(async {
        tokio::join! {biased; new_client_loop, client_poll}
    });
    res.0.map_err(Into::into)
}

fn log_collector() -> &'static fuzzpaint_logger::CollectLogger {
    static LOGGER: std::sync::OnceLock<&fuzzpaint_logger::CollectLogger> =
        std::sync::OnceLock::new();
    #[cold]
    fn init() -> &'static fuzzpaint_logger::CollectLogger {
        let (base_logger, level) = {
            let default_log_level = if cfg!(debug_assertions) {
                log::LevelFilter::Debug
            } else {
                log::LevelFilter::Info
            };

            // Log to a terminal.
            let logger = env_logger::Builder::new()
                .filter_level(default_log_level)
                .parse_default_env()
                .build();
            let level = logger.filter();
            let logger = Box::leak(Box::new(logger)) as &dyn log::Log;
            (logger, level)
        };
        // But also collect the logs to a vec.
        fuzzpaint_logger::CollectLogger::new()
            .with_tee(base_logger)
            .with_level(level)
            .install()
            .unwrap()
    }
    LOGGER.get_or_init(init)
}

fn main() -> AnyResult<()> {
    // Install logger
    let _ = log_collector();

    #[cfg(feature = "dhat_heap")]
    let _profiler = {
        log::trace!("Installed dhat");
        dhat::Profiler::new_heap();
        // Concurrent process filenames.
        todo!()
    };

    let mut args = std::env::args().fuse();
    let _exec = args.next();
    // :3 grog not care (these are not public facing)
    if args.next().as_deref() == Some("--client")
        && args.next().as_deref() == Some("--tcp")
        && let Some(addr) = args.next()
    {
        client(InitialConnection {
            ty: InitialConnectionType::Tcp(addr.parse()?),
        })
    } else if std::env::args().count() == 1 {
        server()
    } else {
        Err(anyhow::anyhow!("invalid arguments"))
    }
}

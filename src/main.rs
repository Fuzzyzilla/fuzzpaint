#![deny(unsafe_op_in_unsafe_fn)]
#![feature(portable_simd)]
#![feature(once_cell_try)]
#![feature(write_all_vectored)]
#![feature(new_uninit)]
#![feature(float_next_up_down)]
#![warn(clippy::pedantic)]
use std::sync::Arc;
pub mod commands;
mod egui_impl;
pub mod renderer;
pub mod repositories;
pub mod vulkano_prelude;
pub mod window;
use vulkano_prelude::*;
pub mod actions;
pub mod blend;
pub mod brush;
pub mod document_viewport_proxy;
pub mod gizmos;
pub mod id;
pub mod io;
pub mod pen_tools;
pub mod picker;
pub mod render_device;
pub mod state;
pub mod stylus_events;
pub mod tess;
pub mod ui;
pub mod view_transform;
use blend::Blend;

pub use id::FuzzID;
pub use tess::StrokeTessellator;

pub use commands::queue::provider::provider as default_provider;

#[cfg(feature = "dhat_heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Obviously will be user specified on a per-document basis, but for now...
const DOCUMENT_DIMENSION: u32 = 1080;
/// Premultiplied RGBA16F for interesting effects (negative + overbright colors and alpha) with
/// more than 11bit per channel precision in the \[0,1\] range.
/// Will it be user specified in the future?
const DOCUMENT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

use anyhow::Result as AnyResult;

#[must_use]
pub fn preferences_dir() -> Option<std::path::PathBuf> {
    let mut base_dir = dirs::preference_dir()?;
    base_dir.push(env!("CARGO_PKG_NAME"));
    Some(base_dir)
}

pub struct GlobalHotkeys {
    failed_to_load: bool,
    actions_to_keys: actions::hotkeys::ActionsToKeys,
    keys_to_actions: actions::hotkeys::KeysToActions,
}
impl GlobalHotkeys {
    const FILENAME: &'static str = "hotkeys.ron";
    /// Shared global hotkeys, saved and loaded from user preferences.
    /// (Or defaulted, if unavailable for some reason)
    #[must_use]
    pub fn get() -> &'static Self {
        static GLOBAL_HOTKEYS: std::sync::OnceLock<GlobalHotkeys> = std::sync::OnceLock::new();

        GLOBAL_HOTKEYS.get_or_init(|| {
            let mut dir = preferences_dir();
            match dir.as_mut() {
                None => Self::no_path(),
                Some(dir) => {
                    dir.push(Self::FILENAME);
                    Self::load_or_default(dir)
                }
            }
        })
    }
    #[must_use]
    pub fn no_path() -> Self {
        use actions::hotkeys::ActionsToKeys;
        log::warn!("Hotkeys weren't available, defaulting.");
        let default = ActionsToKeys::default();
        // Default action map is reversable - this is assured by the default impl when debugging.
        let reverse = (&default).try_into().unwrap();

        Self {
            failed_to_load: true,
            keys_to_actions: reverse,
            actions_to_keys: default,
        }
    }
    #[must_use]
    fn load_or_default(path: &std::path::Path) -> Self {
        use actions::hotkeys::{ActionsToKeys, KeysToActions};
        let mappings: anyhow::Result<(ActionsToKeys, KeysToActions)> = try_block::try_block! {
            let string = std::fs::read_to_string(path)?;
            let actions_to_keys : ActionsToKeys = ron::from_str(&string)?;
            let keys_to_actions : KeysToActions = (&actions_to_keys).try_into()?;

            Ok((actions_to_keys,keys_to_actions))
        };

        match mappings {
            Ok((actions_to_keys, keys_to_actions)) => Self {
                failed_to_load: false,
                actions_to_keys,
                keys_to_actions,
            },
            Err(_) => Self::no_path(),
        }
    }
    /// Return true if loading user's settings failed. This can be useful for
    /// displaying a warning.
    #[must_use]
    pub fn did_fail_to_load(&self) -> bool {
        self.failed_to_load
    }
    pub fn save(&self) -> anyhow::Result<()> {
        let mut preferences =
            preferences_dir().ok_or_else(|| anyhow::anyhow!("No preferences dir found"))?;
        // Explicity do *not* create recursively. If not found, the user probably has a good reason.
        // Ignore errors (could already exist). Any real errors will be emitted by file access below.
        let _ = std::fs::DirBuilder::new().create(&preferences);

        preferences.push(Self::FILENAME);
        let writer = std::io::BufWriter::new(std::fs::File::create(&preferences)?);
        Ok(ron::ser::to_writer_pretty(
            writer,
            &self.actions_to_keys,
            ron::ser::PrettyConfig::default(),
        )?)
    }
}

/// FIXME! This is temp until I can see that everything is working :3
/// There still needs to be a way to intercommunicate between UI selections, Pen actions, and renderer preview.
#[derive(Clone)]
pub struct AdHocGlobals {
    pub document: state::DocumentID,
    pub brush: state::StrokeBrushSettings,
    pub node: Option<state::graph::AnyID>,
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

#[derive(vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct StrokePoint {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32_SFLOAT)]
    pressure: f32,
    /// Arc length of stroke from beginning to this point
    #[format(R32_SFLOAT)]
    dist: f32,
}

impl StrokePoint {
    #[must_use]
    pub const fn archetype() -> repositories::points::PointArchetype {
        use repositories::points::PointArchetype;
        // | isn't const except for on the bits type! x3
        // This is infallible but unwrap also isn't const.
        match PointArchetype::from_bits(
            PointArchetype::POSITION.bits()
                | PointArchetype::PRESSURE.bits()
                | PointArchetype::ARC_LENGTH.bits(),
        ) {
            Some(s) => s,
            None => unreachable!(),
        }
    }
    #[must_use]
    pub fn lerp(&self, other: &Self, factor: f32) -> Self {
        let inv_factor = 1.0 - factor;

        let s = std::simd::f32x4::from_array([self.pos[0], self.pos[1], self.pressure, self.dist]);
        let o =
            std::simd::f32x4::from_array([other.pos[0], other.pos[1], other.pressure, other.dist]);
        // FMA is planned but unimplemented ;w;
        let n = s * std::simd::f32x4::splat(inv_factor) + (o * std::simd::f32x4::splat(factor));
        Self {
            pos: [n[0], n[1]],
            pressure: n[2],
            dist: n[3],
        }
    }
}

pub struct Stroke {
    brush: state::StrokeBrushSettings,
    points: Vec<StrokePoint>,
}
async fn stylus_event_collector(
    mut event_stream: tokio::sync::broadcast::Receiver<stylus_events::StylusEventFrame>,
    ui_requests: crossbeam::channel::Receiver<ui::requests::UiRequest>,
    mut render_requests: tokio::sync::mpsc::Sender<renderer::requests::RenderRequest>,
    mut action_listener: actions::ActionListener,
    mut tools: pen_tools::ToolState,
    document_preview: Arc<document_viewport_proxy::DocumentViewportPreviewProxy>,
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

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    let has_term = std::io::IsTerminal::is_terminal(&std::io::stdin());
    // Log to a terminal, if available. Else, log to "log.out" in the working directory.
    if has_term {
        env_logger::builder()
            .filter_level(log::LevelFilter::max())
            .init();
    } else {
        let _ = simple_logging::log_to_file("log.out", log::LevelFilter::max());
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
        let repo = repositories::points::global();
        paths.into_par_iter().for_each(|path| {
            let try_block =
                || -> Result<crate::commands::queue::DocumentCommandQueue, std::io::Error> {
                    io::read_path(&path, repo)
                };

            match try_block() {
                Err(e) => {
                    log::error!("failed to open file {path:?}: {e:#}");
                }
                Ok(queue) => {
                    // We don't care when it's stored, so long as it gets there eventually.
                    had_success.store(true, std::sync::atomic::Ordering::Relaxed);
                    // Defaulted ID, can't fail
                    let _ = default_provider().insert(queue);
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

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    if let Err(e) = GlobalHotkeys::get().save() {
        log::warn!("Failed to save hotkey config:\n{e:?}");
    };

    let document_view = Arc::new(document_viewport_proxy::DocumentViewportPreviewProxy::new(
        &render_surface,
    )?);
    let window_renderer = window_surface.with_render_surface(
        render_surface,
        render_context.clone(),
        document_view.clone(),
    )?;

    let event_stream = window_renderer.stylus_events();
    let action_listener = window_renderer.action_listener();
    let ui_requests = window_renderer.ui_listener();

    std::thread::Builder::new()
        .name("Stylus+Render worker".to_owned())
        .spawn(move || {
            #[cfg(feature = "dhat_heap")]
            // Keep alive. Winit takes ownership of main, and will never
            // drop this unless we steal it.
            let _profiler = _profiler;

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
                        renderer::render_worker(render_context, recv, document_view.clone(),),
                        stylus_event_collector(
                            event_stream,
                            ui_requests,
                            send,
                            action_listener,
                            tools,
                            document_view,
                        ),
                    )
                })
            };
            if let Err(e) = result {
                log::error!("Helper task exited with err, runtime terminated:\n{e:?}");
            }
        })
        .unwrap();

    window_renderer.run()
}

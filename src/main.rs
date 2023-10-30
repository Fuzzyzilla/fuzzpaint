#![feature(portable_simd)]
#![feature(once_cell_try)]
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
pub mod gpu_tess;
pub mod id;
pub mod pen_tools;
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
    pub fn get() -> &'static Self {
        static GLOBAL_HOTKEYS: std::sync::OnceLock<GlobalHotkeys> = std::sync::OnceLock::new();

        GLOBAL_HOTKEYS.get_or_init(|| {
            let mut dir = preferences_dir();
            match dir.as_mut() {
                None => Self::no_path(),
                Some(dir) => {
                    dir.push(Self::FILENAME);
                    Self::load_or_default(&dir)
                }
            }
        })
    }
    pub fn no_path() -> Self {
        log::warn!("Hotkeys weren't available, defaulting.");
        use actions::hotkeys::*;
        let default = ActionsToKeys::default();
        // Default action map is reversable - this is assured by the default impl when debugging.
        let reverse = (&default).try_into().unwrap();

        Self {
            failed_to_load: true,
            keys_to_actions: reverse,
            actions_to_keys: default,
        }
    }
    fn load_or_default(path: &std::path::Path) -> Self {
        use actions::hotkeys::*;
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
            Default::default(),
        )?)
    }
}

/// FIXME! This is temp until I can see that everything is working :3
/// There still needs to be a way to intercommunicate between UI selections, Pen actions, and renderer preview.
#[derive(Clone, Copy)]
pub struct Selections {
    pub document: state::DocumentID,
    pub node: Option<state::graph::AnyID>,
}
impl Selections {
    pub fn get() -> &'static parking_lot::RwLock<Option<Selections>> {
        static ONCE: std::sync::OnceLock<parking_lot::RwLock<Option<Selections>>> =
            std::sync::OnceLock::new();

        ONCE.get_or_init(|| Default::default())
    }
    pub fn read_copy() -> Option<Self> {
        *Self::get().read()
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct StrokePoint {
    pos: [f32; 2],
    pressure: f32,
    /// Arc length of stroke from beginning to this point
    dist: f32,
}

impl StrokePoint {
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
    mut action_listener: actions::ActionListener,
    mut tools: pen_tools::ToolState,
    document_preview: Arc<document_viewport_proxy::DocumentViewportPreviewProxy>,
    render_send: tokio::sync::mpsc::UnboundedSender<()>,
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
                    .process(&transform, stylus_frame, &action_frame, &render_send)
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
    #[cfg(feature = "dhat_heap")]
    let _profiler = {
        log::trace!("Installed dhat");
        dhat::Profiler::new_heap()
    };
    env_logger::builder()
        .filter_level(log::LevelFilter::max())
        .init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;

    if let Err(e) = GlobalHotkeys::get().save() {
        log::warn!("Failed to save hotkey config:\n{e:?}");
    };

    let (send, recv) = std::sync::mpsc::channel();
    let ui = ui::MainUI::new(send);
    std::thread::spawn(move || {
        while let Ok(recv) = recv.recv() {
            log::trace!("UI Message: {recv:?}");
        }
    });

    // Test image generators.
    //let (image, future) = make_test_image(render_context.clone())?;
    //let (image, future) = load_document_image(render_context.clone(), &std::path::PathBuf::from("/home/aspen/Pictures/thesharc.png"))?;
    //future.wait(None)?;

    let document_view = Arc::new(document_viewport_proxy::DocumentViewportPreviewProxy::new(
        &render_surface,
    )?);
    let window_renderer = window_surface.with_render_surface(
        render_surface,
        render_context.clone(),
        document_view.clone(),
        ui,
    )?;

    let event_stream = window_renderer.stylus_events();
    let action_listener = window_renderer.action_listener();

    std::thread::spawn(move || {
        #[cfg(feature = "dhat_heap")]
        // Keep alive. Winit takes ownership of main, and will never
        // drop this unless we steal it.
        let _profiler = _profiler;

        let result: Result<((), ()), anyhow::Error> = 'block: {
            let mut tools = match pen_tools::ToolState::new_from_renderer(&render_context) {
                Ok(tools) => tools,
                Err(e) => break 'block Err(e),
            };
            // We don't expect this channel to get very large, but it's important
            // that messages don't get lost under any circumstance, lest an expensive
            // document rebuild be needed :P
            let (render_sender, render_reciever) = tokio::sync::mpsc::unbounded_channel::<()>();

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            // between current_thread runtime and try_join, these tasks are
            // not actually run in parallel, just interleaved. This is preferable
            // for now, just a note for future self UwU
            runtime.block_on(async {
                tokio::try_join!(
                    renderer::render_worker(render_context, document_view.clone(), render_reciever,),
                    stylus_event_collector(
                        event_stream,
                        action_listener,
                        tools,
                        document_view,
                        render_sender,
                    ),
                )
            })
        };
        if let Err(e) = result {
            log::error!("Helper task exited with err, runtime terminated:\n{e:?}")
        }
    });

    window_renderer.run();
}

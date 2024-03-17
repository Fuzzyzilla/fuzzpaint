use crate::egui_impl;
use crate::render_device;
use crate::vulkano_prelude::*;

use std::sync::Arc;

use anyhow::Result as AnyResult;

pub struct WindowSurface {
    event_loop: winit::event_loop::EventLoop<()>,
    win: Arc<winit::window::Window>,
}
impl WindowSurface {
    pub fn new() -> AnyResult<Self> {
        const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

        let event_loop = winit::event_loop::EventLoopBuilder::default().build()?;
        let win = winit::window::WindowBuilder::default()
            .with_title(format!("Fuzzpaint v{}", VERSION.unwrap_or("[unknown]")))
            .with_min_inner_size(winit::dpi::LogicalSize::new(500u32, 500u32))
            .with_transparent(false)
            .build(&event_loop)?;

        let win = Arc::new(win);

        Ok(Self { event_loop, win })
    }
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    pub fn event_loop(&self) -> &winit::event_loop::EventLoop<()> {
        &self.event_loop
    }
    pub fn with_render_surface(
        self,
        render_surface: render_device::RenderSurface,
        render_context: Arc<render_device::RenderContext>,
        preview_renderer: Arc<dyn crate::document_viewport_proxy::PreviewRenderProxy>,
    ) -> anyhow::Result<WindowRenderer> {
        let egui_ctx = egui_impl::EguiCtx::new(self.win.as_ref(), &render_surface)?;

        let tablet_manager = octotablet::Builder::new().build_shared(&self.win).ok();

        let (send, stream) = crate::actions::create_action_stream();

        Ok(WindowRenderer {
            win: self.win,
            render_surface: Some(render_surface),
            swapchain_generation: 0,
            render_context,
            event_loop: Some(self.event_loop),
            last_frame_fence: None,
            egui_ctx,
            tablet_manager,
            ui: crate::ui::MainUI::new(stream.listen()),
            preview_renderer,
            action_collector:
                crate::actions::winit_action_collector::WinitKeyboardActionCollector::new(send),
            action_stream: stream,
            stylus_events: crate::stylus_events::WinitStylusEventCollector::default(),
        })
    }
}

pub struct WindowRenderer {
    event_loop: Option<winit::event_loop::EventLoop<()>>,
    win: Arc<winit::window::Window>,
    /// Always Some. This is to allow it to be take-able to be remade.
    /// Could None represent a temporary loss of surface that can be recovered from?
    render_surface: Option<render_device::RenderSurface>,
    render_context: Arc<render_device::RenderContext>,
    egui_ctx: egui_impl::EguiCtx,
    ui: crate::ui::MainUI,

    action_collector: crate::actions::winit_action_collector::WinitKeyboardActionCollector,
    action_stream: crate::actions::ActionStream,
    // May be None on unsupported platforms.
    tablet_manager: Option<octotablet::Manager>,
    stylus_events: crate::stylus_events::WinitStylusEventCollector,
    swapchain_generation: u32,

    last_frame_fence: Option<vk::sync::future::FenceSignalFuture<Box<dyn GpuFuture>>>,

    preview_renderer: Arc<dyn crate::document_viewport_proxy::PreviewRenderProxy>,
}
impl WindowRenderer {
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    pub fn action_listener(&self) -> crate::actions::ActionListener {
        self.action_stream.listen()
    }
    pub fn ui_listener(&self) -> crossbeam::channel::Receiver<crate::ui::requests::UiRequest> {
        self.ui.listen_requests()
    }
    pub fn stylus_events(
        &self,
    ) -> tokio::sync::broadcast::Receiver<crate::stylus_events::StylusEventFrame> {
        self.stylus_events.frame_receiver()
    }
    pub fn render_surface(&self) -> &render_device::RenderSurface {
        //this will ALWAYS be Some. The option is for taking from a mutable reference for recreation.
        self.render_surface.as_ref().unwrap()
    }
    /// Recreate surface after loss or out-of-date. Todo: This only handles out-of-date and resize.
    pub fn recreate_surface(&mut self) -> AnyResult<()> {
        let new_surface = self
            .render_surface
            .take()
            .unwrap()
            .recreate(Some(self.window().inner_size().into()))?;

        self.egui_ctx.replace_surface(&new_surface)?;

        self.render_surface = Some(new_surface);
        self.swapchain_generation = self.swapchain_generation.wrapping_add(1);

        self.preview_renderer
            .surface_changed(self.render_surface.as_ref().unwrap());

        Ok(())
    }
    fn apply_document_cursor(&mut self) {
        // If egui did not assert a cursor, allow the document to provide an icon.
        // winit_egui handles egui's requests for cursor otherwise.
        if !self.egui_ctx.wants_pointer_input() {
            let cursor = self.preview_renderer.cursor();
            let cursor = cursor.unwrap_or(crate::gizmos::CursorOrInvisible::Icon(
                winit::window::CursorIcon::Default,
            ));

            if let crate::gizmos::CursorOrInvisible::Icon(i) = cursor {
                self.win.set_cursor_icon(i);
                self.win.set_cursor_visible(true);
            }
            if let crate::gizmos::CursorOrInvisible::Invisible = cursor {
                self.win.set_cursor_visible(false);
            }
        }
    }
    pub fn run(mut self) -> Result<(), winit::error::EventLoopError> {
        //There WILL be an event loop if we got here
        let event_loop = self.event_loop.take().unwrap();
        self.window().request_redraw();

        event_loop.run(move |event, target| {
            use winit::event::{Event, WindowEvent};
            match event {
                Event::WindowEvent { event, window_id } if window_id == self.window().id() => {
                    let consumed = self
                        .egui_ctx
                        .push_winit_event(&self.window(), &event)
                        .consumed;
                    if !consumed {
                        self.action_collector.push_event(&event);
                    }
                    match event {
                        WindowEvent::CloseRequested => {
                            target.exit();
                        }
                        WindowEvent::Resized(..) => {
                            self.recreate_surface().expect("Failed to rebuild surface");
                        }
                        WindowEvent::CursorLeft { .. } => {
                            self.stylus_events.set_mouse_pressed(false);
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            // Only take if egui doesn't want it!
                            if !consumed {
                                self.stylus_events.push_position(position.into());
                            }
                        }
                        WindowEvent::MouseInput { state, .. } => {
                            let pressed = winit::event::ElementState::Pressed == state;

                            if pressed {
                                // Only take if egui doesn't want it!
                                if !consumed {
                                    self.stylus_events.set_mouse_pressed(true);
                                }
                            } else {
                                self.stylus_events.set_mouse_pressed(false);
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            if let Err(e) = self.paint() {
                                log::error!("{e:?}");
                            };
                        }
                        _ => (),
                    }
                }
                Event::DeviceEvent {
                    event: winit::event::DeviceEvent::Motion { axis: 2, value },
                    ..
                } => {
                    //Pressure out of 65535
                    self.stylus_events.set_pressure(value as f32 / 65535.0);
                    // Other axes (undocumented and X11 only)
                    // 0 -> x in display space
                    // 1 -> y in display space
                    // 2 -> pressure out of 65535, 0 if not pressed
                    // 3 -> Tilt X, degrees from vertical, + to the right
                    // 4 -> Tilt Y, degrees from vertical, + towards user
                    // 5 -> unknown, always zero (barrel rotation?)
                }
                Event::AboutToWait => {
                    let has_tablet_update = if let Some(tab_events) =
                        self.tablet_manager.as_mut().and_then(|m| m.pump().ok())
                    {
                        let mut has_tablet_update = false;
                        for event in tab_events {
                            if let octotablet::events::Event::Tool { event, .. } = event {
                                match event {
                                    octotablet::events::ToolEvent::Pose(p) => {
                                        if let Some(p) = p.pressure.get() {
                                            self.stylus_events.set_pressure(p);
                                        }
                                        self.stylus_events
                                            .push_position((p.position[0], p.position[1]));
                                        has_tablet_update = true;
                                    }
                                    octotablet::events::ToolEvent::Up
                                    | octotablet::events::ToolEvent::Out => {
                                        self.stylus_events.set_mouse_pressed(false);
                                        has_tablet_update = true;
                                    }
                                    octotablet::events::ToolEvent::Down => {
                                        self.stylus_events.set_mouse_pressed(true);
                                        has_tablet_update = true;
                                    }
                                    _ => (),
                                };
                            }
                        }
                        has_tablet_update
                    } else {
                        false
                    };
                    // run UI logics
                    self.do_ui();
                    self.apply_document_cursor();

                    // Request draw if any interactive element wants it (UI, document, or tablet)
                    if has_tablet_update
                        || self.egui_ctx.needs_redraw()
                        || self.preview_renderer.has_update()
                    {
                        self.window().request_redraw();
                    }

                    // End frame
                    self.stylus_events.finish();
                    // Wait. We'll be notified when to redraw UI, but the document preview could assert
                    // an update at any time! Thus, we must poll. U_U
                    target.set_control_flow(winit::event_loop::ControlFlow::wait_duration(
                        std::time::Duration::from_millis(50),
                    ));
                }
                _ => (),
            }
        })
    }
    fn do_ui(&mut self) {
        let mut viewport = Default::default();
        self.egui_ctx
            .update(self.win.as_ref(), |ctx| viewport = self.ui.ui(ctx));

        // Todo: only change if... actually changed :P
        self.preview_renderer
            .viewport_changed(viewport.0, viewport.1);
    }
    fn paint(&mut self) -> AnyResult<()> {
        let (idx, suboptimal, image_future) =
            match vk::acquire_next_image(self.render_surface().swapchain().clone(), None) {
                Err(vk::Validated::Error(vk::VulkanError::OutOfDate)) => {
                    log::info!("Swapchain unusable. Recreating");
                    //We cannot draw on this surface as-is. Recreate and request another try next frame.
                    //TODO: Race condition, somehow! Surface is recreated with an out-of-date size.
                    self.recreate_surface()?;
                    self.window().request_redraw();
                    return Ok(());
                }
                Err(e) => {
                    //Todo. Many of these errors are recoverable!
                    anyhow::bail!("Surface image acquire failed! {e:?}");
                }
                Ok(r) => r,
            };
        // After we present, recreate if suboptimal.
        defer::defer(|| {
            if suboptimal {
                self.recreate_surface().unwrap();
            }
        });
        let commands = self.egui_ctx.build_commands(idx);

        //Wait for previous frame to end. (required for safety of preview render proxy)
        self.last_frame_fence.take().map(|fence| fence.wait(None));

        let preview_commands = unsafe {
            self.preview_renderer.render(
                self.render_surface.as_ref().unwrap().swapchain_images()[idx as usize].clone(),
                idx,
            )
        };
        let preview_commands = match preview_commands {
            Ok(commands) => commands,
            Err(e) => {
                log::warn!("Failed to build preview commands {e:?}");
                smallvec::SmallVec::new()
            }
        };

        let render_complete = match commands {
            Some((Some(transfer), draw)) => {
                let transfer_future = self
                    .render_context
                    .now()
                    .then_execute(
                        self.render_context.queues().transfer().queue().clone(),
                        transfer,
                    )?
                    .boxed()
                    .then_signal_fence_and_flush()?;

                // Todo: no matter what I do, i cannot seem to get semaphores
                // to work. Ideally, the only thing that needs to wait is the
                // egui render commands, however it simply refuses to actually
                // wait for the semaphore. For now, I just stall the thread.
                transfer_future.wait(None)?;

                let mut future = image_future.boxed();

                for buffer in preview_commands {
                    future = future
                        .then_execute(
                            self.render_context.queues().graphics().queue().clone(),
                            buffer,
                        )?
                        .boxed();
                }

                future
                    .then_execute(
                        self.render_context.queues().graphics().queue().clone(),
                        draw,
                    )?
                    .boxed()
            }
            Some((None, draw)) => {
                let mut future = image_future.boxed();

                for buffer in preview_commands {
                    future = future
                        .then_execute(
                            self.render_context.queues().graphics().queue().clone(),
                            buffer,
                        )?
                        .boxed();
                }
                future
                    .then_execute(
                        self.render_context.queues().graphics().queue().clone(),
                        draw,
                    )?
                    .boxed()
            }
            None => image_future.boxed(),
        };

        self.window().pre_present_notify();

        let next_frame_future = render_complete
            .then_swapchain_present(
                self.render_context
                    .queues()
                    .present()
                    .unwrap()
                    .queue()
                    .clone(),
                vk::SwapchainPresentInfo::swapchain_image_index(
                    self.render_surface.as_ref().unwrap().swapchain().clone(),
                    idx,
                ),
            )
            .boxed()
            .then_signal_fence_and_flush()?;

        self.last_frame_fence = Some(next_frame_future);

        Ok(())
    }
}

use crate::document_viewport_proxy::PreviewRenderProxy;
use crate::egui_impl;
use crate::render_device;
use crate::vulkano_prelude::*;

struct RemoteChanged();
type UserEvent = RemoteChanged;
impl crate::connections::Waker for winit::event_loop::EventLoopProxy<UserEvent> {
    fn wake(&mut self, _which: crate::connections::ConnectionID) {
        let _ = self.send_event(RemoteChanged());
    }
}

use std::sync::{Arc, Weak};

use anyhow::Result as AnyResult;

enum State<T> {
    Deferred,
    Extant(T),
    Killed,
}

pub struct Application {
    pre_setup_loop: Option<winit::event_loop::EventLoop<UserEvent>>,
    // Objects that depend on a window, including the window itself.
    window_objects: State<WindowObjects>,
    connections: crate::connections::ClientConnectionsManager,
    render_context: Arc<render_device::RenderContext>,
    // Channel that will be notified when the renderer is made, once the window
    // is ready.
    renderer_sender: Option<oneshot::Sender<Receivers>>,
    renderer_reciever: Option<oneshot::Receiver<Receivers>>,
}
impl Application {
    pub fn new() -> AnyResult<Self> {
        let pre_setup_loop =
            winit::event_loop::EventLoop::<UserEvent>::with_user_event().build()?;
        let render_context = render_device::RenderContext::new_with_display(Some(&pre_setup_loop))?;
        let (send, recv) = oneshot::channel();

        let connections = crate::connections::ClientConnectionsManager::spawn(Box::new(
            pre_setup_loop.create_proxy(),
        ))?;

        Ok(Self {
            pre_setup_loop: Some(pre_setup_loop),
            window_objects: State::Deferred,
            connections,
            render_context,
            renderer_sender: Some(send),
            renderer_reciever: Some(recv),
        })
    }
    pub fn connections(&mut self) -> &mut crate::connections::ClientConnectionsManager {
        &mut self.connections
    }
    pub fn render_context(&self) -> &Arc<render_device::RenderContext> {
        &self.render_context
    }
    /// Take a channel that will recieve the rendering context, once it is created.
    pub fn take_renderer_reciever(&mut self) -> Option<oneshot::Receiver<Receivers>> {
        self.renderer_reciever.take()
    }
    pub fn run(mut self) -> AnyResult<()> {
        self.pre_setup_loop
            .take()
            .unwrap()
            .run_app(&mut self)
            .map_err(Into::into)
    }
}
impl winit::application::ApplicationHandler<UserEvent> for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if matches!(self.window_objects, State::Deferred) {
            // Always emitted first, even on platforms without a suspend-resume
            // cycle. Only recreate if it's the first time (i.e. dont attempt to
            // recreate after a suspend.)
            match WindowObjects::new(self.render_context.clone(), event_loop) {
                Ok(window_objects) => {
                    if let Some(send) = self.renderer_sender.take() {
                        let _ = send.send(window_objects.receivers());
                    }
                    self.window_objects = State::Extant(window_objects);
                }
                Err(e) => {
                    log::error!("FATAL: Failed to create window: {e}");
                    self.window_objects = State::Killed;
                    event_loop.exit();
                }
            }
        }
    }
    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // We can't currently handle the suspend-resume lifecycle, just die :(
        // Drop the window objects immediately if any, and request an exit.
        self.window_objects = State::Killed;
        event_loop.exit();
    }
    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let State::Extant(window_objects) = &mut self.window_objects else {
            return;
        };
        match event {
            winit::event::WindowEvent::RedrawRequested => {
                window_objects.redraw_requested(self.connections.lock())
            }
            event => window_objects.window_event(event),
        }
    }
    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let State::Extant(window_objects) = &mut self.window_objects else {
            return;
        };
        window_objects.device_event(device_id, event);
    }
    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        let State::Extant(window_objects) = &mut self.window_objects else {
            return;
        };
        match event {
            // Must be called by the main thread for portability, hence the use
            // of a user event.
            RemoteChanged() => window_objects.win.request_redraw(),
        }
    }
    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let State::Extant(window_objects) = &mut self.window_objects else {
            return;
        };
        window_objects.about_to_wait(event_loop);
    }
    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Drop the window objects if any.
        self.window_objects = State::Killed;
    }
}
pub struct Receivers {
    pub actions: crate::actions::ActionListener,
    pub ui_actions: crossbeam::channel::Receiver<crate::ui::requests::UiRequest>,
    pub stylus_events: tokio::sync::broadcast::Receiver<crate::stylus_events::StylusEventFrame>,
    pub document_view: Arc<crate::document_viewport_proxy::Proxy>,
}
pub struct WindowObjects {
    win: Arc<winit::window::Window>,
    render_surface: render_device::RenderSurface,
    render_context: Arc<render_device::RenderContext>,
    egui_ctx: egui_impl::Ctx,
    ui: crate::ui::MainUI,

    enable_document_view: bool,

    action_collector: crate::actions::winit_action_collector::WinitKeyboardActionCollector,
    action_stream: crate::actions::ActionStream,
    // May be None on unsupported platforms.
    tablet_manager: Option<octotablet::Manager>,
    stylus_events: crate::stylus_events::WinitStylusEventCollector,
    swapchain_generation: u32,

    last_frame_fence: Option<vk::sync::future::FenceSignalFuture<Box<dyn GpuFuture>>>,

    preview_renderer: Arc<crate::document_viewport_proxy::Proxy>,
}
impl WindowObjects {
    fn new(
        render_context: Arc<render_device::RenderContext>,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> AnyResult<Self> {
        const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");
        let win = event_loop.create_window(
            winit::window::Window::default_attributes()
                .with_title(format!("Fuzzpaint v{}", VERSION.unwrap_or("[unknown]")))
                .with_min_inner_size(winit::dpi::LogicalSize::new(500u32, 500u32))
                .with_transparent(false),
        )?;
        let win = Arc::new(win);

        let render_surface =
            render_device::RenderSurface::new(render_context.clone(), win.clone())?;
        let preview_renderer =
            Arc::new(crate::document_viewport_proxy::Proxy::new(&render_surface)?);

        let tablet_manager = octotablet::Builder::new()
            .emulate_tool_from_mouse(false)
            .build_shared(&win)
            .ok();

        let (send, stream) = crate::actions::create_action_stream();

        let egui_ctx = egui_impl::Ctx::new(win.as_ref(), &render_surface)?;
        win.request_redraw();

        Ok(Self {
            win,
            render_surface,
            swapchain_generation: 0,
            render_context,
            last_frame_fence: None,
            egui_ctx,
            tablet_manager,
            ui: crate::ui::MainUI::new(stream.listen()),
            enable_document_view: true,
            preview_renderer,
            action_collector:
                crate::actions::winit_action_collector::WinitKeyboardActionCollector::new(send),
            action_stream: stream,
            stylus_events: crate::stylus_events::WinitStylusEventCollector::default(),
        })
    }
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    fn receivers(&self) -> Receivers {
        Receivers {
            actions: self.action_stream.listen(),
            ui_actions: self.ui.listen_requests(),
            stylus_events: self.stylus_events.frame_receiver(),
            document_view: self.preview_renderer.clone(),
        }
    }
    pub fn render_surface(&self) -> &render_device::RenderSurface {
        &self.render_surface
    }
    /// Recreate surface after loss or out-of-date. Todo: This only handles out-of-date and resize.
    pub fn recreate_surface(&mut self) -> AnyResult<()> {
        self.render_surface
            .recreate(self.window().inner_size().into())?;
        self.egui_ctx.replace_surface(&self.render_surface)?;

        self.swapchain_generation = self.swapchain_generation.wrapping_add(1);

        self.preview_renderer.surface_changed(&self.render_surface);

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
                self.win.set_cursor(i);
                self.win.set_cursor_visible(true);
            }
            if let crate::gizmos::CursorOrInvisible::Invisible = cursor {
                self.win.set_cursor_visible(false);
            }
        }
    }
    pub fn window_event(&mut self, event: winit::event::WindowEvent) {
        use winit::event::WindowEvent;
        let consumed = self
            .egui_ctx
            .push_winit_event(&self.window(), &event)
            .consumed;
        if !consumed {
            self.action_collector.push_event(&event);
        }
        match event {
            WindowEvent::CloseRequested => {
                // Mark the UI, allowing it to veto this close.
                self.ui.close_requested();
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
                // Handled externally with a call to Self::redraw_requested
                unreachable!()
            }
            _ => (),
        }
    }
    pub fn redraw_requested(&mut self, connections: crate::connections::ConnectionsLock) {
        self.do_ui(connections);
        // Overwrite the Egui provided cursor over the doc area.
        self.apply_document_cursor();

        // Render and present the updated UI
        if let Err(e) = self.paint() {
            log::error!("{e:?}");
        }
    }
    pub fn device_event(
        &mut self,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        use winit::event::DeviceEvent;
        if let DeviceEvent::Motion { axis: 2, value } = event {
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
    }
    pub fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // The UI has requested the app exit. Do so!
        if self.ui.should_close() {
            event_loop.exit();
            // No need to redraw.
            return;
        }
        if self.egui_ctx.take_wants_update() {
            self.win.request_redraw();
        }

        let has_tablet_update = if let Some(tab_events) =
            self.tablet_manager.as_mut().and_then(|m| m.pump().ok())
        {
            let mut has_tablet_update = false;
            for event in tab_events {
                if let octotablet::events::Event::Tool { event, tool } = event {
                    // If the event isn't emulated from some other device, send the event to winit_egui
                    // so that the stylus can be used to interact with the egui layers.
                    if !matches!(tool.tool_type, Some(octotablet::tool::Type::Emulated)) {
                        // Safety: we must not pass the returned event deviceID into any winit functions.
                        if let Some(winit_event) = unsafe {
                            crate::stylus_events::winit_event_from_octotablet(
                                &event,
                                self.win.scale_factor(),
                            )
                        } {
                            // Safety: Looking into the code of this, there is no path where the device ID is taken and given to winit.
                            // If that occurs, it's UB - MAKE SURE TO CHECK BEFORE UPDATING VERS ;3
                            // Last checked `egui-winit` version: 0.33.3
                            let ignore = self
                                .egui_ctx
                                .push_winit_event(&self.win, &winit_event)
                                .consumed;

                            // Egui ate the event, skip further processing.
                            if ignore {
                                continue;
                            }
                        }
                    }

                    // Wasn't consumed, forward it to the event stream for the tools to use.
                    match event {
                        octotablet::events::ToolEvent::Pose(p) => {
                            if let Some(p) = p.pressure.get() {
                                self.stylus_events.set_pressure(p);
                            }
                            self.stylus_events
                                .push_position((p.position[0], p.position[1]));

                            has_tablet_update = true;
                        }
                        octotablet::events::ToolEvent::Up | octotablet::events::ToolEvent::Out => {
                            self.stylus_events.set_mouse_pressed(false);
                            has_tablet_update = true;
                        }
                        octotablet::events::ToolEvent::Down => {
                            self.stylus_events.set_mouse_pressed(true);
                            has_tablet_update = true;
                        }
                        _ => (),
                    }
                }
            }
            has_tablet_update
        } else {
            false
        };

        // Request draw if any interactive element wants it (UI, document, or tablet)
        if has_tablet_update
            || self.egui_ctx.peek_wants_update()
            || self.preview_renderer.has_update()
        {
            // winit automagically coalesces these if we call it too often, that's okay ;3
            self.window().request_redraw();
        }

        // End stylus frame
        self.stylus_events.finish();

        // Wait. We'll be notified when to redraw UI, but the document preview or octotablet could assert
        // an update at any time! Thus, we must poll. U_U
        event_loop.set_control_flow(winit::event_loop::ControlFlow::wait_duration(
            std::time::Duration::from_millis(50),
        ));
    }
    fn do_ui(&mut self, mut connections: crate::connections::ConnectionsLock) {
        let viewport = self
            .egui_ctx
            .update(self.win.as_ref(), |ctx| self.ui.ui(ctx, &mut connections));
        // Drop the lock ASAP.
        drop(connections);

        // Todo: only change if... actually changed :P
        if let Some(viewport) = viewport {
            self.enable_document_view = true;
            self.preview_renderer.viewport_changed(
                cgmath::Point2 {
                    x: viewport.0.x,
                    y: viewport.0.y,
                },
                cgmath::Vector2 {
                    x: viewport.1.x,
                    y: viewport.1.y,
                },
            );
        } else {
            self.enable_document_view = false;
        }
    }
    fn paint(&mut self) -> AnyResult<()> {
        let (idx, suboptimal, image_future) = match vk::acquire_next_image(
            self.render_surface().swapchain().unwrap().clone(),
            None,
        ) {
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

        // Print a warning if swapchain image future is dropped. Per a dire warning in the comments of vulkano,
        // dropping futures can result in that swapchain image being lost forever...!
        let bail_warning = defer::defer(|| log::warn!("Dropped swapchain future."));

        //Wait for previous frame to end. (required for safety of preview render proxy)
        self.last_frame_fence.take().map(|fence| fence.wait(None));

        let preview_commands = self.enable_document_view.then(|| unsafe {
            self.preview_renderer.render(
                self.render_surface.swapchain_images().unwrap()[idx as usize].clone(),
                idx,
            )
        });
        let preview_commands = match preview_commands {
            Some(Ok(commands)) => commands,
            None => smallvec::SmallVec::new(),
            Some(Err(e)) => {
                log::warn!("Failed to build preview commands {e:?}");
                smallvec::SmallVec::new()
            }
        };

        let commands = self
            .egui_ctx
            // Preview commands are responsible for turning the UNDEFINED image into a well-defined state.
            // If there are none, instruct egui renderer to clear it first.
            .build_commands(idx, preview_commands.is_empty());

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
            None => anyhow::bail!("no commands submitted"),
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
                    self.render_surface.swapchain().unwrap().clone(),
                    idx,
                ),
            )
            .boxed()
            .then_signal_fence_and_flush()?;

        std::mem::forget(bail_warning);

        self.last_frame_fence = Some(next_frame_future);

        // After we present, recreate if suboptimal.
        if suboptimal {
            self.recreate_surface().unwrap();
        }

        Ok(())
    }
}

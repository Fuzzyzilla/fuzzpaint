use std::sync::Arc;
use crate::vulkano_prelude::*;
use crate::render_device;
use crate::egui_impl;
use crate::gpu_err::*;

use anyhow::Result as AnyResult;

pub struct WindowSurface {
    event_loop: winit::event_loop::EventLoop<()>,
    win: Arc<winit::window::Window>,
}
impl WindowSurface {
    pub fn new() -> AnyResult<Self> {
        const VERSION : Option<&'static str> = option_env!("CARGO_PKG_VERSION");

        let event_loop = winit::event_loop::EventLoopBuilder::default().build();
        let win = winit::window::WindowBuilder::default()
            .with_title(format!("Fuzzpaint v{}", VERSION.unwrap_or("[unknown]")))
            .with_min_inner_size(winit::dpi::LogicalSize::new(500u32, 500u32))
            .with_transparent(false)
            .build(&event_loop)?; 

        Ok(Self {
            event_loop: event_loop,
            win: Arc::new(win),
        })
    }
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    pub fn with_render_surface(
        self,
        render_surface: render_device::RenderSurface,
        render_context: Arc<render_device::RenderContext>,
        preview_renderer : Arc<parking_lot::RwLock<dyn crate::PreviewRenderProxy>>,
    ) -> GpuResult<WindowRenderer> {
        let egui_ctx = egui_impl::EguiCtx::new(&render_surface)?;
        Ok(WindowRenderer {
            win: self.win,
            render_surface: Some(render_surface),
            swapchain_generation: 0,
            render_context,
            event_loop: Some(self.event_loop),
            last_frame_fence: None,
            egui_ctx,
            preview_renderer,
            stylus_events: Default::default(),
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

    stylus_events: crate::stylus_events::WinitStylusEventCollector,
    swapchain_generation: u32,

    last_frame_fence: Option<vk::sync::future::FenceSignalFuture<Box<dyn GpuFuture>>>,

    preview_renderer: Arc<parking_lot::RwLock<dyn crate::PreviewRenderProxy>>,
}
impl WindowRenderer {
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
    }
    pub fn stylus_events(&self) -> tokio::sync::broadcast::Receiver<crate::stylus_events::StylusEventFrame> {
        self.stylus_events.frame_receiver()
    }
    /*
    pub fn gen_framebuffers(&mut self) {
        self.swapchain_framebuffers = Vec::with_capacity(self.render_surface.swapchain_images.len());

        self.swapchain_framebuffers.extend(
            self.render_surface.swapchain_images.iter()
                .map(|image| {
                    vulkano::render_pass::Framebuffer::
                })
        )
    }*/
    pub fn render_surface(&self) -> &render_device::RenderSurface {
        //this will ALWAYS be Some. The option is for taking from a mutable reference for recreation.
        &self.render_surface.as_ref().unwrap()
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

        self.preview_renderer.write().surface_changed(self.render_surface.as_ref().unwrap());

        Ok(())
    }
    fn apply_platform_output(&mut self, out: egui::PlatformOutput) {
        //Todo: Copied text
        if let Some(url) = out.open_url {
            //Todo: x-platform lol
            let out = std::process::Command::new("xdg-open").arg(url.url).spawn();
            if let Err(e) = out {
                log::error!("Failed to open url: {e:?}");
            }
        }

        if let Some(cursor) = egui_impl::egui_to_winit_cursor(out.cursor_icon) {
            self.win.set_cursor_icon(cursor);
            self.win.set_cursor_visible(true);
        } else {
            self.win.set_cursor_visible(false);
        }
    }
    pub fn run(mut self) -> ! {
        //There WILL be an event loop if we got here
        let event_loop = self.event_loop.take().unwrap();
        self.window().request_redraw();

        event_loop.run(move |event, _, control_flow| {
            use winit::event::{Event, WindowEvent};

            //Weird ownership problems here.
            let Some(event) = event.to_static() else {return};
            self.egui_ctx.push_winit_event(&event);

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                        return;
                    }
                    WindowEvent::Resized(..) => {
                        self.recreate_surface().expect("Failed to rebuild surface");
                    }
                    WindowEvent::CursorLeft { .. } => {
                        self.stylus_events.set_mouse_pressed(false);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        // Only take if egui doesn't want it!
                        if !self.egui_ctx.wants_pointer_input() {
                            self.stylus_events.push_position(position.into());
                        }
                    }
                    WindowEvent::MouseInput { state, .. } => {
                        let pressed = winit::event::ElementState::Pressed == state;

                        if pressed {
                            // Only take if egui doesn't want it!
                            if !self.egui_ctx.wants_pointer_input() {
                                self.stylus_events.set_mouse_pressed(true)
                            }
                        } else {
                            self.stylus_events.set_mouse_pressed(false)
                        }

                    }
                    _ => (),
                },
                Event::DeviceEvent { event, .. } => {

                    match event {
                        //Pressure out of 65535
                        winit::event::DeviceEvent::Motion { axis: 2, value } => {
                            self.stylus_events.set_pressure(value as f32 / 65535.0)
                        }
                        _ => ()
                    }
                    // 0 -> x in display space
                    // 1 -> y in display space
                    // 2 -> pressure out of 65535, 0 if not pressed
                    // 3 -> Tilt X, degrees from vertical, + to the right
                    // 4 -> Tilt Y, degrees from vertical, + towards user
                    // 5 -> unknown, always zero (rotation?)
                }
                Event::MainEventsCleared => {
                    //Draw!
                    if let Some(output) = self.do_ui() {
                        self.apply_platform_output(output);
                    };

                    if self.egui_ctx.needs_redraw() {
                        self.window().request_redraw()
                    }

                    self.stylus_events.finish();
                }
                Event::RedrawRequested(..) => {
                    if let Err(e) = self.paint() {
                        log::error!("{e:?}")
                    };
                }
                Event::RedrawEventsCleared => {
                    *control_flow = winit::event_loop::ControlFlow::Wait;
                }
                _ => (),
            }
        });
    }
    fn do_ui(&mut self) -> Option<egui::PlatformOutput> {
        static mut color : egui::Color32 = egui::Color32::BLUE;
        static mut documents : Vec<(u32, bool)> = Vec::new();

        struct Layer {
            name: String,
            blend_mode: crate::BlendMode,
            opacity: f32,
            key: u32,
        }
        static mut selected_layer_key : u32 = 0;
        static mut layers : Vec<Layer> = Vec::new();
        self.egui_ctx.update(|ctx| {
            egui::TopBottomPanel::top("file")   
                .show(&ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("🐑");
                        egui::menu::bar(ui, |ui| {
                            ui.menu_button("File", |ui| {
                                let add_button = |ui : &mut egui::Ui, label, shortcut| -> egui::Response {
                                    let mut button = egui::Button::new(label);
                                    if let Some(shortcut) = shortcut {
                                        button = button.shortcut_text(shortcut);
                                    }
                                    ui.add(button)
                                };
                                if add_button(ui, "New", Some("Ctrl+N")).clicked() {
                                    unsafe {
                                        documents.push(
                                            (
                                                documents.last().map(|(i, _)| i + 1).unwrap_or_default() as u32,
                                                false
                                            )
                                        ) ;
                                    }
                                };
                                let _ = add_button(ui, "Save", Some("Ctrl+S"));
                                let _ = add_button(ui, "Save as", Some("Ctrl+Shift+S"));
                                let _ = add_button(ui, "Open", Some("Ctrl+O"));
                                let _ = add_button(ui, "Open as new", None);
                                let _ = add_button(ui, "Export", None);
                            });
                            ui.menu_button("Edit", |_| ());
                        });
                    });
                });

            egui::SidePanel::right("Layers")
                .show(&ctx, |ui| {
                    ui.label("Layers");
                    ui.separator();

                    ui.horizontal(|ui| {
                        if ui.button("➕").clicked() {
                            let new_idx = unsafe {layers.len() as u32};
                            let layer = Layer {
                                blend_mode: crate::BlendMode::Normal,
                                key: new_idx,
                                name: format!("Layer {new_idx}"),
                                opacity: 1.0
                            };
                            unsafe {
                                layers.push(layer);
                            }
                        }
                        let _ = ui.button("🗀");
                        let _ = ui.button("⤵").on_hover_text("Merge down");
                        let _ = ui.button("✖").on_hover_text("Delete layer");
                    });

                    ui.separator();
                    egui::ScrollArea::vertical()
                        .show(ui, |ui| {
                            unsafe {
                                for layer in layers.iter_mut() {
                                    ui.group(|ui|{
                                        ui.horizontal(|ui| {
                                            let mut checked = layer.key == selected_layer_key;
                                            ui.checkbox(&mut checked, "").clicked();
                                            if checked {
                                                selected_layer_key = layer.key;
                                            }
                                            ui.text_edit_singleline(&mut layer.name);
                                        });
                                        ui.horizontal_wrapped(|ui| {
                                            ui.add(egui::DragValue::new(&mut layer.opacity).fixed_decimals(2).speed(0.01).clamp_range(0.0..=1.0));
                                            egui::ComboBox::new(layer.key, "Mode")
                                                .selected_text(layer.blend_mode.as_ref())
                                                .show_ui(ui, |ui| {
                                                    for blend_mode in <crate::BlendMode as strum::IntoEnumIterator>::iter() {
                                                        ui.selectable_value(&mut layer.blend_mode, blend_mode, blend_mode.as_ref());
                                                    }
                                                });
                                        })
                                    });
                                }
                            }
                        });
                });

            egui::SidePanel::left("Color picker")
                .show(&ctx, |ui| {
                    ui.label("Color");
                    ui.separator();
                    unsafe {
                        egui::color_picker::color_picker_color32(ui, &mut color, egui::color_picker::Alpha::OnlyBlend)
                    }
                });

            egui::TopBottomPanel::top("documents")
                .show(&ctx, |ui| {
                    egui::ScrollArea::horizontal()
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                //Safety - not running concurrently.
                                unsafe {
                                    let mut new_selected = None;
                                    let mut deleted = Vec::new();
                                    for (i, selected) in documents.iter() {
                                        egui::containers::Frame::group(ui.style())
                                            .outer_margin(egui::Margin::symmetric(0.0, 0.0))
                                            .inner_margin(egui::Margin::symmetric(0.0, 0.0))
                                            .multiply_with_opacity(if *selected {1.0} else {0.0})
                                            .rounding(egui::Rounding{ne: 2.0, nw: 2.0, ..0.0.into()})
                                            .show(ui, |ui| {
                                                if ui.selectable_label(*selected, format!("Document {i}")).clicked() {
                                                    new_selected = Some(i);
                                                }
                                                if ui.small_button("✖").clicked() {
                                                    deleted.push(*i);
                                                    if Some(i) == new_selected {
                                                        new_selected = None;
                                                    }
                                                }
                                            });
                                    }
                                    documents.retain(|(i, _)| !deleted.contains(i));
                                    if let Some(new_selected) = new_selected {
                                        for (i, selected) in documents.iter_mut() {
                                            if i == new_selected {
                                                *selected = true;
                                            } else {
                                                *selected = false;
                                            }
                                        }
                                    }
                                }
                            });
                        });
                });

        })
    }
    fn paint(&mut self) -> AnyResult<()> {
        let (idx, suboptimal, image_future) =
            match vk::acquire_next_image(self.render_surface().swapchain().clone(), None) {
                Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                    log::info!("Swapchain unusable. Recreating");
                    //We cannot draw on this surface as-is. Recreate and request another try next frame.
                    self.recreate_surface()?;
                    self.window().request_redraw();
                    return Ok(())
                }
                Err(e) => {
                    //Todo. Many of these errors are recoverable!
                    anyhow::bail!("Surface image acquire failed! {e:?}");
                }
                Ok(r) => r,
            };


        // Lmao
        // Free up resources from the last time this frame index was rendered
        // Todo: call much much sooner.
        let preview_commands = {
            let mut lock = self.preview_renderer.write();

            lock.render_complete(idx);
            let preview_commands = lock.render(idx)?;
            preview_commands
        };
        let commands = self.egui_ctx.build_commands(idx);

        //Wait for previous frame to end.
        self.last_frame_fence.take().map(|fence| fence.wait(None));

        let render_complete = match commands {
            Some((Some(transfer), draw)) => {
                let transfer_future =
                    self.render_context.now()
                    .then_execute(
                        self.render_context.queues().transfer().queue().clone(),
                        transfer
                    )?
                    .boxed()
                    .then_signal_fence_and_flush()?;

                // Todo: no matter what I do, i cannot seem to get semaphores
                // to work. Ideally, the only thing that needs to wait is the
                // egui render commands, however it simply refuses to actually
                // wait for the semaphore. For now, I just stall the thread.
                transfer_future.wait(None)?;

                image_future
                    .then_execute(
                        self.render_context.queues().graphics().queue().clone(),
                        preview_commands
                    )?
                    .then_execute(
                        self.render_context.queues().graphics().queue().clone(),
                        draw
                    )?
                    .boxed()
            }
            Some((None, draw)) => {
                image_future
                    .then_execute(
                        self.render_context.queues().graphics().queue().clone(),
                        preview_commands
                    )?
                    .then_execute_same_queue(draw)?
                    .boxed()
            }
            None => {
                image_future
                    .then_execute(
                        self.render_context.queues().graphics().queue().clone(),
                        preview_commands
                    )?
                    .boxed()
            }
        };

        let next_frame_future = render_complete
            .then_swapchain_present(
                self.render_context.queues().present().unwrap().queue().clone(),
                vk::SwapchainPresentInfo::swapchain_image_index(self.render_surface.as_ref().unwrap().swapchain().clone(), idx)
            )
            .boxed()
            .then_signal_fence_and_flush()?;

        self.last_frame_fence = Some(next_frame_future);

        if suboptimal {
            self.recreate_surface()?
        }

        Ok(())
    }
}
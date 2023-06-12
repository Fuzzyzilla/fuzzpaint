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
        let event_loop = winit::event_loop::EventLoopBuilder::default().build();
        let win = winit::window::WindowBuilder::default()
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
    ) -> GpuResult<WindowRenderer> {
        let mut egui_ctx = egui_impl::EguiCtx::new(&render_surface)?;
        Ok(WindowRenderer {
            win: self.win,
            render_surface: Some(render_surface),
            render_context,
            event_loop: Some(self.event_loop),
            egui_ctx,
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
}
impl WindowRenderer {
    pub fn window(&self) -> Arc<winit::window::Window> {
        self.win.clone()
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

        self.egui_ctx.replace_surface(&new_surface);

        self.render_surface = Some(new_surface);

        Ok(())
    }
    fn apply_platform_output(&mut self, out: egui::PlatformOutput) {
        //Todo: Copied text
        if let Some(url) = out.open_url {
            //Todo: x-platform lol
            let out = std::process::Command::new("xdg-open").arg(url.url).spawn();
            if let Err(e) = out {
                eprintln!("Failed to open url: {e:?}");
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
                    _ => (),
                },
                Event::MainEventsCleared => {
                    //Draw!
                    if let Some(output) = self.do_ui() {
                        self.apply_platform_output(output);
                    };

                    if self.egui_ctx.needs_redraw() {
                        self.window().request_redraw()
                    }
                }
                Event::RedrawRequested(..) => {
                    self.paint();
                }
                _ => (),
            }
        });
    }
    fn do_ui(&mut self) -> Option<egui::PlatformOutput> {
        self.egui_ctx.update(|ctx| {
            egui::Window::new("🐑 Baa").show(&ctx, |ui| {
                ui.label("Testing testing 123!!");
            });

            egui::Window::new("Beep boop").show(&ctx, |ui| {
                ui.label("Testing testing 345!!");
            });
        })
    }
    fn paint(&mut self) {
        let (idx, suboptimal, image_future) =
            match vk::acquire_next_image(self.render_surface().swapchain().clone(), None) {
                Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                    eprintln!("Swapchain unusable.. Recreating");
                    //We cannot draw on this surface as-is. Recreate and request another try next frame.
                    self.recreate_surface()
                        .expect("Failed to recreate render surface after it went out-of-date.");
                    self.window().request_redraw();
                    return;
                }
                Err(_) => {
                    //Todo. Many of these errors are recoverable!
                    panic!("Surface image acquire failed!");
                }
                Ok(r) => r,
            };
        let commands = self.egui_ctx.build_commands(idx);
        if let None = commands {
            return;
        }

        //Minimize heap allocs (by using excessive code duplication)
        {
            let now = self.render_context.now();
            if let Some((transfer, draw)) = commands {
                if let Some(transfer) = transfer {
                    //Transfer + Draw
                    now.then_execute(
                        self.render_context.queues().transfer().queue().clone(),
                        transfer,
                    )
                    .unwrap()
                    .join(image_future)
                    .then_signal_semaphore()
                    .then_execute(
                        self.render_context.queues().graphics().queue().clone(),
                        draw,
                    )
                    .unwrap()
                    .boxed()
                } else {
                    //Just draw
                    now.join(image_future)
                        .then_execute(
                            self.render_context.queues().graphics().queue().clone(),
                            draw,
                        )
                        .unwrap()
                        .boxed()
                }
            } else {
                //Nothing to do - shouldn't happen (Egui won't request a redraw without anything to draw)
                //but we've already acquired the image, we have to present it when it becomes ready.
                now.join(image_future).boxed()
            }
        }
        .then_swapchain_present(
            self.render_context
                .queues()
                .present()
                .unwrap()
                .queue()
                .clone(),
            vk::SwapchainPresentInfo::swapchain_image_index(
                self.render_surface().swapchain().clone(),
                idx,
            ),
        )
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

        if suboptimal {
            self.recreate_surface()
                .expect("Recreating suboptimal swapchain failed spectacularly");
        }
    }
}
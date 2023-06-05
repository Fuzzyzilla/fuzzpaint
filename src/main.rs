use std::sync::Arc;

pub mod vulkano_prelude;
pub mod gpu_err;
mod egui_impl;
use gpu_err::GpuResult;
use vulkano_prelude::*;
pub mod render_device;

use anyhow::Result as AnyResult;

enum RemoteResource {
    Resident,
    Recoverable,
    Gone,
}




pub struct WindowSurface {
    event_loop : winit::event_loop::EventLoop<()>,
    win : Arc<winit::window::Window>,
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
    pub fn with_render_surface(self, render_surface: render_device::RenderSurface, render_context: Arc<render_device::RenderContext>)
        -> GpuResult<WindowRenderer> {
        
        let mut egui_ctx = egui_impl::EguiCtx::new(&render_surface)?;
        Ok(
            WindowRenderer {
                win: self.win,
                render_surface: Some(render_surface),
                render_context,
                event_loop: Some(self.event_loop),
                egui_ctx,
            }
        )
    }
}

pub struct WindowRenderer {
    event_loop : Option<winit::event_loop::EventLoop<()>>,
    win : Arc<winit::window::Window>,
    /// Always Some. This is to allow it to be take-able to be remade.
    /// Could None represent a temporary loss of surface that can be recovered from?
    render_surface : Option<render_device::RenderSurface>,
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
        let new_surface = self.render_surface
            .take()
            .unwrap()
            .recreate(Some(self.window().inner_size().into()))?;

        self.egui_ctx.replace_surface(&new_surface);

        self.render_surface = Some(new_surface);

        Ok(())
    }
    fn apply_platform_output(&mut self, out: &mut egui::PlatformOutput) {
        //Todo: Copied text
        if let Some(url) = out.open_url.take() {
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
        let mut egui_out = None::<egui::FullOutput>;

        let mut edit_str = String::new();

        event_loop.run(move |event, _, control_flow|{
            use winit::event::{Event, WindowEvent};

            self.egui_ctx.push_winit_event(&event);

            match event {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = winit::event_loop::ControlFlow::Exit;
                            return;
                        }
                        WindowEvent::Resized(..) => {
                            self.recreate_surface().expect("Failed to rebuild surface");
                        }
                        _ => ()
                    }
                }
                Event::MainEventsCleared => {
                    //Draw!
                    let raw_input = self.egui_events.take_raw_input();
                    let mut out = self.do_ui(raw_input);

                    if out.repaint_after.is_zero() {
                        //Immediate redraw requested
                        //Don't need to worry about double-request, Winit will coalesce them
                        self.window().request_redraw()
                    } else {
                        //Egui returns astronomically large number if it doesn't want a redraw - triggers overflow lol
                        let requested_instant = std::time::Instant::now().checked_add(out.repaint_after);

                        println!("Requested time {requested_instant:?}");

                        if let Some(instant) = requested_instant {
                            //Insert sorted
                            match self.requested_redraw_times.binary_search(&instant) {
                                Ok(..) => (), //A redraw is already scheduled for this exact instant
                                Err(pos) => self.requested_redraw_times.insert(pos, instant)
                            }
                        }
                    }

                    self.apply_platform_output(&mut out.platform_output);

                    //There was old data, make sure we don't lose the texture updates when we replace it.
                    if let Some(old_output) = egui_out.take() {
                        prepend_textures_delta(&mut out.textures_delta, old_output.textures_delta);
                    }
                    egui_out = Some(out);
                }
                Event::RedrawRequested(..) => {
                    //Ensure there is actually data for us to draw
                    if let Some(out) = egui_out.take() {
                        self.paint(out);
                    }
                }
                _ => ()
            }
        });
    }
    fn do_ui(&mut self, input : egui::RawInput) -> egui::FullOutput {
        self.egui_ctx.begin_frame(input);

        egui::Window::new("🐑 Baa")
            .show(&self.egui_ctx, |ui| {
                ui.label("Testing testing 123!!");
            });

        egui::Window::new("Beep boop")
            .show(&self.egui_ctx, |ui| {
                ui.label("Testing testing 345!!");
            });
    
        //Return platform output and shapes.
        self.egui_ctx.end_frame()
    }
    fn paint(&mut self, egui_data : egui::FullOutput) {
        let (idx, suboptimal, image_future) =
            match vk::acquire_next_image(
                self.render_surface().swapchain().clone(),
                None
            ) {
                Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                    eprintln!("Swapchain unusable.. Recreating");
                    //We cannot draw on this surface as-is. Recreate and request another try next frame.
                    self.recreate_surface().expect("Failed to recreate render surface after it went out-of-date.");
                    self.window().request_redraw();
                    return;
                }
                Err(_) => {
                    //Todo. Many of these errors are recoverable!
                    panic!("Surface image acquire failed!");
                }
                Ok(r) => r
            };
        let res : AnyResult<()> = try_block::try_block! {
            let transfer_commands = self.egui_renderer.do_image_deltas(egui_data.textures_delta);

            let transfer_queue = self.render_context.queues().transfer().queue();
            let render_queue = self.render_context.queues().graphics().queue();
            let mut future : Box<dyn vk::sync::GpuFuture> = self.render_context.now().boxed();
            if let Some(transfer_commands) = transfer_commands {
                let buffer = transfer_commands?;

                //Flush transfer commands, while we tesselate Egui output.
                future = future.then_execute(transfer_queue.clone(), buffer)?
                    .boxed();
            }
            let tess_geom = self.egui_ctx.tessellate(egui_data.shapes);
            let draw_commands = self.egui_renderer.upload_and_render(idx, &tess_geom)?;
            drop(tess_geom);

            future.then_execute(render_queue.clone(),draw_commands)?
                .join(image_future)
                .then_swapchain_present(
                    self.render_context.queues().present().unwrap().queue().clone(),
                    vk::SwapchainPresentInfo::swapchain_image_index(self.render_surface().swapchain().clone(), idx),
                )
                .then_signal_fence_and_flush()?
                .wait(None)?;

           // let draw_commands = self.egui_renderer.upload_and_render(idx, tesselated_geom);
            Ok(())
        };
        if suboptimal {
            self.recreate_surface().expect("Recreating suboptimal swapchain failed spectacularly");
        }
    }
}

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    env_logger::init();

    let window_surface = WindowSurface::new()?;
    let (render_context, render_surface) = render_device::RenderContext::new_with_window_surface(&window_surface)?;
    let window_renderer = window_surface.with_render_surface(render_surface, render_context)?;
    println!("Made render context!");

    window_renderer.run();
}

use std::sync::Arc;

mod egui_impl;
pub mod gpu_err;
pub mod vulkano_prelude;
pub mod window;
use gpu_err::GpuResult;
use vulkano_prelude::*;
pub mod render_device;
pub mod stylus_events;

use anyhow::Result as AnyResult;

//If we return, it was due to an error.
//convert::Infallible is a quite ironic name for this useage, isn't it? :P
fn main() -> AnyResult<std::convert::Infallible> {
    env_logger::init();

    let window_surface = window::WindowSurface::new()?;
    let (render_context, render_surface) =
        render_device::RenderContext::new_with_window_surface(&window_surface)?;
    let window_renderer = window_surface.with_render_surface(render_surface, render_context)?;
    println!("Made render context!");

    window_renderer.run();
}

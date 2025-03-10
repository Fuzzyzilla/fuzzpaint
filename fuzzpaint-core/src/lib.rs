#![deny(unsafe_op_in_unsafe_fn)]
#![feature(write_all_vectored)]
#![warn(clippy::pedantic)]

pub mod blend;
pub mod brush;
pub mod color;
pub mod commands;
pub mod id;
pub mod io;
pub mod queue;
pub mod repositories;
pub mod state;
pub mod stroke;
pub mod units;
pub mod util;

use id::FuzzID;

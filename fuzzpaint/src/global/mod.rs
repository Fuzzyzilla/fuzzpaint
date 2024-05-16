//! Global singletons.

pub mod hotkeys;
mod provider;

pub use provider::provider;

use fuzzpaint_core::repositories::{brushes::Brushes, fonts::Faces, points::Points};

/// Get the shared global instance of the point repository.
pub fn points() -> &'static Points {
    static REPO: std::sync::OnceLock<Points> = std::sync::OnceLock::new();
    REPO.get_or_init(Points::default)
}

pub fn faces() -> &'static Faces {
    static ONCE: std::sync::OnceLock<Faces> = std::sync::OnceLock::new();
    ONCE.get_or_init(Faces::new_system)
}

pub fn brushes() -> &'static Brushes {
    static ONCE: std::sync::OnceLock<Brushes> = std::sync::OnceLock::new();
    ONCE.get_or_init(Brushes::new)
}

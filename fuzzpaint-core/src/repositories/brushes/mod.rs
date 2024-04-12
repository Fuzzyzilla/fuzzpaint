//! # Brushes and Brush textures

use crate::brush::{self, UniqueID};

#[derive(Default)]
pub struct Brushes {
    brushes: std::collections::HashMap<UniqueID, ()>,
    textures: std::collections::HashMap<UniqueID, ()>,
}
impl Brushes {
    pub fn new() -> Self {
        Self::default()
    }
}

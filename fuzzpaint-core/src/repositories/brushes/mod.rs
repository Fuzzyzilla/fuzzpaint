//! # Brushes and Brush textures

use crate::brush::{self, UniqueID};

/// Metadata about *this installation* of a brush/texture resource.
pub struct RetainedMetadata {
    /// A name the user has given this resource, over it's original name.
    user_alias: Option<String>,
    /// The time the resource was last read or written. Defaults to `installed.`
    last_accessed: chrono::DateTime<chrono::offset::Utc>,
    /// The time the resource was last explicitly used by the user.
    last_used: Option<chrono::DateTime<chrono::offset::Utc>>,
    /// The time that the resource was interned.
    installed: chrono::DateTime<chrono::offset::Utc>,
}

/// A collection of brushes. This is because the brush retention system has several layers -
/// temporary imports from opened files that the user *doesn't* want to retain to disk,
/// work-in-progress ones (created in an unsaved doc), main library from disk...
///
#[derive(Default)]
struct BrushSet {
    // These ought to use a custom hasher which just truncates the UniqueID to 64 bit. The key is already
    // a high-quality hash, to hash it again will make the quality *worse* in all likelihood :P
    brushes: std::collections::HashMap<UniqueID, ()>,
    textures: std::collections::HashMap<UniqueID, ()>,
}

#[derive(Default)]
pub struct Brushes {
    primary: BrushSet,
}
impl Brushes {
    pub fn new() -> Self {
        Self::default()
    }
}

//! # Brushes and Brush textures

use crate::brush::{self, UniqueID, UniqueIDMap};

/// Metadata about *this installation* of a brush/texture resource.
pub struct RetainedMetadata {
    /// A name the user has given this resource, over it's original name.
    pub user_alias: Option<String>,
    /// The time the resource was last read or written. Defaults to `installed.`
    pub last_accessed: chrono::DateTime<chrono::offset::Utc>,
    /// The time the resource was last explicitly used by the user.
    pub last_used: Option<chrono::DateTime<chrono::offset::Utc>>,
    /// The time that the resource was interned.
    pub installed: chrono::DateTime<chrono::offset::Utc>,
}

/// A collection of brushes. This is because the brush retention system has several layers -
/// temporary imports from opened files that the user *doesn't* want to retain to disk,
/// work-in-progress ones (created in an unsaved doc), main library from disk...
#[derive(Default)]
struct BrushSet {
    // Since the key is a high quality hash already, use a custom no-op hasher.
    brushes: UniqueIDMap<RetainedMetadata>,
    textures: UniqueIDMap<&'static [u8]>,
}

#[derive(Default)]
pub struct Brushes {
    primary: BrushSet,
}
impl Brushes {
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }
    #[must_use]
    pub fn new() -> Self {
        const DEFAULT: &[u8] =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/default/circle.png"));

        // Single threaded- this should not be in the lib, it should be in the client where threading is possible.
        // this is just placeholder.
        let id = blake3::hash(DEFAULT);

        let mut this = Self::empty();
        this.primary.textures.insert(id.into(), DEFAULT);

        this
    }
    #[must_use]
    pub fn iter_textures(&self) -> std::collections::hash_map::Iter<'_, UniqueID, &'static [u8]> {
        self.primary.textures.iter()
    }
}

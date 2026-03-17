//! Metadata about the document, not necessary for rendering.

pub struct Meta {
    exif: Exif,
    sofware: Option<Software>,
    users: Vec<User>,
}
impl Meta {
    pub fn created(&self) -> () {
        self.users
            .iter()
            .map(|user| user.joined)
            .min()
            .unwrap_or_default()
    }
    pub fn last_modified(&self) -> () {
        self.users
            .iter()
            .map(|user| user.last_active)
            .max()
            .unwrap_or_default()
    }
}
pub struct User {
    /// A cryptographic public key that the remote can verify their identity
    /// against.
    ///
    /// Public keys are unique per-document, and do not necessarily correspond
    /// between different documents. User with key ABC on document 1 may or may
    /// not be the same as user with key XYZ on document 2. This is done so that
    /// a compromised private key is of extremely limited scope, but comes with
    /// the limitation that identities cannot be verified cross-documents.
    ///
    /// `None` is provided for local documents, where there is only one user and
    /// thus the security considerations are non existent (and maintaining an
    /// external per-document private keystore would be unwieldy and error
    /// prone)
    ///
    /// Unique, including that there can only be one `None`.
    public_key: Option<()>,

    /// A user-selected and unverified display name. Not necessarily unique. If
    /// multiple users of the same name exist with different public keys, a
    /// warning should be displayed.
    name: Option<String>,

    /// Datetime when the user first touched the document.
    joined: (),
    /// Datetime when the user last touched the document.
    last_active: (),
    /// Cumulative activity time, subject to unspecified AFK filtering.
    active_time: (),
}
pub struct Exif {
    values: std::collections::HashMap<ChunkID, Vec<u8>>,
}
type ChunkID = ();
pub struct Software {
    name: String,
    version: (),
    platform: (),
    compiled: (),
    commit: (),
}

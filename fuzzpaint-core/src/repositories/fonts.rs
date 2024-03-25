//! # Font Faces
//!
//! Each conbination of `(face, variation)` is given a unique process-local ID.
//! This interns certain abstractions, such as loading embedded fonts from opened documents as if
//! they were any other font.

#![allow(dead_code)]

use rustybuzz::ttf_parser;
pub struct FaceIDMarker;
pub type FaceID = crate::FuzzID<FaceIDMarker>;
pub struct VariedFontIDMarker;
pub type VariedFaceID = crate::FuzzID<VariedFontIDMarker>;

pub enum EitherFace {
    Default(FaceID),
    Varied(VariedFaceID),
}

/// Location of fonts that the user has chosen to keep from previously opened documents.
///
/// Always read this dir when enumerating fonts. When saving fonts, try the users [`shared_fonts`] first.
/// *It is illegal (perhaps literally) to write a font not tagged `Installable` here.*
/// Installing fonts is an OS-dependent act and not guaranteed to work, so this is the fallback.
#[must_use]
pub fn local_fonts() -> Option<std::path::PathBuf> {
    // *Not* LocalData, as we'd like this data to be available on Windows network logins.
    let mut data = dirs::data_dir()?;
    data.push("/fuzzpaint/fonts/");
    Some(data)
}
/// Location of system fonts. Not checked to be writable.
/// *Do not read from here*, use a font enumeration library.
///
/// *It is illegal (perhaps literally) to write a font not tagged `Installable` here.*
/// Write fonts which the user has chosen to keep here. Fallback on [`local_fonts`] if directory permissions fail.
#[must_use]
pub fn shared_fonts() -> Option<std::path::PathBuf> {
    dirs::font_dir()
}

/// Based on the `fsType` field of OTF/os2. Does not correspond exactly with those flags, instead is a higher-level
/// view of the properties and how fuzzpaint is allowed to interact with them.
///
/// Describes the licensing restriction of a font in increasing permissiveness. The comment description is NOT legal
/// advice, but [Microsoft's](https://learn.microsoft.com/en-us/typography/opentype/spec/os2#fst) is probably closer.
#[derive(Copy, Clone, Debug)]
pub enum EmbedRestrictionLevel {
    /// The font data cannot be embedded for our purposes.
    ///
    /// We can still embed a *reference* (name, family, ver) to it, in the hopes that a user opening
    /// this with the restrictive fonts installed locally will still work.
    LocalOnly,
    /// The font data may be embedded into a read-only fuzzpaint document - no part of the document
    /// can be modified on a system that did not author it as long as this font is embedded.
    /// The wording in Microsoft's document is permissive enough to allow opening the document in a writable mode,
    /// as long as the font data is not loaded in this process and is discarded.
    ReadOnly,
    /// The font data may be embedded into an editable fuzzpaint document.
    Writable,
    /// The font data may be extracted from the document by an end user, for permanent use on their system.
    Installable,
}
#[derive(Copy, Clone, Debug)]
pub struct EmbedRestriction {
    /// Can we save size by embedding just the used subset?
    /// (we don't even have this capability yet and it seems unlikely I'll do it any time soon
    /// + would interfere with document sharability)
    ///
    /// Meaningless if `level == LocalOnly`, but still parsed.
    can_subset: bool,
    /// To what extent are we allowed to embed this font?
    /// Bitmap-only embeds are considered Non-embeddable, as that defeats the whole point.
    level: EmbedRestrictionLevel,
}
impl EmbedRestriction {
    // I originally wrote all of the bitparsing logic for all four versions, before realizing
    // ttf_parser exposes all the fields itself. Whoops!
    /// Extract the embed restrictions from the given table, or None if table failed to parse.
    #[must_use]
    pub fn from_table(table: &rustybuzz::ttf_parser::os2::Table) -> Option<Self> {
        let permissions = table.permissions()?;
        // These flags are meaningless unless ^^ is some, that's okay!
        let can_subset = table.is_subsetting_allowed();
        // Misnomer - this checks if bit 9 is zero.
        // If bit unset, this means all data embeddable, set means bitmap only.
        let bitmap_only = !table.is_bitmap_embedding_allowed();

        let level = if bitmap_only {
            // Bitmap-only mode means there's no use in us embedding, at least for our purposes!
            // (todo: for a bitmap-only font, this is actually permissive. We don't handle those yet anyway!)
            EmbedRestrictionLevel::ReadOnly
        } else {
            use rustybuzz::ttf_parser::Permissions;
            match permissions {
                Permissions::Restricted => EmbedRestrictionLevel::LocalOnly,
                Permissions::PreviewAndPrint => EmbedRestrictionLevel::ReadOnly,
                Permissions::Editable => EmbedRestrictionLevel::Writable,
                Permissions::Installable => EmbedRestrictionLevel::Installable,
            }
        };
        Some(Self { can_subset, level })
    }
    #[must_use]
    pub fn level(&self) -> EmbedRestrictionLevel {
        self.level
    }
    /// Must also abide by `self.level()`, this method does not take it into account.
    #[must_use]
    pub fn can_subset(&self) -> bool {
        self.can_subset
    }
}

/// A font belonging to an opened document.
/// The font should be checked against the Fontdb first, and if it's available there instead
/// we can open without restrictions.
struct DocumentSource {
    // Installable = any doc can use it,
    // all others restrict usage to *this doc only.*
    restriction: EmbedRestriction,
    src_document: crate::state::DocumentID,
    // Font idx, Face idx, data (file ptr or owned?)
    todo: (),
}
enum Source {
    Fontdb(fontdb::ID),
    // Will be transformed into a fontdb source if permissions check succeeds.
    Document(DocumentSource),
}
struct FaceVariation {
    face: FaceID,
    variations: smallvec::SmallVec<[ttf_parser::Variation; 32]>,
}

/// Global Font singleton. Manages Fonts, Faces, Families, and their variations.
pub struct Faces {
    /// Lazily filled with FaceIDs as they are requested.
    face_sources: hashbrown::HashMap<FaceID, Source>,
    /// Lazily created variations of FaceIDs as they are requested.
    variations: hashbrown::HashMap<VariedFaceID, FaceVariation>,
    db: fontdb::Database,
}
#[derive(thiserror::Error, Debug)]
pub enum FaceError {
    // `Display` puts the namespace already lol
    #[error("{} is unknown", .0)]
    UnknownID(FaceID),
    /// Some fonts can "belong" to a single document due to licensing.
    #[error("font private to another document")]
    Private,
    #[error("parsing error")]
    BadFace,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
impl Faces {
    /// Create the database from predefined system font folders, and the fuzzpaint local fonts.
    pub fn new_system() -> Self {
        // Font DB has no way to enumerate fonts, meaning a user selection pane is impossible.
        // need something else! TwT
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        if let Some(locals) = local_fonts() {
            // This directory may not even exist. Fails silently (desired behavior)
            db.load_fonts_dir(locals);
        }
        Self {
            face_sources: hashbrown::HashMap::new(),
            variations: hashbrown::HashMap::new(),
            db,
        }
    }
    /// Get an existing varied face. Optional document id is used to check ownership,
    /// will fail with `Private` if needed and not provided.
    pub fn get_varied(
        &self,
        _face: VariedFaceID,
        _document: Option<crate::state::DocumentID>,
    ) -> Result<(), FaceError> {
        todo!()
    }
    /// Clear all private (non-installable) fonts belonging to this document.
    pub fn clear_private(&self, _document: crate::state::DocumentID) {
        todo!()
    }
}

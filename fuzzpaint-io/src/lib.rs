pub struct LoadOptions {
    /// If true, allow extra space in the document's collections, which will
    /// make edits more efficient. Otherwise, attempt to allocate as little as
    /// possible.
    overcommit: bool,
    /// If true, provide the decoded thumbnail.
    load_thumbnail: bool,
    /// If false, skip loading the history tree. Items not used in the current
    /// history state *may* not be loaded.
    load_history: bool,
    /// If true, eagerly load all embedded resources.
    load_embedded_resources: bool,
    /// If true, allow loading of file-path based resources.
    /// **THIS HAS MAJOR SECURITY IMPLICATIONS** tread with caution.
    allow_local_files: bool,
}

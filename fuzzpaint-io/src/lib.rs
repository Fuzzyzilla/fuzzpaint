pub struct LoadOptions {
    /// If true, allow extra space in the document's collections, which will
    /// make edits more efficient. Otherwise, attempt to allocate as little as
    /// possible.
    overcommit: bool,
    /// If true, provide the decoded thumbnail.
    load_thumb: bool,
    /// If false, skip loading the history tree.
    load_history: bool,
}

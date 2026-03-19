pub type ID = crate::FuzzID<Document>;

#[derive(Clone)]
pub struct Document {
    /// The path from which the file was loaded or saved, or None if opened as new.
    pub path: Option<std::path::PathBuf>,
    /// Name of the document, inferred from its path or generated.
    pub name: String,
}
impl Default for Document {
    fn default() -> Self {
        Self {
            path: None,
            name: "New Document".into(),
        }
    }
}

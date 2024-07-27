pub type ID = crate::FuzzID<Document>;

#[derive(Clone)]
pub struct Document {
    /// The path from which the file was loaded or saved, or None if opened as new.
    pub path: Option<std::path::PathBuf>,
    /// Name of the document, inferred from its path or generated.
    pub name: String,
    pub viewport: Viewport,
}
impl Default for Document {
    fn default() -> Self {
        Self {
            path: None,
            name: "New Document".into(),
            viewport: Viewport::default(),
        }
    }
}

#[derive(Copy, Clone)]
/// The render area of a document.
pub struct Viewport {
    /// Where the top-left corner of the document is located in global space.
    pub origin: [crate::units::Length; 2],
    /// The size of the document area, extending down-right.
    pub size: [crate::units::Length; 2],
    /// Controls the interpretation of physical units into pixels and vice-versa.
    pub resolution: crate::units::Resolution,
    /// Controls the default ratio of physical pixels per logical pixel.
    pub scale_factor: f32,
}
impl Viewport {
    /// Get the resolution (DPI/DPCM) after `scale_factor` is applied
    #[must_use]
    pub fn scaled_resolution(&self) -> crate::units::Resolution {
        let mut res = self.resolution;
        *res.value_mut() *= self.scale_factor;
        res
    }
    /// Calculate the center of the viewport, in the same unit as [`Self::origin`]
    #[must_use]
    pub fn center(&self) -> [crate::units::Length; 2] {
        [
            self.origin[0].add(self.size[0] / 2.0, self.resolution),
            self.origin[1].add(self.size[1] / 2.0, self.resolution),
        ]
    }
    /// Get the offset of the top-left of the viewport, in logical pixels.
    #[must_use]
    pub fn origin_logical_pixels(&self) -> [f32; 2] {
        self.origin
            .map(|length| length.into_logical(self.resolution))
    }
    /// Get the size of the viewport, in logical pixels.
    #[must_use]
    pub fn size_logical_pixels(&self) -> [f32; 2] {
        self.size.map(|length| length.into_logical(self.resolution))
    }
    /// Get the size of the viewport, in rounded physical pixels.
    #[must_use]
    pub fn size_physical_pixels(&self) -> [u32; 2] {
        self.size_logical_pixels()
            .map(|logical| logical * self.scale_factor)
            .map(|physical| physical.round() as u32)
    }
}
impl Default for Viewport {
    fn default() -> Self {
        Self {
            origin: [crate::units::Length::Logical(0.0); 2],
            // Totally arbitrary choice based on my preference >:3c
            size: [crate::units::Length::Logical(1080.0); 2],
            resolution: crate::units::Resolution::Dpi(150.0),
            scale_factor: 1.0,
        }
    }
}

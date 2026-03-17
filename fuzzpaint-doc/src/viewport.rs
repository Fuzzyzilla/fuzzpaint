/// Describes a finite, axis-aligned rectangular view of the infinite virtual
/// canvas.
pub struct Viewport {
    dpi: fuzzpaint_types::dpi::Dpi,
    rect: fuzzpaint_types::dpi::Rect,
}
impl Viewport {
    /// Calculate the physical size of the viewport, in pixels, rounded up.
    pub fn physical_size(&self) -> ultraviolet::UVec2 {
        self.rect
            .size
            .into_physical_px(&self.dpi)
            .into_inner()
            .as_array()
            .map(|f| f.ceil() as u32)
            .into()
    }
}

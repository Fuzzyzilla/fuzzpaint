/// Dynamically scale and position the whole document into the viewport.
#[derive(Copy, Clone, Default)]
pub struct Fit {
    pub flip_h: bool,
}
/// View the document at some multiple of pixel-perfect (i.e. when one document
/// pixel corresponds to one screen pixel).
///
/// One-to-one *physical units* (like cm) doesn't seem possible with Winit, as
/// it does not have a physical DPI concept.
///
/// It isn't clear to me which coordinate systems to use on *either* side of this
/// transform. Physical -> physical, logical -> logical, or physical ->
/// logical???
#[derive(Copy, Clone)]
pub struct PixelPerfect {
    pub similarity: fuzzpaint_types::similarity::Similarity,
}

#[derive(Copy, Clone)]
pub enum View {
    Fit(Fit),
    PixelPerfect(PixelPerfect),
}
impl View {
    /// Horizontally flip the viewport, maintaining the view kind.
    pub fn flip_h_around(&mut self, viewport_center_x: f32) {
        match self {
            Self::Fit(Fit { flip_h }) => *flip_h = !*flip_h,
            Self::PixelPerfect(PixelPerfect { similarity }) => {
                similarity.flip_h_around(viewport_center_x);
            }
        }
    }
    #[must_use = "returns a new transform and doesn't modify `self`"]
    pub fn into_pixel_perfect_similarity(
        &self,
        document: fuzzpaint_types::dpi::UnitlessRect,
        viewport: fuzzpaint_types::dpi::UnitlessRect,
    ) -> fuzzpaint_types::similarity::Similarity {
        match *self {
            Self::Fit(Fit { flip_h }) => {
                // Find the tightest axis, and scale based on that.
                let scale = (viewport.size / document.size).component_min();
                let consumed_size = document.size * scale;
                let translation =
                    // Place the top-left of the document at the top-left of the
                    // viewport
                    viewport.origin - document.origin
                    // Then center it within the viewport by shifting it by half
                    // of the leftover margin.
                    + (viewport.size - consumed_size) / 2.0;

                let mut xform = fuzzpaint_types::similarity::Similarity::from_parts(
                    false,
                    scale,
                    0.0,
                    translation,
                );

                if flip_h {
                    // Flip around the centre of the viewport.
                    xform.flip_h_around(viewport.size.x / 2.0 + viewport.origin.x);
                }
                xform
            }
            Self::PixelPerfect(PixelPerfect { similarity }) => similarity,
        }
    }
    /// Convert self into the [`Self::PixelPerfect`] variant, and return a
    /// reference to it.
    pub fn as_pixel_perfect_mut(
        &mut self,
        document: fuzzpaint_types::dpi::UnitlessRect,
        viewport: fuzzpaint_types::dpi::UnitlessRect,
    ) -> &mut PixelPerfect {
        *self = Self::PixelPerfect(PixelPerfect {
            similarity: self.into_pixel_perfect_similarity(document, viewport),
        });
        let Self::PixelPerfect(pixel_perfect) = self else {
            unreachable!();
        };
        pixel_perfect
    }
}

impl Default for View {
    fn default() -> Self {
        Self::Fit(Fit::default())
    }
}
#[derive(Clone, Copy)]
pub struct ViewInfo {
    pub transform: View,
    // Logical pixels.
    pub viewport: fuzzpaint_types::dpi::UnitlessRect,
    // idk if logical or physical pixels. FIXME :3
    pub document: fuzzpaint_types::dpi::UnitlessRect,
}
impl ViewInfo {
    #[must_use]
    pub fn center(&self) -> ultraviolet::Vec2 {
        self.viewport.center()
    }
    #[must_use]
    pub fn into_pixel_perfect_similarity(&self) -> fuzzpaint_types::similarity::Similarity {
        self.transform
            .into_pixel_perfect_similarity(self.document, self.viewport)
    }
    /// Convert self in-place into a `ViewTransform` representation, returning mutable access to that transform.
    /// `None` if too small to be usable.
    pub fn as_pixel_perfect_mut(&mut self) -> &mut PixelPerfect {
        self.transform
            .as_pixel_perfect_mut(self.document, self.viewport)
    }
}

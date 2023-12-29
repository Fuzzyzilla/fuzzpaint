use super::interface::Vertex;
use rustybuzz::ttf_parser;

impl From<lyon_tessellation::FillVertex<'_>> for Vertex {
    fn from(value: lyon_tessellation::FillVertex) -> Self {
        // The only data we care about is position!
        let lyon_tessellation::math::Point { x, y, .. } = value.position();
        Self { position: [x, y] }
    }
}

/// Consumes Lyon tessellator events into a generic vertex + 16bit index buffer.
struct LyonVertexBuilder<'data, Vertex> {
    vertices: &'data mut Vec<Vertex>,
    indices: &'data mut Vec<u16>,
    // Past-the-end indices for when the last begin_geometry was called.
    begin_index_pos: usize,
    begin_vertex_pos: u16,
}
impl<'data, Vertex> LyonVertexBuilder<'data, Vertex> {
    fn new(vertices: &'data mut Vec<Vertex>, indices: &'data mut Vec<u16>) -> Self {
        Self {
            vertices,
            indices,
            begin_index_pos: 0,
            begin_vertex_pos: 0,
        }
    }
}
impl<Vertex> lyon_tessellation::GeometryBuilder for LyonVertexBuilder<'_, Vertex> {
    fn begin_geometry(&mut self) {
        self.begin_index_pos = self.indices.len();
        // Should bail before saturation is needed. The geometry is invalid if
        // we go past this anyway, so the wrapping results in complete chaos which is fine.
        self.begin_vertex_pos = az::wrapping_cast(self.indices.len());
    }
    fn add_triangle(
        &mut self,
        a: lyon_tessellation::VertexId,
        b: lyon_tessellation::VertexId,
        c: lyon_tessellation::VertexId,
    ) {
        use az::CheckedAs;
        // Lyon promises these are all from within the current begin_geometry..end_geometry scope
        // We make safety assumptions about this later, so make sure this is really the case:
        // (we might be able to ditch the lower bound check, FIXME when i get there :3)
        let Some(a) = a.0.checked_as() else { return };
        let Some(b) = b.0.checked_as() else { return };
        let Some(c) = c.0.checked_as() else { return };
        // Inclusive lower bound of valid indices
        let min = self.begin_vertex_pos;
        // Find inclusive upper bound or bail
        let Some(max) = self
            .vertices
            .len()
            .checked_sub(1)
            .and_then(CheckedAs::checked_as)
        else {
            return;
        };

        // No way to report errors, but if any vertex is out-of-range ignore the whole tri.
        if min <= a && min <= b && min <= c && a <= max && b <= max && c <= max {
            self.indices.extend_from_slice(&[a, b, c]);
        } else {
            // in debug mode print an error
            debug_assert!(false, "bad index requested");
        }
    }
    fn end_geometry(&mut self) {
        // Nothing to do here, `begin_geometry` handes all the work of setting up scope.
    }
    fn abort_geometry(&mut self) {
        // Clear everything written since the start of this geometry.
        // No need to reset the indices after - they're already in the right spot!
        self.vertices.drain(self.begin_vertex_pos as usize..);
        self.indices.drain(self.begin_index_pos..);
    }
}
impl<Vertex> lyon_tessellation::FillGeometryBuilder for LyonVertexBuilder<'_, Vertex>
where
    Vertex: for<'temp> From<lyon_tessellation::FillVertex<'temp>>,
{
    fn add_fill_vertex(
        &mut self,
        vertex: lyon_tessellation::FillVertex,
    ) -> Result<lyon_tessellation::VertexId, lyon_tessellation::GeometryBuilderError> {
        self.vertices.push(Vertex::from(vertex));

        // Index is the position we just inserted, make sure it fits in u16 and return!
        let index = self.vertices.len() - 1;
        u16::try_from(index)
            .map_err(|_| lyon_tessellation::GeometryBuilderError::TooManyVertices)
            .map(Into::into)
    }
}
/// Newtype to allow lyon tess to accept TTF path events.
struct LyonBridge<'builder>(
    lyon_tessellation::path::builder::NoAttributes<lyon_tessellation::FillBuilder<'builder>>,
);
impl ttf_parser::OutlineBuilder for LyonBridge<'_> {
    fn close(&mut self) {
        self.0.close();
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        use lyon_tessellation::math::point;
        self.0
            .cubic_bezier_to(point(x1, y1), point(x2, y2), point(x, y));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        use lyon_tessellation::math::point;
        self.0.quadratic_bezier_to(point(x1, y1), point(x, y));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        use lyon_tessellation::math::point;
        self.0.line_to(point(x, y));
    }
    fn move_to(&mut self, x: f32, y: f32) {
        use lyon_tessellation::math::point;
        // This has a precondition we cannot verify!
        self.0.begin(point(x, y));
    }
}

#[derive(thiserror::Error, Debug)]
pub enum TessellateError {
    #[error("glyph not found")]
    GlyphNotFound,
    #[error(transparent)]
    Lyon(#[from] lyon_tessellation::TessellationError),
}
/// Tessellate glyph without color data. The given scale class is relative to the font's default size.
/// Vectors and tessellator are passed in as arguments to amortize construction expenses over several calls.
///
/// If Ok, every index in `indices` is guaranteed to be `< vertices.len()`
/// # Errors
/// In the event of an error, the vertex and index buffers are inconsistent and should not be used.
pub fn tessellate_glyph(
    face: &ttf_parser::Face<'_>,
    glyph: ttf_parser::GlyphId,
    size_class: super::SizeClass,
    tessellator: &mut lyon_tessellation::FillTessellator,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u16>,
) -> Result<(), TessellateError> {
    // make sure things are cleaned up
    vertices.clear();
    indices.clear();

    let mut options = lyon_tessellation::FillOptions::non_zero();
    // The tessellator works in higher res for larger size classes:
    // A double size scale class tesselates with half the tolerance.
    // With a size class of ONE leading to a tolerance of 1... mystery unit...
    // (It is up to the caller to handle unit shenanigans to acquire the scale class passed in here)
    options.tolerance = size_class.recip().scale_factor();

    let mut output = LyonVertexBuilder::new(vertices, indices);
    let mut bridge = LyonBridge(tessellator.builder(&options, &mut output));
    face.outline_glyph(glyph, &mut bridge)
        .ok_or(TessellateError::GlyphNotFound)?;
    // Outline successfully walked!
    let LyonBridge(builder) = bridge;
    builder.build()?;

    // Assert our return condition.
    let max_index = vertices.len().saturating_sub(1);
    debug_assert!(
        u16::try_from(max_index).is_ok_and(|max| indices.iter().all(|index| index <= &max))
    );

    Ok(())
}
#[derive(thiserror::Error, Debug)]
pub enum ColorError {
    #[error("glyph not found in the COLR table")]
    /// Not found in the COLR table. May still be a valid glyph, but this is not checked.
    NoColor,
    #[error("internal error")]
    StagerError,
}
struct ColorPainter<'infos> {
    staged: Option<ttf_parser::GlyphId>,
    infos_into: &'infos mut Vec<(ttf_parser::GlyphId, super::GlyphColorMode)>,
    borked: bool,
}
impl ttf_parser::colr::Painter for ColorPainter<'_> {
    fn outline(&mut self, glyph_id: ttf_parser::GlyphId) {
        // Render this glyph, and stage it in anticipation of a color command.

        // If there was already a staged glyph, it didn't get properly colored.
        // Silently succeed and signal err at end.
        if self.staged.replace(glyph_id).is_some() {
            self.borked = true;
        }
    }
    fn paint_color(&mut self, color: ttf_parser::RgbaColor) {
        // Finish the staged glyph with the given Rgba
        if let Some(glyph) = self.staged.take() {
            self.infos_into
                .push((glyph, super::GlyphColorMode::Srgba(color)));
        } else {
            // Silently succeed and signal err at the end
            self.borked = true;
        }
    }
    fn paint_foreground(&mut self) {
        // Finish the staged glyph with the given Rgba
        if let Some(glyph) = self.staged.take() {
            self.infos_into
                .push((glyph, super::GlyphColorMode::Foreground));
        } else {
            // Silently succeed and signal err at the end
            self.borked = true;
        }
    }
}
/// Fetch color data for a `COLRv0` glyph, returning many colored subglyphs
/// Each subglyph should be (tessellated)[`tessellate_glyph`] and drawn in-order to represent the full color glyph.
///
/// Vector is passed in as argument to amortize construction expenses over several calls.
/// # Errors
/// In the event of an error (glyph missing color data), the contents of `infos_into` is meaningless.
pub fn colrv0_layers(
    face: &ttf_parser::Face<'_>,
    glyph: ttf_parser::GlyphId,
    palette: u16,
    infos_into: &mut Vec<(ttf_parser::GlyphId, super::GlyphColorMode)>,
) -> Result<(), ColorError> {
    // make sure things are cleaned up
    infos_into.clear();

    // Ruin the immaculate zerocopy api by just collecting the colors and passing them to the caller to tess.
    let mut painter = ColorPainter {
        staged: None,
        infos_into,
        borked: false,
    };
    face.paint_color_glyph(glyph, palette, &mut painter)
        .ok_or(ColorError::NoColor)
        // Fail out with `StagerError` if something went wrong while collecting.
        .and(if painter.borked {
            Err(ColorError::StagerError)
        } else {
            Ok(())
        })
}

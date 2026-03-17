pub mod layer;
pub mod meta;
pub mod strokes;
pub mod viewport;

pub struct Document {
    meta: meta::Meta,
    strokes: strokes::Strokes,
    viewport: viewport::Viewport,
}

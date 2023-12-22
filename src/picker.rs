//! Pickers allow the user to query single points at a time. Some ideas include selecting the top most stroke,
//! top layer, pick a color or brush from existing strokes, etc. Or just regular image pixel color picking!

#[derive(thiserror::Error, Debug)]
/// An error that can occur when picking.
pub enum PickError {
    #[error("sample coordinate out-of-bounds")]
    OutOfBounds,
    /// Data is available somewhere, but this picker doesn't have it.
    /// A new picker should be acquired.
    #[error("picker needs refresh to access this data")]
    NeedsRefresh,
}

pub trait Picker {
    /// What datatype does this picker yield when sampled?
    type Value;
    /// Pick at the given coordinate in the viewport. The constructor of this type must then accept a transformation
    /// matrix to convert this coordiate to whatever internal space for sampling.
    fn pick(&self, viewport_coordinate: ultraviolet::Vec2) -> Result<Self::Value, PickError>;
}

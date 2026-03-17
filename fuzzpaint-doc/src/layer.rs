pub struct Layers {}

/// Input and output data type.
pub enum PortType {
    /// An RGBA image.
    Color,
    /// An alpha-only image.
    Mask,
}

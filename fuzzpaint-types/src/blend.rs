#[derive(strum::AsRefStr, PartialEq, Eq, strum::EnumIter, Copy, Clone, Hash, Debug, Default)]
// Krita, which has the most blend modes of any software i've ever seen, only
// has ~100 blend modes. This smol repr is more than enough :3
#[repr(u8)]
pub enum BlendMode {
    // 0 is reserved for "passthrough" or "none" on nodes that support it.
    #[default]
    Normal = 1,
    Add,
    Multiply,
    Screen,
    Darken,
    Lighten,
    Erase,
}

/// Blend mode for an object, including a mode, opacity modulate, and alpha clip
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Blend {
    pub mode: BlendMode,
    pub opacity: f32,
    /// If alpha clip enabled, it should not affect background alpha, krita style!
    pub alpha_clip: bool,
}
impl Default for Blend {
    fn default() -> Self {
        Self {
            mode: BlendMode::default(),
            opacity: 1.0,
            alpha_clip: false,
        }
    }
}

#[derive(
    strum::AsRefStr,
    PartialEq,
    Eq,
    strum::EnumIter,
    Copy,
    Clone,
    Hash,
    Debug,
    /*serde::Serialize,
    serde::Deserialize,*/
)]
#[repr(u8)]
pub enum BlendMode {
    Normal,
    Add,
    Multiply,
    Overlay,
}
impl Default for BlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Blend mode for an object, including a mode, opacity modulate, and alpha clip
#[derive(Copy, Clone, Debug, PartialEq /*serde::Serialize, serde::Deserialize*/)]
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

//! # Actions API
//!
//! (certainly inspired by Godot's `Input`!)
//!
//! Provides an `ActionStream` to be written into by many asynchronous hotkey sources.
//! Many `ActionListener`s can then be attatched, which maintain their own state. These
//! listeners provide `ActionFrame`s that describe, on a per-listener basis, which actions
//! have been pressed, held, repeated, ect. since the last time it was listened to.

use std::sync::Arc;

pub mod hotkeys;

#[derive(
    serde::Serialize,
    serde::Deserialize,
    Hash,
    PartialEq,
    Eq,
    strum::AsRefStr,
    strum::EnumIter,
    Clone,
    Copy,
    Debug,
    PartialOrd,
    Ord,
)]
pub enum Action {
    New,
    NewFromClipboard,
    Close,

    Undo,
    Redo,

    ViewportPan,
    ViewportScrub,
    ViewportRotate,
    ViewportFlipHorizontal,

    ZoomIn,
    ZoomOut,

    Picker,
    Brush,
    Erase,

    Lasso,
    Marquee,

    BrushSizeUp,
    BrushSizeDown,

    ColorSwap,

    LayerUp,
    LayerDown,
    LayerNew,
    LayerDelete,

    Transform,
    FreeTranform,

    Text,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActionEvent {
    Press,
    /// The action was held long enough that the key is being strobed by the OS.
    Repeat,
    Release,

    /// The action was pressed, but another action with the same keys but stricter modifiers overwrote it.
    /// For example, the key sequence Ctrl + S + Shift could result in:
    ///
    /// * Press Save
    ///   * Shadow Save
    ///     * Press Save as
    ///     * Release Save as
    ///   * Unshadow Save
    /// * Release Save
    ///
    /// It is up to the listener to figure out what the user meant x3
    /// Typically a (non-holding) action that has ever been shadowed should be ignored.
    Shadowed,
    Unshadowed,
}

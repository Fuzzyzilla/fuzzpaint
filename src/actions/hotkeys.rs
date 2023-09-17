//! Collections of hotkeys, for keyboard, mouse, pad, and pen.
//! 
//! Actions can have potentially many hotkeys bound to them, and hotkeys can be bound to at most one action.
//! Mapping in both directions is useful, but for disk storage the one-to-many relation of Actions to keys is
//! easier to edit for the end user. Thus, the reverse many-to-one mapping of keys to actions will be built dynamically.

use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct KeyboardHotkey {
    pub ctrl : bool,
    pub alt : bool,
    pub shift : bool,
    pub key : winit::event::VirtualKeyCode,
}
/// Todo: how to identify a pad across program invocations?
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PadID;
/// Todo: how to identify a pen across program invocations?
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PenID;
/// Pads are not yet implemented, but looking forward:
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PadHotkey {
    /// Which tablet does this come from? (if multiple)
    pub pad: PadID,
    /// Which layer on this pad? For pads with a mode switch key (eg. wacom PTH-451)
    pub layer: u32,
    /// Which key index?
    pub key: u32,
}
/// Pens are not yet implemented, but looking forward:
/// Allows many pens, and different functionality per-pen
/// depending on which pad it is interacting with. (wacom functionality)
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PenHotkey {
    /// Which tablet does this come from? (if multiple)
    pub pad: PadID,
    /// Which pen does this come from? (if multiple)
    pub pen: PenID,
    /// Which button index?
    pub key: u32,
}
/// A collection of many various hotkeys. Contained as Arc'd slices,
/// as it is not intended to change frequently.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct HotkeyCollection {
    pub keyboard_hotkeys: Option<Arc<[KeyboardHotkey]>>,
    pub pad_hotkeys: Option<Arc<[KeyboardHotkey]>>,
    pub pen_hotkeys: Option<Arc<[KeyboardHotkey]>>,
}

pub enum AnyHotkey {
    Key(KeyboardHotkey),
    Pad(PadHotkey),
    Pen(PenHotkey),
}
/// Maps each action onto potentially many hotkeys.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ActionsToKeys(std::collections::HashMap<super::Action, HotkeyCollection>);

/// Derived from ActionsToKeys, maps each hotkey onto at most one action.
pub struct KeysToActions(std::collections::HashMap<AnyHotkey, super::Action>);
pub struct Hotkeys {
    actions_to_keys: ActionsToKeys,
    keys_to_actions: KeysToActions,
}

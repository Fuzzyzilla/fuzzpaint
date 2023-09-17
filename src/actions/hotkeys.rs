//! Collections of hotkeys, for keyboard, mouse, pad, and pen.
//! 
//! Actions can have potentially many hotkeys bound to them, and hotkeys can be bound to at most one action.
//! Mapping in both directions is useful, but for disk storage the one-to-many relation of Actions to keys is
//! easier to edit for the end user. Thus, the reverse many-to-one mapping of keys to actions will be built dynamically.

use std::sync::Arc;

pub trait HotkeyShadow {
    type Other;
    /// Returns true if this event is "more specific" than the other.
    /// i.e., uses the same key but has stricter modifiers, or same pad differnt key.
    /// *Not assymetric* - a.shadows(b) and b.shadows(a) are both allowed to return true.
    /// In that case, it makes sense to shadow the older one and favor the new.
    fn shadows(&self, other: &Self::Other) -> bool;
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct KeyboardHotkey {
    pub ctrl : bool,
    pub alt : bool,
    pub shift : bool,
    pub key : winit::event::VirtualKeyCode,
}
impl KeyboardHotkey {
    /// Get an arbitrary score of how specific this key is - 
    /// Hotkeys with higher specificity shadow those with lower.
    pub fn specificity(&self) -> u8 {
        self.ctrl as u8 + self.alt as u8 + self.shift as u8
    }
}
impl HotkeyShadow for KeyboardHotkey {
    type Other = Self;
    fn shadows(&self, other: &Self::Other) -> bool {
        other.key == self.key && (other.specificity() <= self.specificity())
    }
}
/// Todo: how to identify a pad across program invocations?
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PadID;
/// Todo: how to identify a pen across program invocations?
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq)]
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
impl HotkeyShadow for PadHotkey {
    type Other = Self;
    fn shadows(&self, other: &Self::Other) -> bool {
        other.pad == self.pad && other.layer == self.layer
    }
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
impl HotkeyShadow for PenHotkey {
    type Other = Self;
    fn shadows(&self, other: &Self::Other) -> bool {
        other.pad == self.pad && other.pad == self.pad
    }
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
impl HotkeyShadow for AnyHotkey {
    type Other = Self;
    fn shadows(&self, other: &Self::Other) -> bool {
        match (self, other) {
            (AnyHotkey::Key(k1), AnyHotkey::Key(k2)) => k1.shadows(k2),
            (AnyHotkey::Pad(k1), AnyHotkey::Pad(k2)) => k1.shadows(k2),
            (AnyHotkey::Pen(k1), AnyHotkey::Pen(k2)) => k1.shadows(k2),
            // Different types do not shadow each other
            _ => false,
        }
    }
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

//! Collections of hotkeys, for keyboard, mouse, pad, and pen.
//!
//! Actions can have potentially many hotkeys bound to them, and hotkeys can be bound to at most one action.
//! Mapping in both directions is useful, but for disk storage the one-to-many relation of Actions to keys is
//! easier to edit for the end user. Thus, the reverse many-to-one mapping of keys to actions will be built dynamically.

use std::sync::Arc;
mod defaults;

pub trait HotkeyShadow {
    type Other;
    /// Returns true if this event is "more specific" than the other.
    /// i.e., uses the same key but has stricter modifiers, or same pad different key.
    /// *Not assymetric* - a.shadows(b) and b.shadows(a) are both allowed to return true.
    /// In that case, it makes sense to shadow the older one and favor the new.
    fn shadows(&self, other: &Self::Other) -> bool;
}

#[derive(serde::Serialize, serde::Deserialize, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub struct KeyboardHotkey {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub key: winit::event::VirtualKeyCode,
}
impl KeyboardHotkey {
    /// Get an arbitrary score of how specific this key is -
    /// Hotkeys with higher specificity shadow those with lower.
    #[must_use]
    pub fn specificity(&self) -> u8 {
        u8::from(self.ctrl) + u8::from(self.alt) + u8::from(self.shift)
    }
}
impl HotkeyShadow for KeyboardHotkey {
    type Other = Self;
    fn shadows(&self, other: &Self::Other) -> bool {
        other.key == self.key && (other.specificity() <= self.specificity())
    }
}
/// Todo: how to identify a pad across program invocations?
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy)]
pub struct PadID;
/// Todo: how to identify a pen across program invocations?
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy)]
pub struct PenID;
/// Pads are not yet implemented, but looking forward:
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy)]
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
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy)]
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
        other.pad == self.pad && other.pen == self.pen
    }
}
/// A collection of many various hotkeys. Contained as Arc'd slices,
/// as it is not intended to change frequently.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct HotkeyCollection {
    pub keyboard_hotkeys: Option<Arc<[KeyboardHotkey]>>,
    pub pad_hotkeys: Option<Arc<[PadHotkey]>>,
    pub pen_hotkeys: Option<Arc<[PenHotkey]>>,
}
impl HotkeyCollection {
    pub fn iter(&self) -> impl Iterator<Item = AnyHotkey> + '_ {
        let keyboard = self
            .keyboard_hotkeys
            .iter()
            .flat_map(|keys| keys.iter().map(|key| AnyHotkey::Key(*key)));
        let pad = self
            .pad_hotkeys
            .iter()
            .flat_map(|keys| keys.iter().map(|key| AnyHotkey::Pad(*key)));
        let pen = self
            .pen_hotkeys
            .iter()
            .flat_map(|keys| keys.iter().map(|key| AnyHotkey::Pen(*key)));

        keyboard.chain(pad).chain(pen)
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug, Copy)]
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
impl From<KeyboardHotkey> for AnyHotkey {
    fn from(value: KeyboardHotkey) -> Self {
        Self::Key(value)
    }
}
impl From<PadHotkey> for AnyHotkey {
    fn from(value: PadHotkey) -> Self {
        Self::Pad(value)
    }
}
impl From<PenHotkey> for AnyHotkey {
    fn from(value: PenHotkey) -> Self {
        Self::Pen(value)
    }
}
/// Maps each action onto potentially many hotkeys.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ActionsToKeys(hashbrown::HashMap<super::Action, HotkeyCollection>);
impl Default for ActionsToKeys {
    fn default() -> Self {
        let mut keys_map = hashbrown::HashMap::with_capacity(defaults::KEYBOARD.len());
        // Collect the keys from the defaults array
        for (action, keys) in defaults::KEYBOARD {
            keys_map.insert(
                *action,
                HotkeyCollection {
                    keyboard_hotkeys: Some((*keys).into()),
                    pad_hotkeys: None,
                    pen_hotkeys: None,
                },
            );
        }

        let new = Self(keys_map);
        // Make sure we didn't accidentally bind a single key twice
        // Would be nice if this was a static check.
        debug_assert!(TryInto::<KeysToActions>::try_into(&new).is_ok());
        new
    }
}

/// Derived from [`ActionsToKeys`], maps each hotkey onto at most one action.
pub struct KeysToActions(hashbrown::HashMap<AnyHotkey, super::Action>);
#[derive(thiserror::Error, Debug)]
pub enum KeysToActionsError {
    /// A single key was bound to multiple actions.
    /// Only the first two encountered (in arbitrary order) are reported.
    #[error("hotkey {key:?} used for more than one action: {actions:?}")]
    DuplicateBinding {
        key: AnyHotkey,
        actions: [super::Action; 2],
    },
}
impl TryFrom<&ActionsToKeys> for KeysToActions {
    type Error = KeysToActionsError;
    fn try_from(value: &ActionsToKeys) -> Result<Self, Self::Error> {
        let mut new = KeysToActions(hashbrown::HashMap::default());

        for (action, keys) in &value.0 {
            for key in keys.iter() {
                let old = new.0.insert(key, *action);
                // The slot wasn't empty!
                if let Some(old) = old {
                    return Err(KeysToActionsError::DuplicateBinding {
                        key,
                        actions: [*action, old],
                    });
                }
            }
        }

        Ok(new)
    }
}
impl KeysToActions {
    pub fn contains(&self, key: impl Into<AnyHotkey>) -> bool {
        self.0.contains_key(&key.into())
    }
    pub fn action_of(&self, key: impl Into<AnyHotkey>) -> Option<super::Action> {
        self.0.get(&key.into()).copied()
    }
}

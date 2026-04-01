//! Collections of hotkeys, for keyboard, mouse, pad, and pen.
//!
//! Actions can have potentially many hotkeys bound to them, and hotkeys can be bound to at most one action.
//! Mapping in both directions is useful, but for disk storage the one-to-many relation of Actions to keys is
//! easier to edit for the end user. Thus, the reverse many-to-one mapping of keys to actions will be built dynamically.

mod defaults;
pub mod enum_smuggler;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct Modifiers: u8 {
        const CTRL = 1;

        const ALT = 2;
        const CTRL_ALT = Modifiers::CTRL.bits() | Modifiers::ALT.bits();

        const SHIFT = 4;
        const CTRL_SHIFT = Modifiers::CTRL.bits() | Modifiers::SHIFT.bits();
        const ALT_SHIFT = Modifiers::ALT.bits() | Modifiers::SHIFT.bits();

        const CTRL_ALT_SHIFT = Modifiers::CTRL.bits() | Modifiers::ALT.bits() | Modifiers::SHIFT.bits();
    }
}
impl Modifiers {
    pub const fn ctrl_alt_shift(ctrl: bool, alt: bool, shift: bool) -> Self {
        let mut inner = 0;
        if ctrl {
            inner |= Self::CTRL.bits();
        }
        if alt {
            inner |= Self::ALT.bits();
        }
        if shift {
            inner |= Self::SHIFT.bits();
        }
        Self::from_bits_retain(inner)
    }
    pub fn ctrl(self) -> bool {
        self.intersects(Self::CTRL)
    }
    pub fn alt(self) -> bool {
        self.intersects(Self::ALT)
    }
    pub fn shift(self) -> bool {
        self.intersects(Self::SHIFT)
    }
}
#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub struct KeyboardHotkey {
    pub modifiers: Modifiers,
    pub key: winit::keyboard::KeyCode,
}
impl serde::Serialize for KeyboardHotkey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Crime #1: Use a string so that we can do human-readable formatting.
        // This can be done heapless :V
        serializer.serialize_str(&self.to_string())
    }
}
impl<'de> serde::Deserialize<'de> for KeyboardHotkey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Crime #2: Custom parse-from-string underneath the deserializer lol
        // That's a funny way to do it! :D
        // delegate to FromStr from a borrowed or owned string, depending on capabilities of deserializer.
        let str =
            <std::borrow::Cow<'de, str> as serde::Deserialize<'de>>::deserialize(deserializer)?;
        str.parse().map_err(serde::de::Error::custom)
    }
}
impl KeyboardHotkey {
    /// Get a human-readable string. This string is formatted correctly for [`std::str::FromStr`].
    #[must_use]
    pub fn to_string(&self) -> String {
        let key_name = enum_smuggler::smuggle_out(self.key).unwrap().variant;
        let mut components = smallvec::SmallVec::<[&'static str; 4]>::new();
        if self.modifiers.ctrl() {
            components.push("ctrl");
        }
        if self.modifiers.alt() {
            components.push("alt");
        }
        if self.modifiers.shift() {
            components.push("shift");
        };
        components.push(key_name);
        components.join("+")
    }
}
#[derive(Debug, thiserror::Error)]
pub enum KeyboardHotkeyFromStrError {
    // Would be nice to have a ref to the name of the key here but FromStr errors can't have lifetimes :V
    #[error("unrecognized key name")]
    InvalidKeyName,
}
/// Parse from sytax `[ctrl+][alt+][shift+]<winit key name>`, case-sensitive.
impl std::str::FromStr for KeyboardHotkey {
    type Err = KeyboardHotkeyFromStrError;
    fn from_str(mut str: &str) -> Result<Self, Self::Err> {
        let mut take_if_has = |prefix: &str| -> bool {
            if let Some(new_str) = str.strip_prefix(prefix) {
                str = new_str;
                true
            } else {
                false
            }
        };
        let mut modifiers = Modifiers::empty();
        if take_if_has("ctrl+") {
            modifiers |= Modifiers::CTRL
        };
        if take_if_has("alt+") {
            modifiers |= Modifiers::ALT
        };
        if take_if_has("shift+") {
            modifiers |= Modifiers::SHIFT
        };
        // str now contains only the key name.
        let key = enum_smuggler::smuggle_in(str)
            .map_err(|_| KeyboardHotkeyFromStrError::InvalidKeyName)?;

        Ok(Self { modifiers, key })
    }
}
/// Todo: how to identify a pad across program invocations?
#[derive(
    serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy, PartialOrd, Ord,
)]
pub struct PadID;
/// Todo: how to identify a pen across program invocations?
#[derive(
    serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy, PartialOrd, Ord,
)]
pub struct PenID;
/// Pads are not yet implemented, but looking forward:
#[derive(
    serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy, PartialOrd, Ord,
)]
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
#[derive(
    serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Clone, Debug, Copy, PartialOrd, Ord,
)]
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
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct HotkeyCollection {
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub keyboard: Vec<KeyboardHotkey>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub pad: Vec<PadHotkey>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub pen: Vec<PenHotkey>,
}
impl HotkeyCollection {
    pub fn iter(&self) -> impl Iterator<Item = AnyHotkey> + '_ {
        let keyboard = self.keyboard.iter().copied().map(AnyHotkey::Key);
        let pad = self.pad.iter().copied().map(AnyHotkey::Pad);
        let pen = self.pen.iter().copied().map(AnyHotkey::Pen);

        keyboard.chain(pad).chain(pen)
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug, Copy)]
pub enum AnyHotkey {
    Key(KeyboardHotkey),
    Pad(PadHotkey),
    Pen(PenHotkey),
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
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ActionsToKeys(pub std::collections::BTreeMap<super::Action, HotkeyCollection>);
impl Default for ActionsToKeys {
    fn default() -> Self {
        let mut keys_map = std::collections::BTreeMap::new();
        // Collect the keys from the defaults array
        for (action, keys) in defaults::KEYBOARD {
            keys_map.insert(
                *action,
                HotkeyCollection {
                    keyboard: keys.to_vec(),
                    pad: Vec::new(),
                    pen: Vec::new(),
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
impl ActionsToKeys {
    pub fn get(&self, action: super::Action) -> Option<&HotkeyCollection> {
        self.0.get(&action)
    }
}

/// Derived from [`ActionsToKeys`], maps each hotkey onto at most one action.
#[derive(Clone)]
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
        let mut new = KeysToActions(hashbrown::HashMap::new());

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

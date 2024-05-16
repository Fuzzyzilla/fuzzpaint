use crate::actions;

const DOCUMENTATION: &str = r#"# Fuzzpaint hotkeys. You may edit this file, but be aware that formatting and comments will not
# be preserved, and all keys and values are case sensitive.

# See `actions::Action` for available actions, specified here in [brackets].
# Keyboard hotkeys, specified by the "keyboard" field of an action, are case-sensitive and written `[ctrl+][alt+][shift+]<winit key code>`.
# Each action may have many hotkeys associated with it, but each hotkey should only be used at most once.
# See https://docs.rs/winit/latest/winit/keyboard/enum.KeyCode.html for a list of key codes.

# Examples:
# [Undo]
# keyboard = ["ctrl+KeyZ"]
# [Redo]
# keyboard = ["ctrl+KeyY", "ctrl+shift+KeyZ"]

"#;

#[must_use]
pub fn preferences_dir() -> Option<std::path::PathBuf> {
    let mut base_dir = dirs::preference_dir()?;
    base_dir.push(env!("CARGO_PKG_NAME"));
    Some(base_dir)
}

#[derive(Debug, thiserror::Error)]
/// If certain errors occur, we cannot automatically write new data to the file
/// (otherwise it would clobber the user's preferences, nuh uh!)
pub enum LoadBlockReason {
    /// A parse error.
    #[error("syntax error: {0}")]
    Syntax(#[from] toml::de::Error),
    /// An IO error that's *not* file-not-found.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// A logic error is present in the key specifications, e.g. one key bound to multiple.
    #[error("logic error: {0}")]
    KeysToActions(#[from] crate::actions::hotkeys::KeysToActionsError),
}

pub struct Hotkeys {
    pub load_blocker: Option<LoadBlockReason>,
    pub actions_to_keys: actions::hotkeys::ActionsToKeys,
    // Has an invariant, so private! To create one, use the `TryFrom<ActionsToKeys> for Self` impl.
    keys_to_actions: actions::hotkeys::KeysToActions,
}
impl Hotkeys {
    const FILENAME: &'static str = "hotkeys.toml";
    /// Shared read access to the global hotkeys.
    pub fn read() -> parking_lot::RwLockReadGuard<'static, Self> {
        Self::global().read()
    }
    /// Exclusive write access to the global hotkeys
    pub fn write() -> parking_lot::RwLockWriteGuard<'static, Self> {
        Self::global().write()
    }
    #[must_use]
    pub fn keys_to_actions(&self) -> &actions::hotkeys::KeysToActions {
        &self.keys_to_actions
    }
    /// Shared global hotkeys, saved and loaded from user preferences.
    /// (Or defaulted, if unavailable for some reason)
    fn global() -> &'static parking_lot::RwLock<Self> {
        static GLOBAL_HOTKEYS: std::sync::OnceLock<parking_lot::RwLock<Hotkeys>> =
            std::sync::OnceLock::new();

        GLOBAL_HOTKEYS.get_or_init(|| Self::from_default_file().into())
    }
    #[must_use]
    pub fn default_file_location() -> Option<std::path::PathBuf> {
        let mut dir = preferences_dir()?;
        dir.push(Self::FILENAME);
        Some(dir)
    }
    /// Load from the default file location.
    #[must_use]
    pub fn from_default_file() -> Self {
        Self::default_file_location()
            .as_deref()
            .map_or_else(Self::with_defaults, Self::load_or_default)
    }
    /// Load default hotkeys from static memory.
    #[must_use]
    fn with_defaults() -> Self {
        use actions::hotkeys::ActionsToKeys;
        let default = ActionsToKeys::default();
        // Default action map is reversable - this is assured by the default impl when debugging.
        let reverse = (&default).try_into().unwrap();

        Self {
            load_blocker: None,
            keys_to_actions: reverse,
            actions_to_keys: default,
        }
    }
    /// Attempts to load the settings from the given path. On file-not-found, defaults. On other error, defaults with a load-blocking message for the user.
    #[must_use]
    fn load_or_default(path: &std::path::Path) -> Self {
        use actions::hotkeys::{ActionsToKeys, KeysToActions};

        // The parsed mappings, or an error. If the value is none, file not found - not actually an error.
        let mappings: Result<Option<_>, LoadBlockReason> = try_block::try_block! {
            let string = match std::fs::read_to_string(path) {
                Ok(string) => string,
                // File not found. This isn't an error, the file just doesn't exist. Write it!
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                // Other IO error, block loading.
                Err(e) => return Err(e.into()),
            };
            // Parse and invert, reporting parse or inversion errors as necessary.
            // Hehe, funny map syntax!
            let actions_to_keys : ActionsToKeys = toml::from_str(&string)?;
            let keys_to_actions : KeysToActions = (&actions_to_keys).try_into()?;

            Ok(Some((actions_to_keys,keys_to_actions)))
        };

        match mappings {
            // All went well~!
            Ok(Some((actions_to_keys, keys_to_actions))) => Self {
                load_blocker: None,
                actions_to_keys,
                keys_to_actions,
            },
            // File-not-found, write defaults.
            Ok(None) => {
                log::info!("hotkeys not found, defaulting");
                Self::with_defaults()
            }
            // Some kind of error exists when parsing, load defaults and prevent writes until user clears the error.
            Err(e) => {
                log::error!("failed to load hotkeys: {e}");
                // Take defaults but remember the error.
                Self {
                    load_blocker: Some(e),
                    ..Self::with_defaults()
                }
            }
        }
    }
    /// Returns the reason for read/write blockage, if any.
    #[must_use]
    pub fn load_blocker(&self) -> Option<&LoadBlockReason> {
        self.load_blocker.as_ref()
    }
    /// Save the loaded keys to the default location, overwriting contents.
    /// *This should not be called if [`Self::load_blocker`] is `Some` unless the user explicitly called for it.*
    pub fn save(&self) -> anyhow::Result<()> {
        let mut preferences =
            preferences_dir().ok_or_else(|| anyhow::anyhow!("No preferences dir found"))?;
        // Explicity do *not* create recursively. If not found, the user probably has a good reason.
        // Ignore errors (could already exist). Any real errors will be emitted by file access below.
        let _ = std::fs::DirBuilder::new().create(&preferences);

        preferences.push(Self::FILENAME);
        let mut string = toml::ser::to_string_pretty(&self.actions_to_keys)?;
        // Prefix some documentation.
        string = DOCUMENTATION.to_owned() + &string;
        std::fs::write(preferences, string)?;
        Ok(())
    }
}
impl TryFrom<crate::actions::hotkeys::ActionsToKeys> for Hotkeys {
    type Error = crate::actions::hotkeys::KeysToActionsError;
    fn try_from(
        actions_to_keys: crate::actions::hotkeys::ActionsToKeys,
    ) -> Result<Self, Self::Error> {
        let keys_to_actions = crate::actions::hotkeys::KeysToActions::try_from(&actions_to_keys)?;
        Ok(Self {
            load_blocker: None,
            actions_to_keys,
            keys_to_actions,
        })
    }
}

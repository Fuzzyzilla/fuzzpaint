use crate::actions;

#[must_use]
pub fn preferences_dir() -> Option<std::path::PathBuf> {
    let mut base_dir = dirs::preference_dir()?;
    base_dir.push(env!("CARGO_PKG_NAME"));
    Some(base_dir)
}

pub struct Hotkeys {
    failed_to_load: bool,
    pub actions_to_keys: actions::hotkeys::ActionsToKeys,
    pub keys_to_actions: actions::hotkeys::KeysToActions,
}
impl Hotkeys {
    const FILENAME: &'static str = "hotkeys.ron";
    /// Shared global hotkeys, saved and loaded from user preferences.
    /// (Or defaulted, if unavailable for some reason)
    #[must_use]
    pub fn get() -> &'static Self {
        static GLOBAL_HOTKEYS: std::sync::OnceLock<Hotkeys> = std::sync::OnceLock::new();

        GLOBAL_HOTKEYS.get_or_init(|| {
            let mut dir = preferences_dir();
            match dir.as_mut() {
                None => Self::no_path(),
                Some(dir) => {
                    dir.push(Self::FILENAME);
                    Self::load_or_default(dir)
                }
            }
        })
    }
    #[must_use]
    pub fn no_path() -> Self {
        use actions::hotkeys::ActionsToKeys;
        log::warn!("Hotkeys weren't available, defaulting.");
        let default = ActionsToKeys::default();
        // Default action map is reversable - this is assured by the default impl when debugging.
        let reverse = (&default).try_into().unwrap();

        Self {
            failed_to_load: true,
            keys_to_actions: reverse,
            actions_to_keys: default,
        }
    }
    #[must_use]
    fn load_or_default(path: &std::path::Path) -> Self {
        use actions::hotkeys::{ActionsToKeys, KeysToActions};
        let mappings: anyhow::Result<(ActionsToKeys, KeysToActions)> = try_block::try_block! {
            let string = std::fs::read_to_string(path)?;
            let actions_to_keys : ActionsToKeys = ron::from_str(&string)?;
            let keys_to_actions : KeysToActions = (&actions_to_keys).try_into()?;

            Ok((actions_to_keys,keys_to_actions))
        };

        match mappings {
            Ok((actions_to_keys, keys_to_actions)) => Self {
                failed_to_load: false,
                actions_to_keys,
                keys_to_actions,
            },
            Err(_) => Self::no_path(),
        }
    }
    /// Return true if loading user's settings failed. This can be useful for
    /// displaying a warning.
    #[must_use]
    pub fn did_fail_to_load(&self) -> bool {
        self.failed_to_load
    }
    pub fn save(&self) -> anyhow::Result<()> {
        let mut preferences =
            preferences_dir().ok_or_else(|| anyhow::anyhow!("No preferences dir found"))?;
        // Explicity do *not* create recursively. If not found, the user probably has a good reason.
        // Ignore errors (could already exist). Any real errors will be emitted by file access below.
        let _ = std::fs::DirBuilder::new().create(&preferences);

        preferences.push(Self::FILENAME);
        let writer = std::io::BufWriter::new(std::fs::File::create(&preferences)?);
        Ok(ron::ser::to_writer_pretty(
            writer,
            &self.actions_to_keys,
            ron::ser::PrettyConfig::default(),
        )?)
    }
}

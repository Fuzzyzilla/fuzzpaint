//! Utilities for conversion between process-local and file-local IDs during
//! Serialization and deserialization.

use crate::id::FuzzID;

pub struct FileLocalID<T: std::any::Any> {
    pub id: u32,
    _phantom: std::marker::PhantomData<T>,
}
// Safety - it's literally just a u32 lol
// We need these because if T is !Send or !Sync that is carried
// over to the ID, even though we don't actually store a T and thus
// shouldn't be bound by this.
unsafe impl<T: std::any::Any> Send for FileLocalID<T> {}
unsafe impl<T: std::any::Any> Sync for FileLocalID<T> {}
impl<T: std::any::Any> Clone for FileLocalID<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: std::any::Any> Copy for FileLocalID<T> {}
impl<T: std::any::Any> std::hash::Hash for FileLocalID<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(self.id);
        std::any::TypeId::of::<T>().hash(state);
    }
}
impl<T: std::any::Any> PartialEq for FileLocalID<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T: std::any::Any> Eq for FileLocalID<T> {}
impl<T: std::any::Any> From<u32> for FileLocalID<T> {
    fn from(value: u32) -> Self {
        Self {
            id: value,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum InternError {
    #[error("too many entries")]
    TooManyEntries,
}
/// Given many `FuzzID`s, normalizes them into sequential file-local ids on a
/// first-come-first-serve basis.
///
/// Reverse of [`ProcessLocalInterner`].
pub struct FileLocalInterner<T: std::any::Any> {
    // We can use a fast hasher without care for DOS prevention,
    // as the user cannot influence FuzzID<T> values.
    map: hashbrown::HashMap<FuzzID<T>, FileLocalID<T>>,
    /// latches to None on overflow.
    next_id: Option<u32>,
}
/// Increment in place, short circuiting to None if overflow occurs.
/// returns the value *before* increment.
fn checked_postfix_increment(val: &mut Option<u32>) -> Option<u32> {
    match *val {
        Some(v) => {
            // Become None on overflow
            *val = v.checked_add(1);
            Some(v)
        }
        None => None,
    }
}
impl<T: std::any::Any> Default for FileLocalInterner<T> {
    fn default() -> Self {
        Self {
            map: hashbrown::HashMap::new(),
            next_id: Some(0),
        }
    }
}
impl<T: std::any::Any> FileLocalInterner<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Gets or creates a file-local id for the given `FuzzID`
    pub fn get_or_insert(&mut self, id: FuzzID<T>) -> Result<FileLocalID<T>, InternError> {
        match self.map.entry(id) {
            hashbrown::hash_map::Entry::Occupied(o) => Ok(*o.get()),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let id = checked_postfix_increment(&mut self.next_id)
                    .ok_or(InternError::TooManyEntries)?;

                let id = id.into();
                v.insert(id);
                Ok(id)
            }
        }
    }
    /// Get a file-local id without creating it if it's not present.
    #[must_use]
    pub fn get(&self, id: FuzzID<T>) -> Option<FileLocalID<T>> {
        self.map.get(&id).copied()
    }
    /// Insert an id. Returns Ok(true) if the id was new.
    pub fn insert(&mut self, id: FuzzID<T>) -> Result<bool, InternError> {
        // Implementation of get_or_insert is equally as cheap as a custom insert.
        match self.map.entry(id) {
            hashbrown::hash_map::Entry::Occupied(_) => Ok(false),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let id = checked_postfix_increment(&mut self.next_id)
                    .ok_or(InternError::TooManyEntries)?;

                v.insert(id.into());
                Ok(true)
            }
        }
    }
}

/// Given many `FileLocalID`s, convert them to arbitrary process-local `FuzzID`s.
/// The insert methods are all infallible - process ID overflow is not currently
/// a recoverable condition, and will lead to unclean process termination.
///
/// Reverse of [`FileLocalInterner`].
pub struct ProcessLocalInterner<T: std::any::Any> {
    map: hashbrown::HashMap<FileLocalID<T>, FuzzID<T>>,
}
impl<T: std::any::Any> Default for ProcessLocalInterner<T> {
    fn default() -> Self {
        Self {
            map: hashbrown::HashMap::new(),
        }
    }
}
impl<T: std::any::Any> ProcessLocalInterner<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// From many sequential `FileLocalID`s from zero up to count, allocate IDs.
    /// Much more efficient than allocating one-by-one.
    pub fn many_sequential(count: usize) -> Result<Self, InternError> {
        use az::CheckedAs;
        let count: u32 = count.checked_as().ok_or(InternError::TooManyEntries)?;
        let mut map =
            hashbrown::HashMap::<FileLocalID<T>, FuzzID<T>>::with_capacity(count as usize);
        // Allocate ids.
        let file_local_ids = 0..count;
        let process_local_ids = FuzzID::many(count as usize);

        // Extend with pairs of sequential file id, bulk allocated process ID.
        map.extend(
            file_local_ids
                .zip(process_local_ids)
                .map(|(file_idx, fuzz_id)| (file_idx.into(), fuzz_id)),
        );

        // Make sure we allocated as many as promised
        debug_assert_eq!(map.len(), count as usize);

        Ok(Self { map })
    }
    /// Gets or creates a `FuzzID` id for the given `FileLocalID`
    pub fn get_or_insert(&mut self, id: FileLocalID<T>) -> FuzzID<T> {
        match self.map.entry(id) {
            hashbrown::hash_map::Entry::Occupied(o) => *o.get(),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let new_id = FuzzID::default();
                v.insert(new_id);
                new_id
            }
        }
    }
    #[must_use]
    pub fn get(&self, id: FileLocalID<T>) -> Option<FuzzID<T>> {
        self.map.get(&id).copied()
    }
    /// Insert an id. Convenience fn for `get_or_insert` while discarding the result.
    pub fn insert(&mut self, id: FileLocalID<T>) {
        self.get_or_insert(id);
    }
    pub fn iter(&self) -> impl Iterator<Item = (&FileLocalID<T>, &FuzzID<T>)> {
        self.map.iter()
    }
}

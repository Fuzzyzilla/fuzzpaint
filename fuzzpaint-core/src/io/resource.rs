//! # Resource format
//!
//! Readers and writers for the resource format. See "resource-fileschema.md".

use std::io::{Read, Result as IOResult, Write};

use crate::brush::UniqueID;

use super::common::{MyTake, SoftSeek};

// Repr (C) for matching layout in file. Take care for endianness!
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
#[repr(C)]
struct Node {
    pub id: UniqueID,
    pub len_offset: LenOffset,
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
#[repr(C)]
struct LenOffset {
    /// One bit has-left-node flag, 63 bit data offset.
    pub left_offset: u64,
    /// One bit has-right-node flag, 63 bit data len.
    pub right_len: u64,
}
impl Node {
    const SIZE: u64 = std::mem::size_of::<Self>() as u64;
    const COMMON_PREFIX_SIZE: u64 = Self::SIZE - 1;
}
impl LenOffset {
    const FLAG_BIT: u64 = 0x8000_0000_0000_0000;
    const FIELD_MASK: u64 = !Self::FLAG_BIT;
    /// Create new length/offset bitfields. Returns `None` if either `len` or `offset` are too large (top bit set).
    fn new(len: u64, offset: u64, has_left: bool, has_right: bool) -> Option<Self> {
        if len & Self::FLAG_BIT != 0 || offset & Self::FLAG_BIT != 0 {
            return None;
        }

        Some(Self {
            left_offset: offset | if has_left { Self::FLAG_BIT } else { 0 },
            right_len: len | if has_right { Self::FLAG_BIT } else { 0 },
        })
    }
    fn has_left(&self) -> bool {
        self.left_offset & Self::FLAG_BIT != 0
    }
    fn offset(&self) -> u64 {
        self.left_offset & Self::FIELD_MASK
    }
    fn has_right(&self) -> bool {
        self.right_len & Self::FLAG_BIT != 0
    }
    fn len(&self) -> u64 {
        self.right_len & Self::FIELD_MASK
    }
    fn set_has_left(&mut self, left: bool) {
        if left {
            self.left_offset |= Self::FLAG_BIT;
        } else {
            self.left_offset &= Self::FIELD_MASK;
        }
    }
    fn set_has_right(&mut self, right: bool) {
        if right {
            self.right_len |= Self::FLAG_BIT;
        } else {
            self.right_len &= Self::FIELD_MASK;
        }
    }
    /// Set the bitfield. Returns `Err` if too large (top bit set).
    fn set_offset(&mut self, offset: u64) -> Result<(), ()> {
        if offset & Self::FLAG_BIT != 0 {
            Err(())
        } else {
            // Clear those bits
            self.left_offset ^= self.left_offset & Self::FIELD_MASK;
            self.left_offset |= offset;

            Ok(())
        }
    }
    /// Set the bitfield. Returns `Err` if too large (top bit set).
    fn set_len(&mut self, len: u64) -> Result<(), ()> {
        if len & Self::FLAG_BIT != 0 {
            Err(())
        } else {
            // Clear those bits
            self.right_len ^= self.right_len & Self::FIELD_MASK;
            self.right_len |= len;

            Ok(())
        }
    }
}

/// Read a single node. The cursor is left at the byte past-the-end of the node.
/// # Errors
/// Errs are forwarded from the reader `r`.
fn read_node<R: Read>(common_prefix: Option<u8>, mut r: R) -> IOResult<Node> {
    let mut node = <Node as bytemuck::Zeroable>::zeroed();
    let bytes = bytemuck::bytes_of_mut(&mut node);

    let slice = if let Some(common_prefix) = common_prefix {
        bytes[0] = common_prefix;
        // Skip reading the first byte in common-prefix mode.
        &mut bytes[1..]
    } else {
        &mut bytes[..]
    };

    r.read_exact(slice)?;

    // Change endianness from le -> native.
    node.len_offset.left_offset = u64::from_le_bytes(node.len_offset.left_offset.to_ne_bytes());
    node.len_offset.right_len = u64::from_le_bytes(node.len_offset.right_len.to_ne_bytes());

    Ok(node)
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error("id not found in file")]
    NotFound,
    #[error("invalid file header")]
    BadMagic,
}

/// Find a resource from the given reader. Returns a [`MyTake`] over the resource's data, or `Ok(Err(r))` if the
/// resource is not contained.
/// Does not assume the reader starts at position 0 - a resource tree inside another stream is accounted for.
/// If `common_prefix` is set, all entries are taken to have the same start (`[0]`) byte as the requested `resource`.
/// # Errors
/// Top level errs are forwarded from the reader `r`.
/// inner err occurs when resource missing - the reader is returned as-is as an error if not found.
pub fn fetch<R: Read + SoftSeek>(
    resource: UniqueID,
    common_prefix: bool,
    mut r: R,
) -> Result<MyTake<R>, FetchError> {
    let mut tree_depth: u8 = 0;

    if tree_depth & 0b1100_0000 != 0 {
        return Err(FetchError::BadMagic);
    }
    r.read_exact(std::slice::from_mut(&mut tree_depth))?;

    let node_size = if common_prefix {
        Node::COMMON_PREFIX_SIZE
    } else {
        Node::SIZE
    };

    let common_prefix = common_prefix.then_some(resource.0[0]);

    let offset_of_level = |level: u8| -> u64 {
        let nodes_from_tree_start = (1u64 << level) - 1;
        // +1 for header (tree-depth byte)
        1 + nodes_from_tree_start * node_size
    };
    let offset_of = |level: u8, cursor: u64| -> u64 { offset_of_level(level) + cursor * node_size };

    // "X-position" of current read, from left-to-right on current level.
    // On every new level, this doubles and increases by zero (for left) or one (for right)
    let mut current_x = 0u64;
    for current_depth in 0..tree_depth {
        // We can assume we're at the right place to read the relavent node.
        // (starts at node 0, and the row-advance logic below seeks to next relavent node).
        let node = read_node(common_prefix, &mut r)?;

        // Where are we right now? (x + 1 since we left the cursor went past-the-end of this node during read)
        let cur_stream_pos = offset_of(current_depth, current_x + 1);

        match resource.cmp(&node.id) {
            std::cmp::Ordering::Equal => {
                // Found it~!
                // Special case: zero-len resource. Offset may be bogus.
                if node.len_offset.len() == 0 {
                    return Ok(MyTake::new(r, 0));
                }

                // Where are we going? `offset` is from end-of-tree
                let dest_stream_pos = node
                    .len_offset
                    .offset()
                    // Data from outside world, check check check!
                    // offset of `tree_depth` gives past-the-end ptr of whole tree.
                    .checked_add(offset_of_level(tree_depth))
                    .unwrap();

                // Strictly >= 0, since we're in the tree and the data is always past end of tree.
                let forward_by = dest_stream_pos - cur_stream_pos;
                r.soft_seek(i64::try_from(forward_by).unwrap())?;

                // Woohoo!
                return Ok(MyTake::new(r, node.len_offset.len()));
            }
            std::cmp::Ordering::Greater => {
                // Go right..
                if node.len_offset.has_right() {
                    // (silly opt: this math can be weakened into seek (WIDTH+1)*node_size but it'd make things icky)
                    current_x = current_x * 2 + 1;
                } else {
                    // No nodes to the right, fail!
                    return Err(FetchError::NotFound);
                }
            }
            std::cmp::Ordering::Less => {
                // Go left..
                if node.len_offset.has_left() {
                    current_x *= 2;
                } else {
                    // No nodes to the left, fail!
                    return Err(FetchError::NotFound);
                }
            }
        };

        // Just changed cursor, about to change row. Seek!
        let dest_stream_pos = offset_of(current_depth + 1, current_x);
        // Always strictly forward.
        let forward_by = dest_stream_pos - cur_stream_pos;
        r.soft_seek(i64::try_from(forward_by).unwrap())?;
    }

    // Fell through, not found :<
    Err(FetchError::NotFound)
}

/// Enumerate all resources in the given stream. Order of read objects is arbitrary.
/// # Errors
/// Returns error that occurs if the header byte could not be read.
/// Further errors are returned by the iterator itself.
pub fn enumerate<R: Read + SoftSeek>(
    common_prefix: Option<u8>,
    mut reader: R,
) -> IOResult<IDIter<R>> {
    let mut depth = 0u8;
    reader.read_exact(std::slice::from_mut(&mut depth))?;

    if depth & 0b1100_0000 != 0 {
        unimplemented!("reserved flags");
    }

    let max_width_hint = 1u64 << depth;
    // Ensure we don't explode. Files can't be trusted! We can dynamically grow in case it turns out we actually *do*
    // have that much data.
    let max_width_hint = usize::try_from(max_width_hint.min(1024)).unwrap();
    let mut active_mask = bitvec::vec::BitVec::with_capacity(max_width_hint);

    // Root always available if not empty.
    if depth != 0 {
        active_mask.insert(0, true);
    }

    let next_active_mask = bitvec::vec::BitVec::with_capacity(max_width_hint);

    Ok(IDIter {
        active_mask,
        next_active_mask,
        cur_node: 0,
        cur_width: 1,
        num_nodes: (1u64 << depth) - 1,
        common_prefix,
        reader,
    })
}

/// Iterator over all object IDs in a resource file.
/// IDs are visited in arbitrary order. If any IO error occurs, it is returned and the iterator
/// permanently short circuits to `None`.
pub struct IDIter<R> {
    // A lotttt of fields. This is a pretty complex co-routine, oof.

    // Subtrees may terminate before iteration finishes.
    // Use this to mask those out.
    active_mask: bitvec::vec::BitVec,
    // Active mask of next layer.
    next_active_mask: bitvec::vec::BitVec,
    // Which idx are we on now? counts from 0.
    cur_node: u64,
    // Width of the current layer.
    cur_width: u64,
    // How many nodes are there? inclusive
    num_nodes: u64,
    common_prefix: Option<u8>,
    reader: R,
}
impl<R: Read + SoftSeek> Iterator for IDIter<R> {
    type Item = IOResult<UniqueID>;
    fn next(&mut self) -> Option<Self::Item> {
        // Finished :3
        if self.cur_node >= self.num_nodes {
            return None;
        }
        let mut try_next = || -> IOResult<Option<UniqueID>> {
            loop {
                let node = read_node(self.common_prefix, &mut self.reader)?;
                // Check if this passes the mask!
                let is_real = self.active_mask.pop().unwrap();

                let left = is_real && node.len_offset.has_left();
                let right = is_real && node.len_offset.has_right();

                // Push new mask flags. (needs to be reversed at the end)
                self.next_active_mask.push(left);
                self.next_active_mask.push(right);

                self.cur_node += 1;
                // End-of-line
                // Magic numbers work out since width grows in powers of two!
                if self.cur_node == self.cur_width * 2 - 1 {
                    if self.next_active_mask.not_any() {
                        // Whole next row is masked out - done here! Tree may be oversized.
                        // Short circuit:
                        self.cur_node = self.num_nodes;
                        return Ok(None);
                    }

                    self.cur_width *= 2;
                    // Reverse and set next to current. Re-use alloc by clearing it and swapping :3.
                    self.next_active_mask.reverse();
                    self.active_mask.clear();

                    std::mem::swap(&mut self.next_active_mask, &mut self.active_mask);
                }

                // Found next :3
                if is_real {
                    return Ok(Some(node.id));
                }

                // That was our last one :<
                if self.cur_node >= self.num_nodes {
                    return Ok(None);
                }
            }
        };

        match try_next() {
            Err(e) => {
                self.cur_node = self.num_nodes;
                Some(Err(e))
            }
            // funny transpose!!
            Ok(Some(o)) => Some(Ok(o)),
            Ok(None) => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::try_from(self.num_nodes - self.cur_node).ok();

        // Because IO may short circuit us (and there may be dead spaces) we don't have a precise lower-bound.
        (0, remaining)
    }
}
impl<R> IDIter<R> {
    /// Take the reader back. It will be left at it's current position, with no guarantees as to where that's at!
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Map of [`UniqueID`] -> [`LenOffset`], in a representation (somewhat) efficiently convertible to/from a resource file header.
struct DenseBinaryTree {
    /// Nodes, sorted and deduplicated by ID.
    nodes: Vec<Node>,
}
impl DenseBinaryTree {
    /// # Safety
    /// The given vector must be sorted and deduplicated by `node.id`.
    #[must_use]
    pub unsafe fn from_sorted(mut nodes: Vec<Node>) -> Self {
        // Check unsafe preconditions
        #[cfg(debug_assertions)]
        {
            let mut windows = nodes.windows(2);
            while let Some([a, b]) = windows.next() {
                // Strictly less checks dedup as well.
                debug_assert!(
                    a.id < b.id,
                    "DenseBinaryTree::from_sorted unsafe precondition"
                );
            }
        }
        Self { nodes }
    }
    /// Convert internal repr into file-ready representation.
    /// That is, balanced left-to-right, root-to-branch tree with empty spaces zeroed.
    pub fn into_dense_tree(self) -> Vec<Node> {
        let Self { mut nodes } = self;
        // Convert sorted into our dense tree repr.
        // Sorted tail that hasn't been tree-ified yet:
        // (self invariant guarantees sorted!)
        let mut unsifted_slice = &mut nodes[..];
        // Iterate root-to-branch. This is how many nodes "wide" the tree layer is at current.
        let mut width = 1usize;

        while !unsifted_slice.is_empty() {
            // split into `width` subslices.
            let elems_per_subslice = unsifted_slice.len() / width;
            if elems_per_subslice != 0 {
                for i in 0..width - 1 {
                    // Take middle elem of this subslice, rotate it to the front.
                    let midpoint = i * elems_per_subslice + elems_per_subslice / 2;

                    // Midpoint indexes by original unsifted_slice len, but it
                    // gets one shorter each iter, hence -i. weird!
                    let front = &mut unsifted_slice[..=midpoint - i];

                    // Rotate right takes `midpoint` elem, puts it at front, and
                    // keeps tail sorted.
                    front.rotate_right(1);

                    // Pop the node we inserted into the tree off the front.
                    unsifted_slice = &mut unsifted_slice[1..];
                }
            }

            // Do one more with the rest of the slice
            // This is never empty - for loop above would have never run if len < width, which is the only
            // case where it could be drained.
            // Same op, see above.
            let middle = unsifted_slice.len() / 2;
            let front = &mut unsifted_slice[..=middle];
            front.rotate_right(1);

            unsifted_slice = &mut unsifted_slice[1..];

            width *= 2;
        }

        //todo: set left/right flags.
        //todo: populate zeroes.
        todo!()
    }
    /// Make from a vector of nodes. Returns an error if any node ID appears more than once.
    pub fn from_vec(mut nodes: Vec<Node>) -> Result<Self, ()> {
        nodes.sort_unstable_by(|a, b| a.id.cmp(&b.id));

        if nodes.windows(2).any(|arr| {
            let [a, b] = arr else { unreachable!() };
            a.id == b.id
        }) {
            // Had duplicates.
            return Err(());
        }

        // Safety: We just sorted and asserted that it's deduplicated :P
        Ok(unsafe { Self::from_sorted(nodes) })
    }
    #[must_use]
    pub fn get(&mut self, id: &UniqueID) -> Option<&LenOffset> {
        self.nodes
            .binary_search_by(|node| node.id.cmp(id))
            .ok()
            .map(|idx| &self.nodes[idx].len_offset)
    }
    #[must_use]
    pub fn get_mut(&mut self, id: &UniqueID) -> Option<&mut LenOffset> {
        self.nodes
            .binary_search_by(|node| node.id.cmp(id))
            .ok()
            .map(|idx| &mut self.nodes[idx].len_offset)
    }
    pub fn get_or_insert(&mut self, or_insert: Node) -> &mut LenOffset {
        match self.nodes.binary_search(&or_insert) {
            Err(into_idx) => {
                self.nodes.insert(into_idx, or_insert);
                &mut self.nodes[into_idx].len_offset
            }
            Ok(found_idx) => &mut self.nodes[found_idx].len_offset,
        }
    }
    pub fn insert(&mut self, node: Node) -> Result<&mut LenOffset, Node> {
        match self.nodes.binary_search(&node) {
            Err(into_idx) => {
                self.nodes.insert(into_idx, node);
                Ok(&mut self.nodes[into_idx].len_offset)
            }
            Ok(_) => Err(node),
        }
    }
    pub fn remove(&mut self, id: &UniqueID) -> Option<Node> {
        self.nodes
            .binary_search_by(|node| node.id.cmp(id))
            .ok()
            .map(|idx| self.nodes.remove(idx))
    }
    pub fn iter(&self) -> std::slice::Iter<'_, Node> {
        self.nodes.iter()
    }
    // This type *is* nameable but I don't uhhhh wanna (would be needed for `IntoIterator for &mut Self`)
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&UniqueID, &mut LenOffset)> + '_ {
        self.nodes
            .iter_mut()
            .map(|node| (&node.id, &mut node.len_offset))
    }
    #[must_use]
    /// Get the "depth" or "height" of the contained tree.
    pub fn depth(&self) -> u8 {
        // Unwrap ok - I mean, if you have a arch with 256 bit pointers, im all for changing this.
        u8::try_from(usize::BITS - self.nodes.len().trailing_zeros()).unwrap()
    }
}
impl<'a> IntoIterator for &'a DenseBinaryTree {
    type IntoIter = std::slice::Iter<'a, Node>;
    type Item = &'a Node;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// File updater that caches the node tree, providing read/write access to the tree and
/// data.
pub struct CachedUpdater<RW> {
    // We need Read functionality to initially load tree.
    // We need Read + Seek for fetch.
    // We need Write for initial bulk write
    // We need Write + Seek functionality to add new entries.
    // These requirements are imposed on the impls themselves for flexibility :3
    stream: RW,
    cached_tree: DenseBinaryTree,
    /// After-end index of the data region.
    after_end: u64,
}
/// Updaters
impl<W: Write + SoftSeek> CachedUpdater<W> {
    /// Update a record. None if the record doesn't exist.
    /// API Limitation - records may not grow nor shrink in size. This is fine for all forseeable needs.
    pub fn update(&mut self, id: &UniqueID) -> Option<MyTake<&mut W>> {
        let entry = self.cached_tree.get(id)?;
        let offset = entry.offset();
        let len = entry.len();
        todo!()
    }
    /// Insert data into the collection, where the ID is taken from the `blake3` hash of the data.
    /// Returns `None` without doing any changes if the ID is already contained.
    pub fn insert_new(&mut self, id: UniqueID, data: &[u8]) -> Option<IOResult<UniqueID>> {
        let initial_offset = self.after_end;
        let len = u64::try_from(data.len()).unwrap();
        self.cached_tree
            .insert(Node {
                id,
                len_offset: LenOffset::new(len, initial_offset, false, false).unwrap(),
            })
            .ok()?;
        todo!()
    }
}

#[cfg(test)]
mod test {
    use std::io::Read;

    use super::{enumerate, fetch, FetchError, UniqueID};

    // A buncha hand-written binary files for testing lol
    /// Resource with two data values. See [`smol_load`] for contents.
    const SMOL_RESOURCE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/resource/smol.bin"
    ));
    /// Resource with no tree, no data. (literally a single zero byte)
    const EMPTY_RESOURCE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/resource/empty.bin"
    ));
    /// Big tree structure.
    /// Non-prefixed IDs, all zero with their first byte as follows:
    /// Each has one byte of attached data, matching that ID byte.
    /// <pre>
    ///         80
    ///        /  \
    ///       /    \
    ///      /      \
    ///     40      C0
    ///    /  \    /  \
    ///   20   X  A0   X
    ///  /  \    /  \
    /// X   30  90   X
    /// </pre>
    const BIG_TREE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/resource/bigtree.bin"
    ));

    const CONSECUTIVE_ID: UniqueID = UniqueID([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ]);

    /// Resource fetching.
    #[test]
    fn smol_load() {
        let read = std::io::Cursor::new(SMOL_RESOURCE);
        // Item with consecutive values as it's id maps to the data "owo"
        let mut result = fetch(CONSECUTIVE_ID, false, read.clone()).unwrap();

        let mut str = String::new();
        result.read_to_string(&mut str).unwrap();
        assert_eq!(str, "owo");

        // Item with 0s as it's id maps to the data "uwu"
        let mut result = fetch(UniqueID([0; 32]), false, read.clone()).unwrap();

        let mut str = String::new();
        result.read_to_string(&mut str).unwrap();
        assert_eq!(str, "uwu");

        // This item does not exist, but points to an empty node - tests that
        // left-right flags work.
        assert!(matches!(
            fetch(UniqueID([0xff; 32]), false, read.clone()).err(),
            Some(FetchError::NotFound),
        ));
    }

    /// List resources on a file with large IDs
    #[test]
    fn smol_list_ids() {
        let read = std::io::Cursor::new(SMOL_RESOURCE);
        let iter = enumerate(None, read).unwrap();

        let mut set = [CONSECUTIVE_ID, UniqueID([0; 32])]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();

        for id in iter {
            let id = id.unwrap();
            // Should exist.
            assert!(set.remove(&id));
        }
        // Should have found them all!
        assert!(set.is_empty());
    }

    /// Ensure no spurious reads on an empty tree
    #[test]
    fn empty_list_ids() {
        let read = std::io::Cursor::new(EMPTY_RESOURCE);
        let mut iter = enumerate(None, read).unwrap();
        // Empty resource file.
        assert_eq!(format!("{:?}", iter.next()), "None");
    }

    /// Bigger tree with voids, ensures that masking behavior is workin.
    #[test]
    fn big_tree_list_ids() {
        let read = std::io::Cursor::new(BIG_TREE);
        let iter = enumerate(None, read).unwrap();

        let mut set = [0x80, 0x40, 0xC0, 0x20, 0xA0, 0x30, 0x90]
            .into_iter()
            .map(|low_byte| {
                // Struct update syntax for arrays please please please please
                let mut id = UniqueID([0; 32]);
                id.0[0] = low_byte;
                id
            })
            .collect::<std::collections::BTreeSet<_>>();

        for id in iter {
            let id = id.unwrap();
            // Should exist.
            assert!(set.remove(&id));
        }
        // Should have found them all!
        assert!(set.is_empty());
    }

    /// Visit all nodes in big tree! The data is empty, but that's still a success :3
    #[test]
    fn big_tree_load() {
        let read = std::io::Cursor::new(BIG_TREE);

        for low_byte in [0x80, 0x40, 0xC0, 0x20, 0xA0, 0x30, 0x90] {
            // Struct update syntax for arrays please please please please
            let mut id = UniqueID([0; 32]);
            id.0[0] = low_byte;

            let mut r = fetch(id, false, read.clone()).unwrap();

            // All resources have one byte - their own ID's first byte.
            // Try to read an extra, too (to make sure it fails!)
            let mut bytes = [0, 0];
            let num_read = r.read(&mut bytes).unwrap();

            assert_eq!(num_read, 1);

            assert_eq!(bytes[0], low_byte);
        }
    }
}

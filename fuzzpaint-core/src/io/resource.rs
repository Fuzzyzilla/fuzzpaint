//! # Resource format
//!
//! Readers and writers for the resource format. See "resource-fileschema.md".

use std::io::{Read, Result as IOResult, Take};

use crate::brush::UniqueID;

use super::common::SoftSeek;

// Repr (C) for matching layout in file. Take care for endianness!
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
#[repr(C)]
struct Node {
    pub id: UniqueID,
    /// One bit has-left-node flag, 63 bit data offset.
    pub left_offset: u64,
    /// One bit has-right-node flag, 63 bit data len.
    pub right_length: u64,
}
impl Node {
    const SIZE: u64 = std::mem::size_of::<Self>() as u64;
    const COMMON_PREFIX_SIZE: u64 = std::mem::size_of::<Self>() as u64 - 1;
    fn has_left(&self) -> bool {
        self.left_offset & 0x8000_0000_0000_0000 != 0
    }
    fn offset(&self) -> u64 {
        self.left_offset & 0x7FFF_FFFF_FFFF_FFFF
    }
    fn has_right(&self) -> bool {
        self.right_length & 0x8000_0000_0000_0000 != 0
    }
    fn len(&self) -> u64 {
        self.right_length & 0x7FFF_FFFF_FFFF_FFFF
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
    node.left_offset = u64::from_le_bytes(node.left_offset.to_ne_bytes());
    node.right_length = u64::from_le_bytes(node.right_length.to_ne_bytes());

    Ok(node)
}

/// Find a resource from the given reader. Returns a [`Take`] over the resource's data, or `Ok(Err(r))` if the
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
) -> IOResult<Result<Take<R>, R>> {
    let mut tree_depth: u8 = 0;

    if tree_depth & 0b1100_0000 != 0 {
        unimplemented!("reserved flags");
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
    'bail: for current_depth in 0..tree_depth {
        // We can assume we're at the right place to read the relavent node.
        // (starts at node 0, and the row-advance logic below seeks to next relavent node).
        let node = read_node(common_prefix, &mut r)?;

        // Where are we right now? (x + 1 since we left the cursor went past-the-end of this node during read)
        let cur_stream_pos = offset_of(current_depth, current_x + 1);

        match resource.cmp(&node.id) {
            std::cmp::Ordering::Equal => {
                // Found it~!
                // Special case: zero-len resource. Offset may be bogus.
                if node.len() == 0 {
                    return Ok(Ok(r.take(0)));
                }

                // Where are we going? `offset` is from end-of-tree
                let dest_stream_pos = node
                    .offset()
                    // Data from outside world, check check check!
                    // offset of `tree_depth` gives past-the-end ptr of whole tree.
                    .checked_add(offset_of_level(tree_depth))
                    .unwrap();

                // Strictly >= 0, since we're in the tree and the data is always past end of tree.
                let forward_by = dest_stream_pos - cur_stream_pos;
                r.soft_seek(i64::try_from(forward_by).unwrap())?;

                // Woohoo!
                return Ok(Ok(r.take(node.len())));
            }
            std::cmp::Ordering::Greater => {
                // Go right..
                if node.has_right() {
                    // (silly opt: this math can be weakened into seek (WIDTH+1)*node_size but it'd make things icky)
                    current_x = current_x * 2 + 1;
                } else {
                    // No nodes to the right, fail!
                    break 'bail;
                }
            }
            std::cmp::Ordering::Less => {
                // Go left..
                if node.has_left() {
                    current_x *= 2;
                } else {
                    // No nodes to the left, fail!
                    break 'bail;
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
    Ok(Err(r))
}

/// Enumerate all resources in the given stream. Order of read objects is arbitrary.
pub fn enumerate<R: Read + SoftSeek>(
    common_prefix: Option<u8>,
    mut reader: R,
) -> IOResult<ResourceIter<R>> {
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

    Ok(ResourceIter {
        active_mask,
        next_active_mask,
        cur_node: 0,
        cur_width: 1,
        num_nodes: (1u64 << depth) - 1,
        common_prefix,
        reader,
    })
}

pub struct ResourceIter<R> {
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
impl<R: Read + SoftSeek> Iterator for ResourceIter<R> {
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

                let left = is_real && node.has_left();
                let right = is_real && node.has_right();

                // Push new mask flags. (needs to be reversed at the end)
                self.next_active_mask.push(left);
                self.next_active_mask.push(right);

                self.cur_node += 1;
                // End-of-line
                // Magic numbers work out since width grows in powers of two!
                if self.cur_node == self.cur_width * 2 - 1 {
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

#[cfg(test)]
mod test {
    use std::io::Read;

    use super::{enumerate, fetch, UniqueID};

    /// Resource with two data values.
    const SMOL_RESOURCE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/resource/smol.bin"
    ));
    /// Resource with no tree, no data.
    const EMPTY_RESOURCE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/resource/empty.bin"
    ));

    const CONSECUTIVE_ID: UniqueID = UniqueID([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ]);

    #[test]
    fn smol_load() {
        let read = std::io::Cursor::new(SMOL_RESOURCE);
        // Item with consecutive values as it's id maps to the data "owo"
        let mut result = fetch(CONSECUTIVE_ID, false, read.clone()).unwrap().unwrap();

        let mut str = String::new();
        result.read_to_string(&mut str).unwrap();
        assert_eq!(str, "owo");

        // Item with 0s as it's id maps to the data "uwu"
        let mut result = fetch(UniqueID([0; 32]), false, read.clone())
            .unwrap()
            .unwrap();

        let mut str = String::new();
        result.read_to_string(&mut str).unwrap();
        assert_eq!(str, "uwu");

        // This item does not exist, but points to an empty node - tests that
        // left-right flags work.
        assert!(fetch(UniqueID([0xff; 32]), false, read.clone())
            .unwrap()
            .is_err());
    }

    #[test]
    fn smol_list_ids() {
        let read = std::io::Cursor::new(SMOL_RESOURCE);
        let iter = enumerate(None, read).unwrap();

        let mut set = [CONSECUTIVE_ID, UniqueID([0; 32])]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();

        for id in iter {
            let id = id.unwrap();
            println!("{id:?}");
            // Should exist.
            assert!(set.remove(&id));
        }
        // Should have found them all!
        assert!(set.is_empty());
    }

    #[test]
    fn empty_list_ids() {
        let read = std::io::Cursor::new(EMPTY_RESOURCE);
        let mut iter = enumerate(None, read).unwrap();
        // Empty resource file.
        assert_eq!(format!("{:?}", iter.next()), "None");
    }
}

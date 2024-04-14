# Resource file schema version `0.0.0`
Resource files contain re-usable data collected from opened files as well as user-created assets. Such resources currently include textures and brushes.
They represent a `Key -> Value` structure optimized for random-access, where the Key is a 256-bit `BLAKE3` hash of the value contained.

## Security considerations
Because these resources come from untrusted sources, the format requires extra consideration to avoid DOS attacks. An attacker has some degree of control
over the IDs used in their resources - If, for example, the structure was implemented as a linear scan, the attacker could insert an arbitrary number of bogus
entries near the start and then repeatedly request a resource near the end.

## Data types
All values are stored in little-endian order.

The format has two modes, *common-prefix* and *normal*. In common-prefix mode, all entries are taken to have the same 8-bits at the start, and thus those bits are not stored.

* `UniqueID`:
    `[u8; 32]` if normal mode file, else `[u8;31]` where the byte at `[31]` is taken to be constant and is provided by the reader. A 256-bit `BLAKE3` hash of the referenced data.
* `Node`:
    | Type           | Meaning                                                                   |
    |----------------|---------------------------------------------------------------------------|
    | `UniqueID`     | Hash of the data this node references                                     |
    | `u64` bitfield | `[0:62]`: Data offset past-the-end of the tree. `[63]`: Has left-node. |
    | `u64` bitfield | `[0:62]`: Length of the data. `[63]`: Has right-node.                  |

## Structure
The file consists of a balanced binary tree, followed by a blob of all the entries' data. The binary tree, being O(logn) to find a given resource, should be sufficient to
counteract the attack vector described above.

| Type           | Meaning                                               |
|----------------|-------------------------------------------------------|
| `u8`           | Tree depth. Number of nodes is `2**(tree_depth) - 1`. |
| `Node[]`       | Dense [Binary tree](#binary-tree).                    |
| `u8[the rest]` | Blob into which the offsets and lengths refer.        |

### Binary tree
The tree is stored as "dense" - that is, even unoccupied nodes are to be included, but zeroed. It is stored "left"-to-"right," root-to-branch. This is chosen such that traversal may be implemented
as a strictly-forward scan, finding any given node path is an `O(1)` operation, insertion cost is amortized, and calculating the end-of-tree offset does not require traversal. The tree-depth may be oversized.

"Left" nodes are strictly-less, "right" nodes are strictly-greater. The offset provided in each node does not need to be ordered in any way.

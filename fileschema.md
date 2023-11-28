# `.fzp` file schema version `0.0.0`
---

The file version will follow SemVer following stable release, **but for version `0.0.X` it is ignored**:
 * Major versions represent fundamental shifts in file format or serialization.
 * Minor versions represent breaking changes of the same fundamental structure.
 * Patch versions are compatible with older software on loading but not necessarily saving. (TODO: Specify)
It uses the RIFF file container with a draft extension of `.fzp`.

## Data types
Rust type syntax is used to describe the schema where appropriate, with an implicit `repr(packed)`. Data types are in little-endian byte order as specified by RIFF, except where otherwise specified.

* `ChunkID` - A RIFF chunk ID, [u8;4]. Usually ascii, but any arbitrary bytes are allowed.
* `varint` - a variable-length integer, spanning 1-4 bytes. For each byte in little-endian order, the first seven bits
contain the least-significant digits of the represented number and the eighth bit is a continue flag which, when set,
indicates the integer spans another byte.
* `string` - a length-prefixed UTF-8 string without terminator, as in `{length: varint, utf8: [u8; length]}`
* `strz` - a length-prefixed and zero-terminated UTF-8 string, which when prefixed with an ID is valid as an entire chunk definition. The null terminator *is* included in the length. Used only in the `INFO` chunk for EXIF interoperability. UTF-8 is normally allowed to contain the null character, which would need to be explicitly disallowed by the writing software. `{length: u32, utf8: [NonZeroU8; length-1], 0u8}`
* `Version` - `{major: u8, minor: u8, patch: u8}` A SemVer version.
* `OrphanMode` - An enum flag to determine what a reader should do with a chunk if it is unable to parse it. `enum OrphanMode : u8 {Keep = 0, Discard = 1, Deny = 2}` When set to **Keep**, the reader must copy the data verbatim from the source to the destination, in an arbitrary location within the same parent, when saving the document in the future. When set to **Discard**, the reader must not copy the data to the destination if the file has been changed, preferably giving a message to the user that the operation is lossy. It is a hint that it is interdependent with other chunks of the document, and would be out-of-date if not also updated alongside them. When set to **Deny**, the reader is not allowed to make changes to the file if it cannot parse this chunk. 
* `VersionedChunkHeader` - `(Version, OrphanMode)`. Every custom chunk defined here and in the future must begin with this data. (todo: do i really wanna commit to that? lets give it a few versions lol) If a reader is unable to parse a given chunk due to its Version, it *must* respect the OrphanMode.
## Chunk Structure
The file is a flattened tree structure that does not allow runtime recursion.

Chunks may refer to arbitrary data structures within its own or its children's binary data by an offset from the beginning of itself. Chunks should not, however, refer to data outside of their own through offsets nor absolute pointers. This is to ensure that readers which cannot parse a given block are free to move it around without needing to touch its contents. If a chunk must make such reference, it must have `OrphanMode::Deny` or `OrphanMode::Discard`. (TODO: Is this sound?) Cross-references should instead be done using IDs.

1. `RIFF` `"fzp "`
   1. `LIST` `"INFO"`
      - `ICMT`: `strz`
      - `ISFT`: `strz`
      - `IART`: `strz`
      - ... example fields taken from [exiftool.org](https://exiftool.org/TagNames/RIFF.html#Info). These are actually defacto standards from AVI and WAV and are thus actually designed for music/video distribution, not digital images.
   2. [`thmb`](#thmb)
   May come in any order:
   - [`docv`](#docv)
   - `LIST` `"objs"` Document object tables
     - [`DICT`](#dict) [`"strk"`](#strk)
     - [`DICT`](#dict) [`"ptls"`](#ptls)
     - [`DICT`](#dict) [`"brsh"`](#brsh)
     - [`GRPH`](#grph) [`"blnd"`](#blnd)
   - [`GRPH`](#grph) [`"hist"`](#hist)

### `thmb`
An optional thumbnail-sized image (usually longest edge length 128 or 256 pixels, at user's preference) in [QOI format](https://qoiformat.org/) showing the merged document from the primary viewport at the moment of writing. If included, it must come second (or first, if `LIST "INFO"` is omitted) in the top-level chunk list. Writers should only populate this field if such an image is readily available at the time of writing, otherwise requiring a specific request from the user. Failure to decode or encode the thumbnail should not be a fatal error.

QOI is chosen for its high speed and fixed-sized memory footprint, lowering the file write delay and risk of allocation failure during file serialization.

This chunk may be oversized to allow for parallel serialization. The chunk size must therefore be treated as an upper bound for the length of the image data stream, not a precise size. The chunk is padded up to the declared size - padding data is undefined and may be trimmed or discarded by a reader.
### `DICT`
A chunk schema which provides a number of ordered entities, a table of statically-sized tightly packed metadata for each entry, followed by an optional stream of variable length data which the entities are allowed to spill into. Intended for bulk data storage with quick O(1) access times to a given entry's metadata and data.

`MetadataTy` consists of a `u32` offset, `u32` length, followed by any fixed length user data as specified by the subtype. A Zero offset represents the first element of the dynamic spillover array.
A reader may discard any spillover data which is not referenced by any entry. It may also duplicate overlapping spillover data into separate regions.
| Type                        | Meaning                                                                  |
|-----------------------------|--------------------------------------------------------------------------|
| `ChunkID`                   | Subtype                                                                  |
| `VersionedChunkHeader`      | Version and handling information for the metadatas and spillover data    |
| `u32`                       | Number of entries in the metadata table                                  |
| `u32`                       | sizeof MetadataTy                                                        |
| `[MetadataTy; num_entries]` | Entries                                                                  |
| `[u8]`                      | Dynamic sized spillover area, taking up remainder of this chunk's length |
### `GRPH`
Consists of many nodes of potentially heterogeneous type joined together into a graph structure.
Much like standard `LIST`, it lists a subtype before a series of subchunks:
* Zero or more [`node`](#node) with unique subtypes within this graph, each describing a type of node and listing zero or more such nodes.
* Exactly one [`conn`](#conn), describing how these nodes are linked together to form a tree, graph, etc.
### `node`
| Type                   | Meaning                                                                     |
|------------------------|-----------------------------------------------------------------------------|
| `ChunkID`              | Node type identifier (unique within the GRPH parent).                       |
| `VersionedChunkHeader` | Version and handling information, specific to each node type.               |
| ...                    | Todo                                                                        |
### `conn`
Describes the connection between many nodes within the parent `GRPH` chunk, each Node referred to by the unique combination of `{node_ty: ChunkID, idx: u32}`. Additionally, the special node `{"root", 0}` may be used as an implicit empty root node even if no such node type was declared.
### `docv`
Information about document viewport layouts, including positions, sizes, resolutions, background colors, ect. of viewports within the document.
### `strk`
A `DICT` Subtype. Contains lists of brush strokes. Each brush stroke contains a reference id to a point list (ptls), brush settings, ect. needed to place the stroke on the page.
### `ptls`
A `DICT` Subtype.
Contains zero or more point lists in Array-of-structures encoding. (SoA and compression to come) Points can come in several different schemas depending on the capabilities of the graphics interface device which generated them.

Extends the `DICT` `MetadataTy` with `fuzzpaint_vk::repositories::points::PointArchetype`.
Spillover data per entry consists of a slice of dynamic sized Points who's size is determined by PointArchetype. Every point in a given entry has the same size.
### `hist`
Optional. Contains the history tree for the document. May be arbitrarily trimmed, however it should be assured that any navigation of the listed history tree always results in valid changes to the document state as presented in the rest of the chunks. Failure to do this may lead to file history being lost!
Corresponds with `fuzzpaint_vk::commands`
### `brsh`
A `DICT` Subtype.
Contains zero or more brush definitions. Every brush utilized in the document must be included, although there may be extra brushes not used by the document listed as well. This allows for documents to serve as a method of brush distribution.
Corresponds with `fuzzpaint_vk::repositories::brush` (not implemented)

# `.fzp` file schema version `0.0.0`
---

The file version will follow SemVer following stable release, **but for version `0.0.X` it is ignored**:
 * Major versions represent fundamental shifts in file format or serialization.
 * Minor versions represent breaking changes of the same fundamental structure.
 * Patch versions are compatible with older software on loading but not necessarily saving. (TODO: Specify)
It uses the RIFF file container with a draft extension of `.fzp`.

## Data types
Rust type syntax is used to describe the schema where appropriate, with an implicit `repr(packed)`. Data types are in little-endian byte order as specified by RIFF, except where otherwise specified.

* `varint` - a variable-length integer, spanning 1-4 bytes. For each byte in little-endian order, the first seven bits
contain the least-significant digits of the represented number and the eighth bit is a continue flag which, when set,
indicates the integer spans another byte.
* `string` - a length-prefixed UTF-8 string without terminator, as in `{length: varint, utf8: [u8; length]}`
* `strz` - a zero-terminated UTF-8 string. Used only in the `INFO` chunk for EXIF interoperability. UTF-8 is allowed to contain the null character, which would need to be explicitly dissallowed by the writing software.

## Chunk Specifications
 * [INFO](#info)
 * [fdoc](#grph)
 * [grph](#grph)
 * [ptls](#ptls)
 * [hist](#hist)
 * [brsh](#brsh)


### `INFO`
Standard RIFF information chunk, using de-facto standard subchunks from [exiftool.org](https://exiftool.org/TagNames/RIFF.html#Info).
These fields can be useful for shells to display information on the file without being able to parse it, as well as for users who do not have access to the fuzzpaint software to view basic information about it's contents.

This needs work! Should I support `exif` chunk aswell? Potentially an exif preview image as well!
| Subchunk ID | Content desc.             |
|-------------|---------------------------|
| `ICMT`      | comment: strz             |
| `IART`      | artist: strz              |
| `ISFT`      | "fuzzpaint vX.X.X\0"      |
| ...         |                           |

### `fdoc`
Information about document layout, including it's position, size, resolution, background color, ect.
### `grph`
Contains zero or more blend nodes and their relationships, specifying how items are to be rendered and composited down into a single image.
Corresponds with `fuzzpaint_vk::state::graph`.
### `ptls`
Contains zero or more point lists in Array-of-structures or Structure-of-arrays encoding, potentially using compression. Points can come in several different schemas depending on the capabilities of the graphics device which generated them.
Corresponds with `fuzzpaint_vk::repositories::points`.
### `hist`
Optional. Contains the history tree for the document. May be arbitrarily trimmed, however it should be assured that the cumulative results of the history tree are equivalent to the document state presented in the rest of the chunks.
Corresponds with `fuzzpaint_vk::commands`
### `brsh`
Contains zero or more brush definitions. Every brush utilized in the document must be included, although there may be extra brushes not used by the document listed as well. This allows for documents to serve as a method of brush distribution.
Corresponds with `fuzzpaint_vk::repositories::brush` (not implemented)
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
* `strz` - a length-prefixed and zero-terminated UTF-8 string, valid as an entire chunk definition. the null terminator *is* included in the length. Used only in the `INFO` chunk for EXIF interoperability. UTF-8 is normally allowed to contain the null character, which would need to be explicitly dissallowed by the writing software. `{length: u32, utf8: [NonZeroU8; length-1], 0u8}`

## Chunk Structure
The file is a flattened tree structure that does not allow arbitrary depth or recursion.

1. RIFF "fzp "
   1. LIST "INFO"
      - ICMT: `strz`
      - ISFT: `strz`
      - IART: `strz`
      - ... example fields taken from [exiftool.org](https://exiftool.org/TagNames/RIFF.html#Info). These are actually defacto standards from AVI and WAV and are thus actually designed for music/video distribution, not digitial images.
   - [`docv`](#docv)
   - [`grph`](#grph)
   - [`ptls`](#ptls)
   - [`hist`](#hist)
   - [`brsh`](#brsh)

### `docv`
Information about document viewport layouts, including positions, sizes, resolutions, background colors, ect. of viewports within the document.
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
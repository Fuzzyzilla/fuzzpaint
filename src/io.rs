/// IO utilities not specific to the format.
pub mod common;
pub mod id;
pub mod riff;

/// Fields read from a file that were not understood, either due to unrecognized
/// chunkID or incompatible version, but the fields requested to be preserved through read/writes.
///
/// The data is not inspectible, as that would be an anti-pattern!
/// Extend the reader instead. When I inevitably come back to add
/// an accessor for this for whatever reason I ought to think really hard about it.
pub struct Residual {
    // Since the tree shape is static and well-known, we can simply
    // store the levels by name lol. If some extension adds recursion or
    // whatever, it will still fall into one of these buckets and the whole
    // structure will get dumped into a single ResidualChunk.
    /// Chunks from the top level RIFF
    riff: Vec<ResidualChunk>,
    /// Chunks from RIFF > LIST OBJS
    riff_list_objs: Vec<ResidualChunk>,
}
impl Residual {
    /// No residual data.
    pub fn empty() -> Self {
        Self {
            riff: vec![],
            riff_list_objs: vec![],
        }
    }
}
struct ResidualChunk {
    id: riff::ChunkID,
    header: VersionedChunkHeader,
    /// chunk length is implicit from this vec's length.
    /// bytes include the header, but not the id - just as RIFF does.
    data: Vec<u8>,
}

#[derive(thiserror::Error, Debug)]
pub enum WriteError {
    #[error("{}", .0)]
    IO(std::io::Error),
    #[error("{}", .0)]
    Anyhow(anyhow::Error),
}
impl From<std::io::Error> for WriteError {
    fn from(value: std::io::Error) -> Self {
        Self::IO(value)
    }
}
impl From<anyhow::Error> for WriteError {
    fn from(value: anyhow::Error) -> Self {
        Self::Anyhow(value)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, bytemuck::Contiguous, bytemuck::NoUninit)]
#[repr(u8)]
pub enum OrphanMode {
    /// The reader should copy this chunk to the output even if it cannot parse it.
    Keep = 0,
    /// The reader should *not* copy this chunk to the output if it cannot parse it
    /// and modifications have been made to the document.
    Discard = 1,
    /// The reader should not parse the document if it cannot parse this chunk.
    Deny = 2,
}
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Version(pub u8, pub u8, pub u8);
impl Version {
    pub const CURRENT: Self = Version(0, 0, 0);
}

#[repr(C)]
pub struct VersionedChunkHeader(Version, OrphanMode);
/// Try to create a versioned chunk header from four bytes.
/// Returns an error only if the final byte is invalid as a [OrphanMode]
impl TryFrom<[u8; 4]> for VersionedChunkHeader {
    type Error = ();
    fn try_from(value: [u8; 4]) -> Result<Self, Self::Error> {
        use bytemuck::Contiguous;
        Ok(Self(
            Version(value[0], value[1], value[2]),
            OrphanMode::from_integer(value[3]).ok_or(())?,
        ))
    }
}

/// Presets for how the writer implementation should behave.
///
/// *All of the strategies should result in a complete file with no unrecoverable data left out!*
///
/// All the underlying writers use `Cautious` implementations - allocations are avoided
/// to the greatest extent reasonable, no global state is accessed, ect.
/// This is therefore a front-end hint (should we thread? should we use the
/// graphics device to assist? ...) and does not effect the backend.
///
/// Implementation is free to fallback on a lower-level strategy should an error occur.
///
/// This is just a sketch for future me to read and implement :3
pub enum IOStrategy {
    /// The writer should be careful not to consume any resources more than
    /// necessary. Useful for if the program is facing exhaustion and we need to
    /// dump the files before the ship goes down. Should not rely on the graphics device and should
    /// write minimum data necessary to be fully recoverable.
    Cautious,
    /// The writer should optimize for speed. Optional elements like thumbnails are left out.
    /// Useful for automated background activities - evicting old open documents to disk, autosaves, ect.
    Fast,
    /// A save in normal circumstances, i.e. user pressed Ctrl+S. We are free to use extra resources,
    /// the graphics device, whatever. Include full optional datas, attempt to thumbnail, all those goodies.
    Normal,
}
const EMPTY_DICT: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// From the given document state reader and repository handle, write a `.fzp` document into the given writer.
pub fn write_into<Document, Writer>(
    document: Document,
    point_repository: &crate::repositories::points::PointRepository,
    writer: Writer,
) -> Result<(), WriteError>
where
    Document: crate::commands::queue::state_reader::CommandQueueStateReader,
    Writer: std::io::Write + std::io::Seek,
{
    use riff::{encode::*, ChunkID};
    use std::io::Write;
    let mut root = BinaryChunkWriter::new_subtype(writer, ChunkID::RIFF, ChunkID::FZP_)?;
    {
        {
            let mut info = BinaryChunkWriter::new_subtype(&mut root, ChunkID::LIST, ChunkID::INFO)?;
            SizedBinaryChunkWriter::write_buf(&mut info, ChunkID(*b"ISFT"), b"fuzzpaint\0")?;
        }
        {
            const TEST_QOI: &'static [u8] = include_bytes!("../test-data/test image.qoi");
            SizedBinaryChunkWriter::write_buf(&mut root, ChunkID::THMB, TEST_QOI)?;
        }
        SizedBinaryChunkWriter::write_buf(&mut root, ChunkID::DOCV, &[])?;
        {
            let mut objs = BinaryChunkWriter::new_subtype(&mut root, ChunkID::LIST, ChunkID::OBJS)?;

            let collections = document.stroke_collections();
            point_repository
                .write_dict_into(
                    collections
                        .0
                        .iter()
                        .flat_map(|collection| collection.1.strokes.iter())
                        .map(|stroke| stroke.point_collection),
                    &mut objs,
                )
                .map_err(|err| -> anyhow::Error { err.into() })?;
            SizedBinaryChunkWriter::write_buf(&mut objs, ChunkID::GRPH, &[])?;
            SizedBinaryChunkWriter::write_buf_subtype(
                &mut objs,
                ChunkID::DICT,
                ChunkID::BRSH,
                &EMPTY_DICT,
            )?;
        }
        SizedBinaryChunkWriter::write_buf(&mut root, ChunkID::HIST, &[])?;
    }

    Ok(())
}

// Todo: explicit bufread support in chunks!
pub fn read_path<Path: Into<std::path::PathBuf>>(
    path: Path,
    point_repository: &crate::repositories::points::PointRepository,
) -> Result<crate::commands::queue::DocumentCommandQueue, std::io::Error> {
    use riff::{decode::*, ChunkID};
    use std::io::{Error as IOError, Read};
    let path_buf = path.into();
    let r = std::io::BufReader::new(std::fs::File::open(&path_buf)?);
    // Dont need to check magic before extracting subchunks. If extracting fails, it
    // must've been bad anyway!
    let root = BinaryChunkReader::new(r)?.into_subchunks()?;
    if root.id() != ChunkID::RIFF || root.subtype_id() != ChunkID::FZP_ {
        return Err(std::io::Error::other("bad file magic"))?;
    }

    root.try_for_each(|subchunk| match subchunk.id() {
        ChunkID::LIST => {
            let subchunk = subchunk.into_subchunks()?;
            match subchunk.subtype_id() {
                ChunkID::INFO => Ok(()),
                ChunkID::OBJS => subchunk.try_for_each(|obj| match obj.id() {
                    ChunkID::DICT => {
                        let dict = obj.into_dict()?;
                        match dict.subtype_id() {
                            ChunkID::PTLS => point_repository.read_dict(dict).map(|_| ()),
                            ChunkID::BRSH => Ok(()),
                            other => Err(IOError::other(anyhow::anyhow!(
                                "Unrecognized dict \"{other}\""
                            ))),
                        }
                    }
                    ChunkID::GRPH => Ok(()),

                    other => Err(IOError::other(anyhow::anyhow!(
                        "Unrecognized obj \"{other}\""
                    ))),
                }),
                other => Err(IOError::other(anyhow::anyhow!(
                    "Unrecognized list \"{other}\""
                ))),
            }
        }
        ChunkID::THMB => Ok(()),
        ChunkID::HIST => Ok(()),
        ChunkID::DOCV => Ok(()),
        other => Err(IOError::other(anyhow::anyhow!(
            "Unrecognized chunk \"{other}\""
        ))),
    })?;

    let document_info = crate::state::Document {
        // File stem (without ext) if available, else the whole path.
        name: path_buf
            .file_stem()
            .map(|p| p.to_string_lossy().to_owned())
            .unwrap_or_else(|| path_buf.to_string_lossy().to_owned())
            .to_string(),
        path: Some(path_buf),
    };
    Ok(crate::commands::queue::DocumentCommandQueue::from_state(
        document_info,
        Default::default(),
        Default::default(),
    ))
}

/// IO utilities not specific to the format.
pub mod common;
pub mod id;
pub mod riff;

/// Data that has been read from a file newer than this
/// version supports, but is marked by the writer as keepable.
pub struct OrphanedData {
    /// TODO: keep track of from where in the RIFF tree this
    /// node belongs. It must have the same parent as it originally had,
    /// but may be placed in any index within that parent.
    position: (),
    id: riff::ChunkID,
    version: Version,
    /// Entire data of the chunk, including header.
    data: Vec<u8>,
}
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

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
#[repr(u8)]
pub enum OrphanMode {
    Keep = 0,
    Discard = 1,
    Deny = 2,
}
impl OrphanMode {
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Keep),
            1 => Some(Self::Discard),
            2 => Some(Self::Deny),
            _ => None,
        }
    }
}
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C, packed)]
pub struct DictMetadata<InnerMeta: bytemuck::Pod + bytemuck::Zeroable + Copy> {
    pub offset: u32,
    pub len: u32,
    pub inner: InnerMeta,
}
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct Version(pub u8, pub u8, pub u8);
impl Version {
    pub const CURRENT: Self = Version(0, 0, 0);
}

#[repr(C)]
pub struct VersionedChunkHeader(Version, OrphanMode);
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
        SizedBinaryChunkWriter::write_buf(&mut root, ChunkID::GRPH, &[])?;
        SizedBinaryChunkWriter::write_buf(&mut root, ChunkID::HIST, &[])?;
        {
            let collections = document.stroke_collections();
            point_repository
                .write_dict_into(
                    collections
                        .0
                        .iter()
                        .flat_map(|collection| collection.1.strokes.iter())
                        .map(|stroke| stroke.point_collection),
                    &mut root,
                )
                .map_err(|err| -> anyhow::Error { err.into() })?;
        }
        SizedBinaryChunkWriter::write_buf_subtype(&mut root, ChunkID::DICT, ChunkID::BRSH, &[])?;
    }

    Ok(())
}

// Todo: explicit bufread support in chunks!
pub fn read_path<Path: Into<std::path::PathBuf>>(
    path: Path,
    point_repository: &crate::repositories::points::PointRepository,
) -> Result<crate::commands::queue::DocumentCommandQueue, std::io::Error> {
    use riff::{decode::*, ChunkID};
    let path_buf = path.into();
    let r = std::io::BufReader::new(std::fs::File::open(&path_buf)?);
    // Dont need to check magic before extracting subchunks. If extracting fails, it
    // must've been bad anyway!
    let mut root = BinaryChunkReader::new(r)?.subchunks()?;
    if root.id() != ChunkID::RIFF || root.subtype_id() != ChunkID::FZP_ {
        return Err(std::io::Error::other("bad file magic"))?;
    }
    // while let Ok(mut subchunk) = root.next_subchunk() {
    //     match subchunk.id() {
    //         ChunkID::THMB => (),
    //         ChunkID::DOCV => (),
    //         ChunkID::LIST => {
    //             // We're goin deeper!
    //             let nested_subchunks = subchunk.subchunks()?;
    //             match nested_subchunks.subtype_id() {
    //                 ChunkID::INFO => (),
    //                 ChunkID::OBJS => (),
    //                 other => unimplemented!("can't parse chunk LIST \"{other}\""),
    //             }
    //         }
    //         ChunkID::HIST => (),
    //         other => unimplemented!("can't parse chunk {other}"),
    //     }
    //     // Advance cursor. This should be automatic, fix your api, me!!!
    //     subchunk.skip()?;
    // }

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

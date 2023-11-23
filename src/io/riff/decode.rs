use super::ChunkID;
use crate::io::common::MyTake;
use az::SaturatingAs;

use std::io::{BufRead, Error as IOError, Read, Result as IOResult, Seek, SeekFrom};
// read an ID, size
fn read_chunk_header(r: &mut impl Read) -> IOResult<(ChunkID, u32)> {
    let mut data = [0; 8];
    r.read_exact(&mut data)?;

    Ok((
        // Both infallible.
        ChunkID(data[0..4].try_into().unwrap()),
        u32::from_le_bytes(data[4..8].try_into().unwrap()),
    ))
}
pub struct BinaryChunkReader<R> {
    id: ChunkID,
    reader: MyTake<R>,
}
impl<R> Read for BinaryChunkReader<R>
where
    MyTake<R>: Read,
{
    // Defer all calls
    fn read(&mut self, buf: &mut [u8]) -> IOResult<usize> {
        self.reader.read(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> IOResult<usize> {
        self.reader.read_vectored(bufs)
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> IOResult<()> {
        self.reader.read_exact(buf)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> IOResult<usize> {
        self.reader.read_to_end(buf)
    }
    fn read_to_string(&mut self, buf: &mut String) -> IOResult<usize> {
        self.reader.read_to_string(buf)
    }
}
impl<R> BufRead for BinaryChunkReader<R>
where
    MyTake<R>: BufRead,
{
    // Defer all calls
    fn fill_buf(&mut self) -> IOResult<&[u8]> {
        self.reader.fill_buf()
    }
    fn consume(&mut self, amt: usize) {
        self.reader.consume(amt)
    }
    fn read_line(&mut self, buf: &mut String) -> IOResult<usize> {
        self.reader.read_line(buf)
    }
    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> IOResult<usize> {
        self.reader.read_until(byte, buf)
    }
}
impl<R> Seek for BinaryChunkReader<R>
where
    MyTake<R>: Seek,
{
    // Defer all calls
    /// Seek the stream within this reader's address space. Seeks past-the-end are clamped.
    fn seek(&mut self, pos: SeekFrom) -> IOResult<u64> {
        self.reader.seek(pos)
    }
    fn stream_position(&mut self) -> IOResult<u64> {
        self.reader.stream_position()
    }
    fn rewind(&mut self) -> IOResult<()> {
        self.reader.rewind()
    }
}

impl<R: Read> BinaryChunkReader<R> {
    /// Read a chunk from the given Read. Immediately fetches 8 bytes
    /// from the stream to get the ID and length.
    pub fn new(mut read: R) -> IOResult<Self> {
        let (id, len) = read_chunk_header(&mut read)?;

        Ok(Self {
            id,
            reader: MyTake::new(read, len as u64),
        })
    }
}
// The bound Seek might be relaxed later, needed due to the need to skip unread bytes and get to the next chunk.
// Sad that Read doesn't have a skip method D:
impl<R> BinaryChunkReader<R>
where
    MyTake<R>: Seek + Read,
{
    /// Interpret the rest of the unstructured payload as RIFF/LIST subchunks with an inner ID.
    pub fn into_subchunks(mut self) -> IOResult<SubchunkReader<R>> {
        let mut inner_id = ChunkID([0; 4]);
        self.reader.read_exact(&mut inner_id)?;

        Ok(SubchunkReader {
            outer_id: self.id,
            inner_id,
            // Cut out already read bytes, including inner ID
            reader: self.reader.retake_remaining(),
        })
    }
    pub fn into_dict(mut self) -> IOResult<DictReader<R>> {
        // Inner-id, versioned header, num entries, sizeof entries
        let mut data = [0; 16];
        self.reader.read_exact(&mut data)?;
        // All unwraps infallible.
        let inner_id = ChunkID(data[0..4].try_into().unwrap());
        let version: [u8; 4] = data[4..8].try_into().unwrap();
        let version = crate::io::VersionedChunkHeader::try_from(version)
            .map_err(|_| IOError::other(anyhow::anyhow!("malformed chunk version header")))?;
        let meta_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let meta_stride = u32::from_le_bytes(data[12..16].try_into().unwrap());

        // Is there actually enough data present for the reported amount of metadata?
        let overflows_chunk_size = meta_stride
            .checked_mul(meta_count)
            .map(|size| size as u64 > self.reader.remaining())
            .unwrap_or(true);
        // Temporary hack to prevent DOS
        if overflows_chunk_size || meta_stride.saturating_as::<usize>() > MAX_METADATA_SIZE {
            return Err(IOError::other("dict metadata too large"));
        }
        // Must have enough space for len + offset
        if meta_stride < 8 && meta_count > 0 {
            return Err(IOError::other("malformed dict header"));
        }

        Ok(DictReader {
            outer_id: self.id,
            inner_id,
            version,
            meta_stride,
            meta_count,
            reader: self.reader.retake_remaining(),
        })
    }
}
impl<R> BinaryChunkReader<R> {
    pub fn id(&self) -> ChunkID {
        self.id
    }
    /// Size of chunk payload
    ///
    /// This value is read directly from the chunk header, and may be wrong or malicious.
    /// Do not trust this to be correct.
    pub fn data_len_unsanitized(&self) -> usize {
        // Unwrap ok - we constructed with a size that was a u32, so should fit!
        self.reader.len() as usize
    }
    /// Size the the chunk including ID and length secti
    ///_ons
    /// This value is calculated directly from the chunk header, and may be wrong or malicious.
    /// Do not trust this to be correct.
    pub fn self_len(&self) -> usize {
        self.reader.len() as usize + 8
    }
}
impl<R: Read + Seek> BinaryChunkReader<R> {
    /// Advance the inner reader to the end of this chunk.
    pub fn skip(self) -> IOResult<()> {
        self.reader.skip().map(|_| ())
    }
}

/// A reader of RIFF subchunks, created via `BinaryChunkWriter::subchunks`
pub struct SubchunkReader<R> {
    outer_id: ChunkID,
    inner_id: ChunkID,
    reader: MyTake<R>,
}
impl<R> SubchunkReader<R>
where
    MyTake<R>: Seek + Read,
{
    /// Visit each subchunk with a closure that returns an IO Result.
    /// Bails if the closure errors or an internal IO error occurs.
    ///
    /// The argument type of F is terribly verbose. Treat it as an opaque type, `impl Read + Seek + ?BufRead + ?Write`,
    /// depending on the innermost stream type.
    pub fn try_for_each<F>(mut self, mut f: F) -> IOResult<()>
    where
        F: FnMut(BinaryChunkReader<&mut MyTake<R>>) -> IOResult<()>,
    {
        while self.reader.remaining() > 0 {
            let cur_position = self.reader.cursor();
            let read = BinaryChunkReader::new(&mut self.reader)?;
            // We can't guaruntee the user will read all the data.
            // Seek to the end of the chunk afterwards.
            let cursor_after = cur_position
                .checked_add(read.self_len() as u64)
                .ok_or_else(|| {
                    IOError::other(anyhow::anyhow!("chunk too long, overflows cursor"))
                })?;
            f(read)?;
            // Seek to end.
            self.reader.seek(SeekFrom::Start(cursor_after))?;
        }
        Ok(())
    }
}
impl<R> SubchunkReader<R> {
    pub fn id(&self) -> ChunkID {
        self.outer_id
    }
    pub fn subtype_id(&self) -> ChunkID {
        self.inner_id
    }
}
/// The maximum allowable size of a metadata field, a hacky impermanent way to prevent a DOS attack
/// from a maliciously crafted `DICT` chunk. Given that MetadataTy is supposed to be a short descriptor
/// of the larger, free-form spillover data, this is perfectly reasonable to limit to even 100 bytes. Alas...
pub const MAX_METADATA_SIZE: usize = 1024 * 1024;
/// Reader of a `DICT` chunk, as specified in the `.fzp` schema.
/// Contains many Metadata entries, each referring to a piece of unstructured binary.
pub struct DictReader<R> {
    /// should be DICT
    outer_id: ChunkID,
    inner_id: ChunkID,

    version: crate::io::VersionedChunkHeader,

    meta_stride: u32,
    meta_count: u32,

    reader: MyTake<R>,
}
impl<R> DictReader<R> {
    /// Outer ID of the chunk reader that was converted to a dict reader. Usually `DICT`.
    pub fn id(&self) -> ChunkID {
        self.outer_id
    }
    /// Identifier for what this DICT contains.
    pub fn subtype_id(&self) -> ChunkID {
        self.inner_id
    }
    /// Length of each metadata entry, in bytes.
    /// None if there is no metadata.
    ///
    /// This value is read directly from the chunk header, and may be wrong or malicious.
    /// Do not trust this to be correct.
    pub fn meta_len_unsanitized(&self) -> Option<std::num::NonZeroUsize> {
        if self.meta_count == 0 {
            None
        } else {
            std::num::NonZeroUsize::new(self.meta_stride.saturating_as::<usize>())
        }
    }
    /// Length of number of metadata entries, in bytes.
    ///
    /// This value is read directly from the chunk header, and may be wrong or malicious.
    /// Do not trust this to be correct.
    pub fn meta_count_unsanitized(&self) -> usize {
        self.meta_count.saturating_as::<usize>()
    }
    /// Length of the metadata section, in bytes.
    ///
    /// This value is calculated directly from the chunk header, and may be wrong or malicious.
    /// Do not trust this to be correct.
    pub fn metas_len_unsanitized(&self) -> usize {
        self.meta_stride
            .saturating_as::<usize>()
            .saturating_mul(self.meta_count.saturating_as())
    }
    /// Length of the spillover data section, in bytes.
    ///
    /// This value is calculated directly from the chunk header, and may be wrong or malicious.
    /// Do not trust this to be correct.
    pub fn spillover_len_unsanitized(&self) -> usize {
        self.reader
            .len()
            .saturating_as::<usize>()
            .saturating_sub(self.metas_len_unsanitized())
    }
    pub fn version(&self) -> crate::io::Version {
        self.version.0
    }
    /// Get the orphan mode of the chunk header. If the version is no recognized,
    /// it is up to the user of this API to respect the OrphanMode!
    pub fn orphan_mode(&self) -> crate::io::OrphanMode {
        self.version.1
    }
}
impl<R> DictReader<R>
where
    MyTake<R>: Read + Seek,
{
    /// Consume the binary of each metadata. On successful reading of every meta,
    /// a BinaryChunkReader is returned to refer to the unstructured spillover area.
    ///
    /// If there are no metadatas, this method is infallible.
    ///
    /// The argument type of F is terribly verbose. Treat it as an opaque type, `impl Read + Seek + ?BufRead + ?Write`,
    /// depending on the innermost stream type.
    pub fn try_for_each<F>(mut self, mut f: F) -> IOResult<BinaryChunkReader<R>>
    where
        F: FnMut(MyTake<&mut MyTake<R>>) -> IOResult<()>,
    {
        for _ in 0..self.meta_count {
            let cursor_after = self
                .reader
                .cursor()
                .checked_add(self.meta_stride as u64)
                .ok_or_else(|| {
                    IOError::other(anyhow::anyhow!("metadata too long, overflows cursor"))
                })?;
            let subtake = MyTake::new(&mut self.reader, self.meta_stride as u64);
            f(subtake)?;
            // Consume untaken bytes.
            self.reader.seek(SeekFrom::Start(cursor_after))?;
        }

        Ok(BinaryChunkReader {
            id: self.inner_id,
            reader: self.reader.retake_remaining(),
        })
    }
}

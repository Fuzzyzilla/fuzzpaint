use std::io::{
    BufRead, Error as IOError, ErrorKind as IOErrorKind, Read, Result as IOResult, Seek, SeekFrom,
    Write,
};

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ChunkID(pub [u8; 4]);
impl ChunkID {
    // RIFF standard chunks
    pub const RIFF: Self = ChunkID(*b"RIFF");
    pub const INFO: Self = ChunkID(*b"INFO");
    pub const LIST: Self = ChunkID(*b"LIST");
    // fuzzpaint custom chunks
    pub const FZP_: Self = ChunkID(*b"fzp ");
    pub const DOCV: Self = ChunkID(*b"docv");
    pub const GRPH: Self = ChunkID(*b"grph");
    pub const PTLS: Self = ChunkID(*b"ptls");
    pub const HIST: Self = ChunkID(*b"hist");
    pub const BRSH: Self = ChunkID(*b"brsh");
    pub fn id_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }
}
impl std::fmt::Display for ChunkID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write as a string if possible, otherwise as a hex string.
        if let Some(str) = self.id_str() {
            f.write_str(str)
        } else {
            write!(f, "{:x?}", self.0)
        }
    }
}
impl std::ops::Deref for ChunkID {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for ChunkID {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A RIFF Chunk composed of unstructured binary.
pub struct BinaryChunkWriter<W: Write + Seek> {
    id: ChunkID,
    /// 4 GiB filesize limit is inherent to RIFF. I don't think this will be an issue, and if it
    /// becomes one it's a hint that something more efficient is needed lol
    cursor: u32,
    len: u32,
    needs_len_flush: bool,
    writer: W,
}
impl<W: Write + Seek> BinaryChunkWriter<W> {
    /// Creates a writer with the given header and dynamic length.
    pub fn new(mut writer: W, id: ChunkID) -> IOResult<Self> {
        // Write the ID and zeroed length field.
        let start_data: [u8; 8] = [id[0], id[1], id[2], id[3], 0, 0, 0, 0];
        writer.write_all(&start_data)?;

        Ok(Self {
            id,
            cursor: 0,
            len: 0,
            needs_len_flush: false,
            writer,
        })
    }
    /// Creates a writer with LIST or RIFF style subtype header.
    // Should this be it's own writer type?
    pub fn new_subtype(mut writer: W, id: ChunkID, subtype: ChunkID) -> IOResult<Self> {
        // Write the ID, length field set to 4, and subtype.
        #[rustfmt::skip]
        let start_data: [u8; 12] = [
            id[0], id[1], id[2], id[3],
            // Little endian 4u32
            4, 0, 0, 0,
            subtype[0], subtype[1], subtype[2], subtype[3],
        ];
        writer.write_all(&start_data)?;

        Ok(Self {
            id,
            cursor: 4,
            len: 4,
            needs_len_flush: false,
            writer,
        })
    }
    pub fn id(&self) -> ChunkID {
        self.id
    }
    /// Flush the len field of this chunk. Prefer this over Self::drop, as it will report errors.
    /// In the event of an error, the status of the inner writer is unknown.
    pub fn update_len(&mut self) -> IOResult<()> {
        let length_offs = self.cursor as i64;

        self.writer.seek(SeekFrom::Current(-length_offs - 4))?;
        self.writer.write(&self.len.to_le_bytes())?;
        self.writer.seek(SeekFrom::Current(length_offs))?;

        self.needs_len_flush = false;
        Ok(())
    }
}

impl<W: Write + Seek> Drop for BinaryChunkWriter<W> {
    fn drop(&mut self) {
        // Inspired by std::io::BufWriter, errors at implicit closure are ignored.
        if self.needs_len_flush {
            let _ = self.update_len();
        }
    }
}
impl<W: Write + Seek> Write for BinaryChunkWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IOResult<usize> {
        let written = self.writer.write(buf)?;
        // Closure to lazily create an error object
        let err = || IOError::other(anyhow::anyhow!("RIFF chunk {} exceeds 4GiB", self.id));
        // Check that self.size + written doesn't overflow u32
        self.cursor = written
            .try_into()
            .ok()
            .and_then(|written: u32| self.cursor.checked_add(written))
            .ok_or_else(err)?;
        self.len = self.len.max(self.cursor);

        self.needs_len_flush = true;
        Ok(written)
    }
    fn flush(&mut self) -> IOResult<()> {
        self.update_len()?;
        self.writer.flush()
    }
    // TODO: write_vectored
}
impl<W: Write + Seek> Seek for BinaryChunkWriter<W> {
    /// Seek the stream within this reader's address space. Behavior of seeks past-the-end are deferred to the writer.
    fn seek(&mut self, pos: SeekFrom) -> IOResult<u64> {
        let add_with_overflow = || IOError::other(anyhow::anyhow!("seek with overflow"));
        let new_cursor: i64 = match pos {
            SeekFrom::End(delta) => (self.len as i64).checked_add(delta),
            SeekFrom::Current(delta) => (self.cursor as i64).checked_add(delta),
            SeekFrom::Start(delta) => {
                // Overflow with None if too large to fit
                delta.try_into().ok()
            }
        }
        .ok_or_else(add_with_overflow)?;

        if new_cursor < 0 {
            return Err(IOError::other(anyhow::anyhow!("seek past-the-start")));
        }
        let Ok(new_cursor): Result<u32, _> = new_cursor.try_into() else {
            return Err(add_with_overflow())?;
        };

        let delta = new_cursor as i64 - self.cursor as i64;
        self.writer.seek(SeekFrom::Current(delta))?;

        self.cursor = new_cursor;
        Ok(self.cursor as u64)
    }
}
pub struct BinaryChunkReader<R: Read> {
    id: ChunkID,
    /// How far into the chunk we've read. Zero is the basis
    /// for Seeks, and reads will EOF at cursor == len.
    cursor: u32,
    len: u32,
    reader: R,
}
impl<R: Read> Read for BinaryChunkReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> IOResult<usize> {
        let max_bytes = self.len - self.cursor;
        let clamped_buf_len = buf.len().min(max_bytes as usize);
        let clamped_buf = &mut buf[..clamped_buf_len];

        let num_read = self.reader.read(clamped_buf)?;

        // As cast and addition OK as we clampled the buf size.
        self.cursor += num_read as u32;
        debug_assert!(self.cursor <= self.len);

        Ok(num_read)
    }
}
impl<R: Read + Seek> Seek for BinaryChunkReader<R> {
    /// Seek the stream within this reader's address space. Seeks past-the-end are clamped.
    fn seek(&mut self, pos: SeekFrom) -> IOResult<u64> {
        let new_cursor = match pos {
            SeekFrom::Current(delta) => {
                // Signed add to find absolute position
                (self.cursor as i64).checked_add(delta)
            }
            SeekFrom::End(delta) => {
                // Signed add to find absolute position
                (self.len as i64).checked_add(delta)
            }
            SeekFrom::Start(pos) => pos.try_into().ok(),
        }
        .ok_or_else(|| IOError::other(anyhow::anyhow!("seek with overflow")))?;
        // Seek-before-start is an error
        if new_cursor < 0 {
            return Err(IOError::other(anyhow::anyhow!("seek past-the-start")));
        }
        // Clamp to end
        let new_cursor = new_cursor.min(self.len as i64) as u32;
        let diff = new_cursor as i64 - self.cursor as i64;

        // Seek underlying stream by the clamped diff, update own cursor.
        self.reader.seek(SeekFrom::Current(diff))?;
        self.cursor = new_cursor;

        Ok(self.cursor as u64)
    }
}
/* /// How to clamp the inner BufRead from reading past chunk EOF?
/// Perhaps BufReader<ChunkReader> is correct instead of ChunkReader<BufRead>
impl<R: BufRead> BufRead for ChunkReader<R> {
    fn fill_buf(&mut self) -> IOResult<&[u8]> {
        self.reader.fill_buf()
    }
    fn consume(&mut self, amt: usize) {
        self.reader.consume(amt)
    }
}*/
impl<R: Read> BinaryChunkReader<R> {
    /// Read a chunk from the given Read. Immediately fetches 8 bytes
    /// from the stream to get the ID and length.
    pub fn new(mut read: R) -> IOResult<Self> {
        let mut id = ChunkID([0; 4]);
        let mut le_len = [0; 4];
        if read.read(&mut id)? != 4 {
            return Err(IOError::new(
                IOErrorKind::UnexpectedEof,
                anyhow::anyhow!("not enough bytes to read chunk ID"),
            ));
        }
        if read.read(&mut le_len)? != 4 {
            return Err(IOError::new(
                IOErrorKind::UnexpectedEof,
                anyhow::anyhow!("not enough bytes to read chunk length"),
            ));
        }

        Ok(Self {
            id,
            cursor: 0,
            len: u32::from_le_bytes(le_len),
            reader: read,
        })
    }
    pub fn id(&self) -> ChunkID {
        self.id
    }
    /// Size of chunk payload
    pub fn data_len(&self) -> usize {
        self.len as usize
    }
    /// Size the the chunk including ID and length sections
    pub fn self_len(&self) -> usize {
        self.data_len() + 8
    }
    /// Interpret the unstructured payload as RIFF/LIST subchunks.
    pub fn subchunks(mut self) -> IOResult<SubchunkReader<R>> {
        let mut inner_id = ChunkID([0; 4]);
        if self.read(&mut inner_id)? != 4 {
            return Err(IOError::new(
                IOErrorKind::UnexpectedEof,
                anyhow::anyhow!("failed to read subchunk id"),
            ));
        };
        Ok(SubchunkReader {
            inner_id,
            reader: self,
        })
    }
}
impl<R: Read + Seek> BinaryChunkReader<R> {
    /// Advance the inner reader to the end of this chunk.
    pub fn skip(mut self) -> IOResult<()> {
        let remaining = self.len - self.cursor;
        self.reader.seek(SeekFrom::Current(remaining as i64))?;

        Ok(())
    }
}

pub struct SubchunkReader<R: Read> {
    inner_id: ChunkID,
    reader: BinaryChunkReader<R>,
}
impl<R: Read> SubchunkReader<R> {
    pub fn id(&self) -> ChunkID {
        self.reader.id()
    }
    pub fn subtype_id(&self) -> ChunkID {
        self.inner_id
    }
    pub fn data_len(&self) -> usize {
        self.reader.data_len()
    }
    pub fn self_len(&self) -> usize {
        self.reader.self_len()
    }
    /// Read a subchunk at the current position. Use [BinaryChuynkReader::skip] to advance.
    pub fn next_subchunk<'s>(
        &'s mut self,
    ) -> IOResult<BinaryChunkReader<&mut BinaryChunkReader<R>>> {
        BinaryChunkReader::new(&mut self.reader)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::Cursor;
    const EMPTY_FZP: &'static [u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/empty.fzp"));
    /// Test that a handwritten fzp document can be parsed, NOT that a document can be assembled from the data.
    #[test]
    fn read_empty_fzp_riff() {
        let file = Cursor::new(EMPTY_FZP);
        let root = BinaryChunkReader::new(file).unwrap();
        let outer_chunk_size = root.data_len();
        // Root node should be a RIFF element.
        assert_eq!(root.id(), ChunkID(*b"RIFF"),);
        // Start with RIFF inner id (4 bytes)
        let mut total_inner_size = 4;
        // Read inner chunks
        let mut subchunks = root.subchunks().unwrap();

        // RIFF subtype is fuzzpaint file
        assert_eq!(subchunks.subtype_id(), ChunkID(*b"fzp "),);

        // Names we're looking for. Remove as we go, and check that none
        // remain at the end.
        let mut names = vec![
            ChunkID::LIST,
            ChunkID::DOCV,
            ChunkID::GRPH,
            ChunkID::PTLS,
            ChunkID::HIST,
            ChunkID::BRSH,
        ];

        while let Ok(sub) = subchunks.next_subchunk() {
            // Check that the list contains this id, remove it.
            let this_id = sub.id();
            let Some(idx) = names.iter().position(|chunk| *chunk == this_id) else {
                panic!("unexpected chunk id {this_id}")
            };
            names.remove(idx);

            // Total up running length.
            total_inner_size += sub.self_len();
            sub.skip().unwrap();
        }

        // Ensure that sum of subchunk's size is equal to RIFF's reported size.
        assert_eq!(total_inner_size, outer_chunk_size);
        assert_eq!(&names[..], &[]);
    }
    /// Test that a generated empty RIFF file has the structure we expect, NOT that an empty document generates such data.
    #[test]
    fn write_empty_riff() {
        let mut file = Vec::<u8>::new();
        let writer = Cursor::new(&mut file);

        {
            let mut root =
                BinaryChunkWriter::new_subtype(writer, ChunkID::RIFF, ChunkID::FZP_).unwrap();
            {
                let _ = BinaryChunkWriter::new_subtype(&mut root, ChunkID::LIST, ChunkID::INFO)
                    .unwrap();
                let _ = BinaryChunkWriter::new(&mut root, ChunkID::DOCV).unwrap();
                let _ = BinaryChunkWriter::new(&mut root, ChunkID::GRPH).unwrap();
                let _ = BinaryChunkWriter::new(&mut root, ChunkID::PTLS).unwrap();
                let _ = BinaryChunkWriter::new(&mut root, ChunkID::HIST).unwrap();
                let _ = BinaryChunkWriter::new(&mut root, ChunkID::BRSH).unwrap();
            }
        }

        assert_eq!(&file, EMPTY_FZP)
    }
}

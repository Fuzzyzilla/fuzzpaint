use std::io::{
    BufRead, Error as IOError, ErrorKind as IOErrorKind, Read, Result as IOResult, Seek, SeekFrom,
    Write,
};

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ChunkID(pub [u8; 4]);
impl ChunkID {
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
struct BinaryChunkWriter<W: Write + Seek> {
    id: ChunkID,
    /// 4 GiB filesize limit is inherent to RIFF. I don't think this will be an issue, and if it
    /// becomes one it's a hint that something more efficient is needed lol
    len: u32,
    needs_len_flush: bool,
    writer: W,
}
impl<W: Write + Seek> BinaryChunkWriter<W> {
    pub fn new(mut writer: W, id: ChunkID) -> IOResult<Self> {
        // Write the ID and zeroed length field.
        let start_data: [u8; 8] = [id[0], id[1], id[2], id[3], 0, 0, 0, 0];
        writer.write_all(&start_data)?;

        Ok(Self {
            id,
            len: 0,
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
        let length_offs = self.len as i64;

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
        self.len = written
            .try_into()
            .ok()
            .and_then(|written: u32| self.len.checked_add(written))
            .ok_or_else(err)?;

        self.needs_len_flush = true;
        Ok(written)
    }
    fn flush(&mut self) -> IOResult<()> {
        self.update_len()?;
        self.writer.flush()
    }
    // TODO: write_vectored
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
    fn seek(&mut self, pos: SeekFrom) -> IOResult<u64> {
        let new_cursor = match pos {
            SeekFrom::Current(delta) => {
                // Signed add to find absolute position
                (self.cursor as i64).checked_add(delta).ok_or_else(|| {
                    IOError::other(anyhow::anyhow!("Attempt to seek with overflow"))
                })?
            }
            SeekFrom::End(delta) => {
                // Signed add to find absolute position
                (self.len as i64).checked_add(delta).ok_or_else(|| {
                    IOError::other(anyhow::anyhow!("Attempt to seek with overflow"))
                })?
            }
            SeekFrom::Start(pos) => pos as i64,
        };
        // Seek-before-start is an error
        if new_cursor < 0 {
            return Err(IOError::other(anyhow::anyhow!(
                "Attempt to seek past start of chunk"
            )));
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
                anyhow::anyhow!("Not enough bytes to read chunk ID"),
            ));
        }
        if read.read(&mut le_len)? != 4 {
            return Err(IOError::new(
                IOErrorKind::UnexpectedEof,
                anyhow::anyhow!("Not enough bytes to read chunk length"),
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
                anyhow::anyhow!("failed to read inner_id"),
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
    pub fn inner_id(&self) -> ChunkID {
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
    #[test]
    fn emtpy_fzp() {
        use super::*;
        let file = std::io::Cursor::new(include_bytes!("../test-data/empty.fzp"));
        let chunk = BinaryChunkReader::new(file).unwrap();
        let outer_chunk_size = chunk.data_len();
        // Root node should be a RIFF element.
        assert_eq!(chunk.id(), ChunkID(*b"RIFF"),);
        // Start with RIFF inner id (4 bytes)
        let mut total_inner_size = 4;
        // Read inner chunks
        let mut subchunks = chunk.subchunks().unwrap();

        // RIFF subtype is fuzzpaint file
        assert_eq!(subchunks.inner_id(), ChunkID(*b"fzp "),);

        while let Ok(sub) = subchunks.next_subchunk() {
            total_inner_size += sub.self_len();
            sub.skip().unwrap();
        }

        // Ensure that sum of subchunk's size is equal to RIFF's reported size.
        assert_eq!(total_inner_size, outer_chunk_size);
    }
}

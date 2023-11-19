use super::ChunkID;
use crate::io::common::MyTake;
use az::CheckedAs;
use std::io::{
    Error as IOError, ErrorKind as IOErrorKind, Read, Result as IOResult, Seek, SeekFrom,
};
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

        // Add to cursor, ensure that inner reader didn't do a silly.
        self.cursor += num_read.checked_as::<u32>().ok_or_else(|| {
            IOError::other(anyhow::anyhow!(
                "internal reader violated len requirements!"
            ))
        })?;
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
            SeekFrom::Start(pos) => pos.checked_as(),
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
    fn stream_position(&mut self) -> IOResult<u64> {
        Ok(self.cursor as u64)
    }
    /*fn stream_len(&mut self) -> IOResult<u64> {
        Ok(self.len as u64)
    }*/
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

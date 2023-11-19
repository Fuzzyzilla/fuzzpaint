use super::ChunkID;
use crate::io::common::MyTake;
use az::CheckedAs;
use std::io::{Error as IOError, Result as IOResult, Seek, SeekFrom, Write};
/// A RIFF Chunk composed of unstructured binary, with fixed size in bytes.
/// This relaxes the Seek bound on the writer and allows for greater
/// efficiency uwu
/// Bytes not written before the writer is dropped are zeroed. This could change at any time.
///
/// EOF will be signaled upon reaching the end of the chunk.
pub struct SizedBinaryChunkWriter<W: Write> {
    id: ChunkID,
    len_remaining: u32,
    writer: W,
}
impl<W: Write> SizedBinaryChunkWriter<W> {
    const ZEROS: &'static [u8] = &[0; 1024];
    pub fn new(mut writer: W, id: ChunkID, len: usize) -> IOResult<Self> {
        // Hide the fact that we can only store u32::MAX bytes as an implementation detail.
        let len: u32 = len
            .checked_as()
            .ok_or_else(|| IOError::other(anyhow::anyhow!("RIFF chunk {} exceeded 4GiB", id)))?;
        let len_le = len.to_le_bytes();
        #[rustfmt::skip]
        let start_data = [
            id[0],     id[1],     id[2],     id[3],
            len_le[0], len_le[1], len_le[2], len_le[3],
        ];
        writer.write_all(&start_data)?;

        Ok(Self {
            id,
            len_remaining: len,
            writer,
        })
    }
    /// Make a new chunk with a subtype header included. subtype is NOT included in `len`
    pub fn new_subtype(mut writer: W, id: ChunkID, subtype: ChunkID, len: usize) -> IOResult<Self> {
        // Hide the fact that we can only store u32::MAX bytes as an implementation detail.
        // Add four bytes for the subtype id.
        let len: u32 = len
            .checked_as::<u32>()
            // 4 more bytes for inner ID
            .and_then(|len| len.checked_add(4))
            .ok_or_else(|| IOError::other(anyhow::anyhow!("RIFF chunk {} exceeded 4GiB", id)))?;

        let len_le = len.to_le_bytes();
        #[rustfmt::skip]
        let start_data = [
            id[0],      id[1],      id[2],      id[3],
            len_le[0],  len_le[1],  len_le[2],  len_le[3],
            subtype[0], subtype[1], subtype[2], subtype[3],
        ];
        writer.write_all(&start_data)?;

        Ok(Self {
            id,
            // We already wrote 4 bytes for subtype id
            len_remaining: len - 4,
            writer,
        })
    }
    /// Convenience fn to place a whole buffer as a SizedBinaryChunk
    pub fn write_buf(mut writer: W, id: ChunkID, data: &[u8]) -> IOResult<()> {
        let len: u32 = data
            .len()
            .checked_as()
            .ok_or_else(|| IOError::other(anyhow::anyhow!("RIFF chunk {} exceeded 4GiB", id)))?;

        let len_le = len.to_le_bytes();
        #[rustfmt::skip]
        let start_data = [
            id[0],     id[1],     id[2],     id[3],
            len_le[0], len_le[1], len_le[2], len_le[3],
        ];

        let mut slices = [
            std::io::IoSlice::new(&start_data),
            std::io::IoSlice::new(data),
        ];

        writer.write_all_vectored(&mut slices)
    }
    /// Convenience fn to place a whole buffer as a SizedBinaryChunk subtype
    pub fn write_buf_subtype(
        mut writer: W,
        id: ChunkID,
        subtype: ChunkID,
        data: &[u8],
    ) -> IOResult<()> {
        let len: u32 = data
            .len()
            .checked_as::<u32>()
            // 4 more bytes for inner ID
            .and_then(|len| len.checked_add(4))
            .ok_or_else(|| IOError::other(anyhow::anyhow!("RIFF chunk {} exceeded 4GiB", id)))?;

        let len_le = len.to_le_bytes();
        #[rustfmt::skip]
        let start_data = [
            id[0],      id[1],      id[2],      id[3],
            len_le[0],  len_le[1],  len_le[2],  len_le[3],
            subtype[0], subtype[1], subtype[2], subtype[3],
        ];

        let mut slices = [
            std::io::IoSlice::new(&start_data),
            std::io::IoSlice::new(data),
        ];

        writer.write_all_vectored(&mut slices)
    }
    /// Same as Drop, but is able to report errors.
    pub fn finish(mut self) -> IOResult<()> {
        self.pad()?;
        self.flush()
        // Immediately drops, which shouldn't re-pad as len is set to zero within self::pad
    }
    /// Write padding bytes to finish the write operation.
    fn pad(&mut self) -> IOResult<()> {
        if self.len_remaining != 0 {
            let slice = std::io::IoSlice::new(Self::ZEROS);
            let (quotient, remainder) = (
                self.len_remaining as usize / Self::ZEROS.len(),
                self.len_remaining as usize % Self::ZEROS.len(),
            );
            // Fill a vec with enough slices
            let num_slices = quotient + if remainder != 0 { 1 } else { 0 };
            let mut slices = Vec::with_capacity(num_slices);
            // `quotient` full slices...
            slices.extend(std::iter::repeat(slice).take(quotient));
            // plus one partial slice if there's a remainder.
            if remainder != 0 {
                slices.push(std::io::IoSlice::new(&Self::ZEROS[..remainder]));
            }

            // Set to zero before fallible call.
            // That way we don't double-drop.
            self.len_remaining = 0;
            self.writer.write_all_vectored(&mut slices)?;
        }
        Ok(())
    }
}
impl<W: Write> Write for SizedBinaryChunkWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IOResult<usize> {
        let clamped_len = (self.len_remaining as usize).min(buf.len());
        // Explicit hint that the stream is full.
        if clamped_len == 0 {
            return Ok(0);
        }
        let trimmed_buf = &buf[..clamped_len];
        let written = self.writer.write(trimmed_buf)?;
        self.len_remaining = written
            .checked_as()
            .and_then(|written| self.len_remaining.checked_sub(written))
            .ok_or_else(|| IOError::other("inner writer overflowed chunk"))?;

        Ok(written)
    }
    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> IOResult<usize> {
        // Todo: pre-clamp length.
        if self.len_remaining == 0 {
            return Ok(0);
        }
        let written = self.writer.write_vectored(bufs)?;
        self.len_remaining = written
            .checked_as()
            .and_then(|written| self.len_remaining.checked_sub(written))
            .ok_or_else(|| IOError::other("inner writer overflowed chunk"))?;

        Ok(written)
    }
    fn flush(&mut self) -> IOResult<()> {
        self.writer.flush()
    }
}
impl<W: Write> Drop for SizedBinaryChunkWriter<W> {
    fn drop(&mut self) {
        // We have to pad out the rest of the chunk to match the set length!
        // Generally a bad idea...
        if self.len_remaining != 0 {
            log::warn!("padding in SizedBinaryChunkWriter dtor!");
            if let Err(e) = self.pad() {
                log::error!("error while padding in dtor: {e:?}")
            }
        }
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
    /// Creates a writer with LIST, RIFF, or DICT style subtype header.
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
        // Check that self.size + written doesn't overflow u32
        self.cursor = written
            .checked_as()
            .and_then(|written| self.cursor.checked_add(written))
            .ok_or_else(|| IOError::other("inner writer overflowed chunk"))?;
        self.len = self.len.max(self.cursor);

        self.needs_len_flush = true;
        Ok(written)
    }
    fn flush(&mut self) -> IOResult<()> {
        self.update_len()?;
        self.writer.flush()
    }
    /*
    fn is_write_vectored(&self) -> bool {
        self.writer.is_write_vectored()
    }*/
    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> IOResult<usize> {
        let written = self.writer.write_vectored(bufs)?;
        // Check that self.size + written doesn't overflow u32
        self.cursor = written
            .checked_as()
            .and_then(|written| self.cursor.checked_add(written))
            .ok_or_else(|| IOError::other("inner writer overflowed chunk"))?;
        self.len = self.len.max(self.cursor);

        self.needs_len_flush = true;
        Ok(written)
    }
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
                delta.checked_as()
            }
        }
        .ok_or_else(add_with_overflow)?;

        if new_cursor < 0 {
            return Err(IOError::other(anyhow::anyhow!("seek past-the-start")));
        }
        let Some(new_cursor) = new_cursor.checked_as() else {
            return Err(add_with_overflow())?;
        };

        let delta = new_cursor as i64 - self.cursor as i64;
        self.writer.seek(SeekFrom::Current(delta))?;

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

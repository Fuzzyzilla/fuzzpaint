use super::ChunkID;
use crate::io::common::MyTake;
use az::CheckedAs;
use std::io::{Error as IOError, Result as IOResult, Seek, SeekFrom, Write};

/// A RIFF Chunk composed of unstructured binary, with fixed size in bytes.
/// This relaxes the Seek bound on the writer and allows for greater
/// efficiency uwu
///
/// EOF will be signaled upon reaching the end of the chunk's allocated size.
pub struct SizedBinaryChunkWriter<W: Write> {
    id: ChunkID,
    writer: MyTake<W>,
}
impl<W> SizedBinaryChunkWriter<W>
where
    W: Write,
{
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
            writer: MyTake::new(writer, len as u64),
        })
    }
    /// Make a new chunk with a subtype header included. subtype is automatically added and not to be included in `len`
    pub fn new_subtype(mut writer: W, id: ChunkID, subtype: ChunkID, len: usize) -> IOResult<Self> {
        // Hide the fact that we can only store u32::MAX bytes as an implementation detail.
        // Add four bytes for the subtype id.
        let len_in_header: u32 = len
            .checked_as::<u32>()
            // 4 more bytes for inner ID
            .and_then(|len| len.checked_add(4))
            .ok_or_else(|| IOError::other(anyhow::anyhow!("RIFF chunk {} exceeded 4GiB", id)))?;

        let len_le = len_in_header.to_le_bytes();
        #[rustfmt::skip]
        let start_data = [
            id[0],      id[1],      id[2],      id[3],
            len_le[0],  len_le[1],  len_le[2],  len_le[3],
            subtype[0], subtype[1], subtype[2], subtype[3],
        ];
        writer.write_all(&start_data)?;

        Ok(Self {
            id,
            // We already wrote len + 4 for header for the subtype
            // now take requested len.
            writer: MyTake::new(writer, len as u64),
        })
    }
    pub fn id(&self) -> ChunkID {
        self.id
    }
    /// Overwrite from the current cursor up until the end of the block with arbitrary padding.
    /// Allows handling errors that would usually be silent on drop.
    ///
    /// Prefer `self::seek(SeekFrom::End(0))` if available.
    pub fn pad_slow(&mut self) -> IOResult<()> {
        self.writer.pad_slow()
    }
}
impl<W: Write> Write for SizedBinaryChunkWriter<W>
where
    MyTake<W>: Write,
{
    // Defer all calls.
    fn write(&mut self, buf: &[u8]) -> IOResult<usize> {
        self.writer.write(buf)
    }
    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> IOResult<usize> {
        self.writer.write_vectored(bufs)
    }
    fn write_all(&mut self, buf: &[u8]) -> IOResult<()> {
        self.writer.write_all(buf)
    }
    fn write_all_vectored(&mut self, bufs: &mut [std::io::IoSlice<'_>]) -> IOResult<()> {
        self.writer.write_all_vectored(bufs)
    }
    fn flush(&mut self) -> IOResult<()> {
        self.writer.flush()
    }
    fn write_fmt(&mut self, fmt: std::fmt::Arguments<'_>) -> IOResult<()> {
        self.writer.write_fmt(fmt)
    }
}
impl<W: Write> Seek for SizedBinaryChunkWriter<W>
where
    MyTake<W>: Seek,
{
    // Defer all calls.
    fn rewind(&mut self) -> IOResult<()> {
        self.writer.rewind()
    }
    fn seek(&mut self, pos: SeekFrom) -> IOResult<u64> {
        self.writer.seek(pos)
    }
    fn stream_position(&mut self) -> IOResult<u64> {
        self.writer.stream_position()
    }
}
impl<W: Write> Drop for SizedBinaryChunkWriter<W> {
    fn drop(&mut self) {
        // I reallllly need specialization here, not only is this slow when W: Seek it's is an incorrect
        // implementation when W: Seek. If the user seeks to the beginning and then drops, all the data they
        // wrote gets clobbered by this naive padding impl. >:(
        if self.writer.remaining() != 0 {
            // Best-practices warning
            #[cfg(debug_assertions)]
            log::warn!("Padding in SizedBinaryChunkWriter dtor!");

            if let Err(e) = self.writer.pad_slow() {
                log::error!("Error in SizedBinaryChunkWriter dtor: {e}");
            }
        }
    }
}
/// A RIFF Chunk composed of unstructured binary with dynamic length.
/// In order to update the length, a Seek implementation of the underlying writer is required.
pub struct BinaryChunkWriter<W: Seek + Write> {
    id: ChunkID,

    cursor: u32,
    len: u32,
    needs_len_flush: bool,

    writer: W,
}
impl<W: Write + Seek> BinaryChunkWriter<W> {
    /// Creates a writer with the given header and dynamic length.
    pub fn new(mut writer: W, id: ChunkID) -> IOResult<Self> {
        // Write the ID and zeroed length field.
        #[rustfmt::skip]
        let start_data: [u8; 8] = [
            id[0], id[1], id[2], id[3],
            0,     0,     0,     0,
        ];
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
    pub fn new_subtype(mut writer: W, id: ChunkID, sub_id: ChunkID) -> IOResult<Self> {
        // Write the ID, length field set to 4, and subtype.
        #[rustfmt::skip]
        let start_data: [u8; 12] = [
            id[0],     id[1],     id[2],     id[3],
            // Little endian 4u32 for the subtype
            4,         0,         0,         0,
            sub_id[0], sub_id[1], sub_id[2], sub_id[3],
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
        // Seek to -4, in local address. that's where the len field is!
        // -u32::MAX - 4 will always fit in i64
        let length_offs = self.cursor as i64;
        self.writer.seek(SeekFrom::Current(-length_offs - 4))?;
        // Write len
        self.writer.write_all(&self.len.to_le_bytes())?;
        self.needs_len_flush = false;
        // Return cursor
        self.writer.seek(SeekFrom::Current(length_offs))?;
        Ok(())
    }
}
impl<W: Write + Seek> Drop for BinaryChunkWriter<W> {
    /// Flushes the writer if needed. Errors are printed to the error stream,
    /// prefer `update_len` and `flush` instead to catch such errors!
    fn drop(&mut self) {
        if self.needs_len_flush {
            // Give a best-practices warning
            #[cfg(debug_assertions)]
            log::warn!("Flushing in BinaryChunkWriter dtor.");

            // Flush and report errors.
            if let Err(e) = self.update_len() {
                log::error!("Error in BinaryChunkWriter dtor: {e}");
            }
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
        // Simpler impl than MyTake's, due to the fact that the cursor is u32.
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
}

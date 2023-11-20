use az::{CheckedAs, SaturatingAs};
use std::io::{BufRead, Error as IOError, Read, Result as IOResult, Seek, SeekFrom, Write};

/// std::io::Take, except it's Seek when S:Seek. Not sure why std's isn't D:
///
/// Works with readers, writers, BufReaders, all three, whatever!
///
/// If the base reader is Seek, it shifts the basis of it
/// such that the position at the time of MyTake's construction is the start,
/// and that position + len is the end. Seeks past-the-end are clamped to the end.
///
/// There are numerous ways to jailbreak this (for example, since Read + Write are safe traits, they may report incorrect
/// return values for read/write allowing the cursor to drift out-of-bounds), being 100% watertight is a non-goal
/// and it's instead implemented on a best-effort basis.
pub struct MyTake<S> {
    stream: S,
    cursor: u64,
    len: u64,
}
impl<S> MyTake<S> {
    /// Create a new taker, taking `len` bytes past the *current* stream position.
    ///
    /// No check is performed that the stream is actually long enough!
    pub fn new(stream: S, len: u64) -> Self {
        Self {
            stream,
            len,
            cursor: 0,
        }
    }
    /// How many bytes remain
    pub fn remaining(&self) -> u64 {
        self.len
            .checked_sub(self.cursor)
            // Invariant must be upheld everywhere else.
            .expect("cursor past the end")
    }
    /// Consume the Take, returning the inner stream. The cursor is not touched.
    ///
    /// To instead consume and jump to the end, use MyTake::skip.
    pub fn into_inner(self) -> S {
        self.stream
    }
    /// Returns the current stream position, in bytes since the point where the MyTake was constructed.
    pub fn cursor(&self) -> u64 {
        self.cursor
    }
    /// Returns the total length of the stream, exactly as passed into `new` regardless of stream position.
    ///
    /// For length remaining, use `remaining`.
    pub fn len(&self) -> u64 {
        self.len
    }
    /// Consumes this take, returning a smaller one from the current position.
    /// Fails and returns ownership of self if the requested length is longer than the remaining length.
    ///
    /// The resulting MyTake is faster than constructing a new MyTake around `self`
    /// as it reduces the recursion needed for stream operations, with the limitation that the original MyTake
    /// cannot be recovered.
    pub fn retake(self, len: u64) -> Result<MyTake<S>, Self> {
        if len > self.remaining() {
            Err(self)
        } else {
            Ok(Self {
                stream: self.stream,
                cursor: 0,
                len,
            })
        }
    }
    /// Create a new take from the current position up to the end of this take.
    pub fn retake_remaining(self) -> Self {
        Self {
            len: self.remaining(),
            stream: self.stream,
            cursor: 0,
        }
    }
}
impl<S: Seek> MyTake<S> {
    /// Advance the cursor to the end, returning the stream.
    /// If remaining > i64::MAX, will require two seeks.
    ///
    /// In case of an error, stream state is left undefined and is lost.
    pub fn skip(self) -> IOResult<S> {
        let remaining = self.remaining();
        // Skip a syscall. This optimization used a lot by the riff decoders.
        if remaining == 0 {
            return Ok(self.stream);
        }
        let mut stream = self.stream;
        let iremaining: i64 = remaining.saturating_as();

        // Seek from current by a u64.
        // Seek takes i64, so we might have to do two seeks to reach u64::MAX
        stream.seek(SeekFrom::Current(iremaining))?;
        if (iremaining as u64) < remaining {
            // Have to seek again :V
            let iremaining = (remaining.saturating_sub(i64::MAX as u64)).saturating_as();
            stream.seek(SeekFrom::Current(iremaining))?;
        }

        Ok(stream)
    }
}
impl<R: Read> Read for MyTake<R> {
    fn read(&mut self, buf: &mut [u8]) -> IOResult<usize> {
        let buf = trim_buf_mut(buf, self.remaining());
        // Short circuit if we can't read any more data
        if buf.is_empty() {
            return Ok(0);
        }
        let num_read = self.stream.read(buf)?;
        // Defensive checks for bad inner reader impl
        // (or my own bugs :P)
        let new_cursor = self
            .cursor
            .checked_add(num_read as u64)
            .filter(|new| *new <= self.len)
            .ok_or_else(|| IOError::other("inner reader overflowed MyTake cursor"))?;
        self.cursor = new_cursor;

        Ok(num_read)
    }
    // todo: fast read_vectored impl
    // todo: eargerly fail read_all, read_all_vectored if too long
}
impl<R: BufRead> BufRead for MyTake<R> {
    fn consume(&mut self, amt: usize) {
        // Only allow consuming as much as we're allowed to view.
        let trimmed_amt = (amt as u64).min(self.remaining());
        self.cursor = self
            .cursor
            .checked_add(trimmed_amt)
            .expect("consume overflowed cursor");
        debug_assert!(self.cursor <= self.len);

        let trimmed_amt: usize = trimmed_amt.saturating_as();
        self.stream.consume(trimmed_amt)
    }
    fn fill_buf(&mut self) -> IOResult<&[u8]> {
        // Early call. Borrow weirdness.
        let remaining = self.remaining();

        let buf = self.stream.fill_buf()?;

        // Limit buffer's size, prevent user from seeing past-the-end
        let buf = trim_buf(buf, remaining);

        Ok(buf)
    }
}
impl<W: Write> Write for MyTake<W> {
    fn write(&mut self, buf: &[u8]) -> IOResult<usize> {
        let buf = trim_buf(buf, self.remaining());
        // Short circuit if we can't write any more data
        if buf.is_empty() {
            return Ok(0);
        }
        let num_written = self.stream.write(buf)?;
        // Defensive checks for bad inner writer impl
        // (or my own bugs :P)
        let new_cursor = self
            .cursor
            .checked_add(num_written as u64)
            .filter(|new| *new <= self.len)
            .ok_or_else(|| IOError::other("inner writer overflowed MyTake cursor"))?;

        self.cursor = new_cursor;

        Ok(num_written)
    }
    fn flush(&mut self) -> IOResult<()> {
        self.stream.flush()
    }
    // todo: fast write_vectored impl
    // todo: eargerly fail write_all, write_all_vectored if too long
}
impl<R: Seek> Seek for MyTake<R> {
    // TODO: fast seek using BufReader::seek_relative when R = BufReader
    // (not sure how to make the type system let me do that!)
    // (why doesn't BufReader's Seek impl just do that automatically TwT)
    fn seek(&mut self, pos: std::io::SeekFrom) -> IOResult<u64> {
        let err_past_the_start = "seek offset past-the-start";
        let err_overflow_cursor = "seek offset overflows cursor";
        let new_cursor: u64 = match pos {
            SeekFrom::Current(delta) => {
                // Clamp upper bound to self length
                let delta = if delta > 0 {
                    // Saturate OK - we're taking the min with a i64 anyway
                    delta.min(self.remaining().saturating_as())
                } else {
                    delta
                };
                self.cursor
                    .checked_add_signed(delta)
                    // Also catches past-the-start
                    .ok_or_else(|| IOError::other(err_overflow_cursor))?
            }
            SeekFrom::Start(pos) => pos.min(self.len),
            SeekFrom::End(pos) => {
                // Clamp upper bound, flip to positive for subtraction
                let pos = pos.max(0).unsigned_abs();
                self.len
                    .checked_sub(pos)
                    .ok_or_else(|| IOError::other(err_past_the_start))?
            }
        };

        // Avoid a syscall. This optimization used a lot in the RIFF decoders.
        if new_cursor != self.cursor {
            // Each branch checks this individually. Still, make very sure.
            debug_assert!(new_cursor <= self.len);

            // We must seek the underlying reader with a Relative seek, as we
            // don't know what it's End and Start are relative to ours
            let delta: i64 = new_cursor
                .checked_as::<i64>()
                .zip(self.cursor.checked_as::<i64>())
                .and_then(|(new, old)| new.checked_sub(old))
                .ok_or_else(|| IOError::other("delta seek overflows"))?;

            self.stream.seek(SeekFrom::Current(delta))?;
            self.cursor = new_cursor;
        }

        Ok(self.cursor)
    }
    fn stream_position(&mut self) -> IOResult<u64> {
        Ok(self.cursor())
    }
}
/// Clamp a buffer to the given length.
/// u64 since this is just for internal use.
fn trim_buf<T>(buf: &[T], len: u64) -> &[T] {
    let len = buf.len().min(len.saturating_as());
    &buf[..len]
}
/// Clamp a buffer to the given length.
/// u64 since this is just for internal use.
fn trim_buf_mut<T>(buf: &mut [T], len: u64) -> &mut [T] {
    let len = buf.len().min(len.saturating_as());
    &mut buf[..len]
}
// TODO: clamp [IoSlice{Mut}]. Hard to do because it requires alloc'ing a new vec of slices, which is no good!
// Could use a stack-based tinyvec, but then what do I do if the number of slices exceeds capacity?

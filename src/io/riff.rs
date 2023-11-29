//! Zero-copy RIFF readers and writers.
//!
//! Contains many utilities for reading arbitrary RIFF-formatted data,
//! as well as several fuzzpaint-specific utility readers.

pub mod decode;
pub mod encode;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
pub struct ChunkID(pub [u8; 4]);
impl ChunkID {
    pub const RIFF: Self = ChunkID(*b"RIFF");
    // LISTs
    pub const LIST: Self = ChunkID(*b"LIST");
    pub const INFO: Self = ChunkID(*b"INFO");
    pub const OBJS: Self = ChunkID(*b"objs");
    // fuzzpaint custom chunks
    pub const FZP_: Self = ChunkID(*b"fzp ");
    pub const THMB: Self = ChunkID(*b"thmb");
    pub const DOCV: Self = ChunkID(*b"docv");
    // DICT items
    pub const DICT: Self = ChunkID(*b"DICT");
    pub const BRSH: Self = ChunkID(*b"brsh");
    pub const PTLS: Self = ChunkID(*b"ptls");
    pub const STRK: Self = ChunkID(*b"strk");
    // GRPH items
    pub const GRPH: Self = ChunkID(*b"GRPH");
    pub const NODE: Self = ChunkID(*b"node");
    pub const CONN: Self = ChunkID(*b"conn");
    pub const BLND: Self = ChunkID(*b"blnd");
    pub const HIST: Self = ChunkID(*b"hist");
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

#[cfg(test)]
mod test {
    use super::*;
    use super::{decode::*, encode::*};
    use std::io::{Cursor, Read};
    const EMPTY_FZP: &'static [u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/empty.fzp"));
    /// Test that a handwritten fzp document can be parsed, NOT that a document can be assembled from the data.
    #[test]
    fn read_empty_fzp_riff() {
        let file = Cursor::new(EMPTY_FZP);
        let root = BinaryChunkReader::new(file).unwrap();
        let outer_chunk_size = root.data_len_unsanitized();
        // Root node should be a RIFF element.
        assert_eq!(root.id(), ChunkID::RIFF);
        // Start with RIFF inner id (4 bytes)
        let mut total_inner_size = 4;
        // Read inner chunks
        let subchunks = root.into_subchunks().unwrap();

        // RIFF subtype is fuzzpaint file
        assert_eq!(subchunks.subtype_id(), ChunkID::FZP_);

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

        subchunks
            .try_for_each(|sub| {
                // Check that the list contains this id, remove it.
                let this_id = sub.id();
                let Some(idx) = names.iter().position(|chunk| *chunk == this_id) else {
                    panic!("unexpected chunk id {this_id}")
                };
                names.remove(idx);

                // Total up running length.
                total_inner_size += sub.self_len_unsanitized();

                Ok(())
            })
            .unwrap();

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
    /// Read a handwritten RIFF file containing a DICT full of test structures.
    #[test]
    fn read_dict() {
        let file = Cursor::new(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test-data/dict.fzp"
        )));

        let root = BinaryChunkReader::new(file).unwrap();
        let subchunks = root.into_subchunks().unwrap();
        assert_eq!(subchunks.id(), ChunkID::RIFF);
        assert_eq!(subchunks.subtype_id(), ChunkID::FZP_);

        // Ensure we visit every chunk
        let mut chunks_remaining = 1;

        subchunks
            .try_for_each(|chunk| {
                assert!(chunks_remaining > 0);
                chunks_remaining -= 1;
                assert_eq!(chunk.id(), ChunkID::DICT);
                let dict = chunk.into_dict().unwrap();
                assert_eq!(dict.subtype_id(), ChunkID(*b"test"));

                // expected metas bytes, in order.
                // Coincidentally also asserts that the metadata len is proper
                // and that there's the right count.
                let mut test_contents: Vec<_> = [
                    [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
                    *b"Hello!!!",
                    [0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00],
                    *b"It works",
                ]
                .into_iter()
                .collect();

                let spillover = dict
                    .try_for_each(|mut meta| {
                        assert_eq!(meta.remaining(), 8);
                        let mut data = [0; 8];
                        meta.read_exact(&mut data).unwrap();
                        assert_eq!(meta.remaining(), 0);

                        let meta = data;

                        // Check it matches expected.
                        assert_eq!(test_contents.remove(0), meta);

                        Ok(())
                    })
                    .unwrap();

                assert!(test_contents.is_empty());
                assert_eq!(spillover.data_len_unsanitized(), 0);

                Ok(())
            })
            .unwrap();
        assert_eq!(chunks_remaining, 0)
    }
}

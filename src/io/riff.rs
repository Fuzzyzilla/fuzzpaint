pub mod decode;
pub mod encode;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
pub struct ChunkID(pub [u8; 4]);
impl ChunkID {
    // RIFF standard chunks
    pub const RIFF: Self = ChunkID(*b"RIFF");
    pub const INFO: Self = ChunkID(*b"INFO");
    pub const LIST: Self = ChunkID(*b"LIST");
    // fuzzpaint custom chunks
    pub const DICT: Self = ChunkID(*b"DICT");
    pub const FZP_: Self = ChunkID(*b"fzp ");
    pub const OBJS: Self = ChunkID(*b"objs");
    pub const THMB: Self = ChunkID(*b"thmb");
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

#[cfg(test)]
mod test {
    use super::*;
    use super::{decode::*, encode::*};
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

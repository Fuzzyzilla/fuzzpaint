/// A Globally-unique identifier, stable + sharable over the network or written
/// to a file.
///
/// This is a 256-bit `blake3` hash of the data. - For a texture, this is a hash
/// of the *packed* image data + meta. For a brush, it is the hash of the full
/// settings that make it up.
///
/// *`*Ord` and `*Eq` are non-cryptographic*. This is desirable for our uses :3
// * Originally, this was a randomized UUID. However, it occured to me that a
// bad actor could then trivially make a brush conflict with an existing popular
// brush, leading to *permanent* strange behavior from any client that ever
// observes both the genuine and fake brushes.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct UniqueID(pub [u8; 32]);
impl std::fmt::Debug for UniqueID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts = [
            u128::from_be_bytes(self.0[..16].try_into().unwrap()),
            u128::from_be_bytes(self.0[16..].try_into().unwrap()),
        ];

        write!(f, "UniqueID({:032X}{:032X})", parts[0], parts[1])
    }
}
impl std::fmt::Display for UniqueID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use base64::Engine;

        // I chose an alg at random UwU Use 8 bits, as that perfectly consumes
        // the padding space that would be included in the Base64 repr.
        // (actually, only two bits are spare, but a 1/4 chance for a typo to
        // pass is *not great(tm)*)
        let checksum = crc::Crc::<u8>::new(&crc::CRC_8_DARC).checksum(self.0.as_slice());
        // -w-;;;;;;;
        #[rustfmt::skip]
        let input = [
            self.0[0],  self.0[1],  self.0[2],  self.0[3],  self.0[4],  self.0[5],  self.0[6],  self.0[7],  self.0[8],  self.0[9],
            self.0[10], self.0[11], self.0[12], self.0[13], self.0[14], self.0[15], self.0[16], self.0[17], self.0[18], self.0[19],
            self.0[20], self.0[21], self.0[22], self.0[23], self.0[24], self.0[25], self.0[26], self.0[27], self.0[28], self.0[29],
            self.0[30], self.0[31],
            checksum
        ];
        let mut output = [0; 44];

        // Base64 encoding is deterministic of course! 33 bytes in == exactly 44
        // bytes out
        assert_eq!(
            44,
            base64::engine::general_purpose::STANDARD_NO_PAD
                .encode_slice(input.as_slice(), &mut output)
                .unwrap()
        );

        f.write_str(std::str::from_utf8(output.as_slice()).unwrap())
    }
}
impl std::hash::Hash for UniqueID {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // No need for length prefix as we have a fixed-length repr. (AFAIK
        // that's right?)
        state.write(&self.0);
    }
    fn hash_slice<H: std::hash::Hasher>(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        // Unstable?? Bruh? state.write_length_prefix(data.len());
        state.write(bytemuck::cast_slice(data));
    }
}

/// A special hasher that just takes the first 64 bits from a [`UniqueID`].
/// Since [`UniqueID`]s are high quality hashes already, hashing them again with
/// a real-time algorithm will make the quality of the result strictly worse.
///
/// Not for general use, will panic or give you awful results if you try -
/// don't. :P
///
// How can I make this private? The name of the type must be exposed through the
// type alias [`UniqueIDMap`] so perhaps I cannot.
#[derive(Default)]
pub struct UniqueIDHasher {
    first: u64,
}
impl std::hash::Hasher for UniqueIDHasher {
    fn finish(&self) -> u64 {
        self.first
    }
    fn write(&mut self, bytes: &[u8]) {
        assert_eq!(bytes.len(), 32, "misuse of UniqueIDHasher");
        self.first = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    }
}
type BuildUniqueIDHasher = std::hash::BuildHasherDefault<UniqueIDHasher>;
pub type UniqueIDMap<T> = std::collections::HashMap<UniqueID, T, BuildUniqueIDHasher>;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum UniqueIDParseError {
    #[error("invalid checksum")]
    ChecksumMismatch,
    #[error("contains invalid character")]
    InvalidCharacter,
    #[error("incorrect length")]
    BadLength,
}
impl std::str::FromStr for UniqueID {
    type Err = UniqueIDParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use base64::Engine;
        if s.len() != 44 {
            return Err(UniqueIDParseError::BadLength);
        }
        let mut bytes = [0; 33];
        let res = base64::engine::general_purpose::STANDARD_NO_PAD.decode_slice(s, &mut bytes);
        match res {
            // Success!
            Ok(33) => (),
            Err(base64::DecodeSliceError::DecodeError(base64::DecodeError::InvalidByte(..))) => {
                return Err(UniqueIDParseError::InvalidCharacter);
            }
            // All other cases are length problems
            _ => return Err(UniqueIDParseError::BadLength),
        }

        // Split two parts of the data..
        let (id, checksum) = (<[_; 32]>::try_from(&bytes[..32]).unwrap(), bytes[32]);

        // DARC 8 crc of first 32 bytes should == 33rd byte
        let actual_checksum = crc::Crc::<u8>::new(&crc::CRC_8_DARC).checksum(&bytes[..32]);
        if checksum == actual_checksum {
            Ok(UniqueID(id))
        } else {
            Err(UniqueIDParseError::ChecksumMismatch)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{UniqueID, UniqueIDParseError};
    const CONSECUTIVE_ID: UniqueID = UniqueID([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ]);
    // manually calculated expected value :3 ID data with DARC CRC8 appended put
    // into a "standard alphabet" padless base64

    const BASE64_CONSECUTIVE_ID: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8L";
    #[test]
    fn fmt_debug() {
        assert_eq!(
            format!("{CONSECUTIVE_ID:?}"),
            "UniqueID(000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F)"
        );
    }
    #[test]
    fn fmt_base64() {
        assert_eq!(format!("{CONSECUTIVE_ID}"), BASE64_CONSECUTIVE_ID);
    }
    #[test]
    fn from_success() {
        assert_eq!(BASE64_CONSECUTIVE_ID.parse(), Ok(CONSECUTIVE_ID));
    }
    #[test]
    fn from_bad_len() {
        assert_eq!(
            "abc123".parse::<UniqueID>(),
            Err(UniqueIDParseError::BadLength)
        );
    }
    #[test]
    fn from_bad_checksum() {
        // same as BASE64_CONSECUTIVE_ID but with a two chars swapped..
        assert_eq!(
            //                     vv
            "AAECAwQFBgcICQoLDA0ODxRAEhMUFRYXGBkaGxwdHh8L".parse::<UniqueID>(),
            Err(UniqueIDParseError::ChecksumMismatch)
        );
    }
    #[test]
    fn from_bad_char() {
        // same as BASE64_CONSECUTIVE_ID but with an invalid alphabet
        assert_eq!(
            //                     v
            "AAECAwQFBgcICQoLDA0ODx^REhMUFRYXGBkaGxwdHh8L".parse::<UniqueID>(),
            Err(UniqueIDParseError::InvalidCharacter)
        );
    }
}

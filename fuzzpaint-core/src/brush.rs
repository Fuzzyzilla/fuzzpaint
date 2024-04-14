//! # Brush

/// A Globally-unique identifier, stable + sharable over the network or written to a file.
///
/// This is a 256-bit `blake3` hash of the data. - For a texture, this is a hash of the *packed* image data + meta.
/// For a brush, it is the hash of the full settings that make it up.
///
/// *`*Ord` and `*Eq` are non-cryptographic*. This is desirable for our uses :3
// * Originally, this was a randomized UUID. However, it occured to me that a bad actor could then trivially
// make a brush conflict with an existing popular brush, leading to *permanent* strange behavior from any client
// that ever observes both the genuine and fake brushes.
#[derive(Clone, Copy, Hash, bytemuck::Pod, bytemuck::Zeroable, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct UniqueID([u8; 32]);
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

        // I chose an alg at random UwU
        // Use 8 bits, as that perfectly consumes the padding space that would be included in the Base64 repr.
        // (actually, only two bits are spare, but a 1/4 chance for a typo to pass is *not great(tm)*)
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

        // Base64 encoding is deterministic of course! 33 bytes in == exactly 44 bytes out
        assert_eq!(
            44,
            base64::engine::general_purpose::STANDARD_NO_PAD
                .encode_slice(input.as_slice(), &mut output)
                .unwrap()
        );

        f.write_str(std::str::from_utf8(output.as_slice()).unwrap())
    }
}

#[cfg(test)]
mod test {
    use super::UniqueID;
    #[test]
    fn fmt_debug() {
        let id = UniqueID([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ]);

        assert_eq!(
            format!("{id:?}"),
            "UniqueID(000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F)"
        );
    }
    #[test]
    fn fmt_base64() {
        let id = UniqueID([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ]);

        // manually calculated expected value :3
        // ID data with DARC CRC8 appended put into a "standard alphabet" padless base64
        assert_eq!(
            format!("{id}"),
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8L"
        );
    }
}

bitflags::bitflags! {
    #[derive(Clone,Copy, PartialEq, Eq, Hash, Debug)]
    pub struct Mirroring : u8 {
        /// Flip on X-Axis
        const H_FLIP   = 0b0000_0001;
        /// Flip on Y-Axis
        const V_FLIP   = 0b0000_0010;
        /// Vertically adjust the start position of the current repetition to match the the end value of the last repetition.
        /// Eg, a line of slope 1 becomes an infinite ramp instead of a sawtooth curve
        const V_ALIGN  = 0b0000_0100;
        /// Don't allow exceeding min or max Y.
        const SATURATE = 0b0000_1000;
        /// Unused bits for the future :>
        const RESERVED = 0b1111_0000;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MirroringMode {
    /// Clamp to edge.
    Clamp,
    Mirror(Mirroring),
}

/// [0, 1) value.
#[derive(
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Copy,
    Hash,
    Debug,
    Default,
    bytemuck::Zeroable,
    bytemuck::Pod,
)]
#[repr(transparent)]
pub struct NormalizedU32(pub u32);
impl NormalizedU32 {
    pub const ZERO: Self = Self(0);
    pub const MAX: Self = Self(u32::MAX);
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
    pub fn from_float(value: f32) -> Option<Self> {
        if !value.is_finite() || value < 0.0 || value >= 1.0 {
            None
        } else {
            Some(Self((value * u32::MAX as f32) as u32))
        }
    }
}
impl From<NormalizedU32> for f32 {
    fn from(value: NormalizedU32) -> Self {
        // I wrote a bitwise crime for this, would be fun to see if it's any more efficient >:3c
        value.0 as f32 / (u32::MAX as f32 + 1.0)
    }
}

fn lerp(a: NormalizedU32, b: NormalizedU32, t: NormalizedU32) -> NormalizedU32 {
    // Probably a more precise way to do this. Todo!
    let t = f32::from(t);
    let val = f32::from(a) * (1.0 - t) + f32::from(b) * t;
    // bad!! badd!!!!
    NormalizedU32::from_float(val).unwrap()
}

/// Like [`lerp`], but takes a parameter specifying the endpoint of `t`.
fn lerp_max(
    a: NormalizedU32,
    b: NormalizedU32,
    t: NormalizedU32,
    max_t: NormalizedU32,
) -> NormalizedU32 {
    // Probably a more precise way to do this. Todo!
    let t = f32::from(t) / f32::from(max_t);
    let val = f32::from(a) * (1.0 - t) + f32::from(b) * t;
    NormalizedU32::from_float(val).unwrap()
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct CurvePoint {
    /// X position, from min to max.
    frac_x: NormalizedU32,
    /// The value at that point.
    value: NormalizedU32,
}

pub struct Curve<'a> {
    min_y: f32,
    scale_y: f32,
    /// Divide all inputs by this scale.
    scale_x: f32,
    /// Value at 0
    start: NormalizedU32,
    /// Value at 1
    end: NormalizedU32,
    /// Values between 0 and 1.
    /// Must be sorted and de-duplicated by frac_x
    points: &'a [CurvePoint],
    mirroring: MirroringMode,
}
impl Curve<'_> {
    /// Sample from a floating point X value, handling mirroring modes.
    pub fn sample(&self, mut pos: f32) -> Option<f32> {
        pos /= self.scale_x;
        let mirror_idx = pos.floor();
        // zero to one within the current mirror
        let fract = NormalizedU32::from_float(pos - mirror_idx)?;
        // which mirror we're on
        let mirror_idx = mirror_idx as i32;

        let norm_y = match self.mirroring {
            // Clamp mode. If outside normal range (mirror_idx != 0) clamp to relavent edge.
            MirroringMode::Clamp => match mirror_idx {
                ..=-1 => f32::from(self.start),
                // Inside range, sample as normal :3
                0 => self.sample_normalized(fract),
                1.. => f32::from(self.end),
            },
            MirroringMode::Mirror(mirroring) => {
                let mirror_x = mirroring.intersects(Mirroring::H_FLIP) && mirror_idx & 1 == 1;
                let mirror_y = mirroring.intersects(Mirroring::V_FLIP) && mirror_idx & 1 == 1;

                // Sample with maybe mirrored x
                let mut sample = self.sample_normalized_raw(if mirror_x {
                    NormalizedU32::MAX.saturating_sub(fract)
                } else {
                    fract
                });

                if mirror_y {
                    sample = NormalizedU32::MAX.saturating_sub(sample);
                }

                let mut sample = f32::from(sample);

                // Apply vertical shift (mirror_x cancels out the effect!)
                if mirroring.intersects(Mirroring::V_ALIGN) && !mirror_x {
                    // How much y increases per mirror (in normalized space!)
                    let v_shift = f32::from(self.end) - f32::from(self.start);

                    sample = v_shift * mirror_idx as f32;
                    // todo: take into account mirror Y :V
                }

                if mirroring.intersects(Mirroring::SATURATE) {
                    sample = sample.clamp(0.0, 1.0);
                }

                sample
            }
        };

        Some(norm_y * self.scale_y + self.min_y)
    }
    pub fn sample_normalized_raw(&self, pos: NormalizedU32) -> NormalizedU32 {
        match self.points.binary_search_by(|p| p.frac_x.cmp(&pos)) {
            // Highly unlikely - hit it spot on!
            Ok(idx) => self.points[idx].value,
            // More likely, between two.
            Err(idx) => {
                match idx {
                    // Special case - no inner points, between start and end anchor
                    0 if self.points.is_empty() => lerp(self.start, self.end, pos),
                    // Between start anchor and first point
                    0 => {
                        let first = self.points.first().unwrap();
                        lerp_max(self.start, first.value, pos, first.frac_x)
                    }
                    // Between two inner points
                    _ if idx < self.points.len() - 1 => {
                        let before = self.points[idx];
                        let after = self.points[idx + 1];
                        // These can be unchecked sub but gwos
                        let pos = pos.saturating_sub(before.frac_x);
                        let dist = after.frac_x.saturating_sub(before.frac_x);

                        lerp_max(before.value, after.value, pos, dist)
                    }
                    // Between last point and end anchor
                    _ => {
                        let last = self.points.last().unwrap();

                        // Can be unchecked sub but gwos
                        let pos = pos.saturating_sub(last.frac_x);
                        let dist = NormalizedU32::MAX.saturating_sub(last.frac_x);
                        lerp_max(last.value, self.end, pos, dist)
                    }
                }
            }
        }
    }
    pub fn sample_normalized(&self, pos: NormalizedU32) -> f32 {
        f32::from(self.sample_normalized_raw(pos)) * self.scale_y + self.min_y
    }
}
#[derive(Copy, Clone, Debug, Hash)]
pub enum Swizzle {
    /// One channel, coverage of white.
    /// Swizzle = RRRR
    Alpha,
    /// Two channels, greyscale + alpha
    /// Swizzle = RRRG
    GreyAlpha,
    /// Four channels, full color.
    /// Swizzle = RGBA
    ColorAlpha,
}
#[derive(Copy, Clone, Debug, Hash)]
pub enum Format {
    // /// Floating point color format, linear space.
    // // This would be very nice to have. However, I cannot find an existing image codec that falls within
    // // the rest of our needs that also supports this pixel format!!
    // F16,
    /// Normalized bytes. RGB/Grey components are sRGB, alpha component is Linear.
    // We use sRGB for better perceptual linearity in representable colors. This could be
    // fixed by using Linear u16, but then data size is doubled!
    SRGBA8,
}
/// Properties inherent to a texture file (ie, settings that one wouldn't want to change if they're re-using the texture)
#[derive(Copy, Clone, Debug)]
pub struct Texture {
    pub swizzle: Swizzle,
    pub format: Format,
    pub size: [std::num::NonZeroU32; 2],
    /// Where the 0,0 point is on the texture, in pixels from the top-left of the texture
    pub origin: [NormalizedU32; 2],
    /// The radius containing the bulk of the ink in the texture, in texture pixels around the `origin`.
    pub diagonal_radius: NormalizedU32,
    // Todo: Set of categories, like Geometric, Ink, Texture, Splatter, ect to aid in searching through a library.
    // categories: (),
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Debug)]
    pub struct Filter  : u8 {
        /// Linear with linear mips. If not set, Nearest neighbor.
        const DOWNSCALE_TRILINEAR = 0b0000_0001;
        /// Linear. If not set, Nearest neighbor.
        const UPSCALE_BILINEAR    = 0b0000_0010;
    }
}

/// Properties of how a texture is used in a brush.
pub struct Tip {
    pub texture: UniqueID,
    /// Angle offset, radians.
    pub base_rotation: NormalizedU32,
    pub base_scale: f32,
    pub filter: Filter,
}

pub fn default_brush() -> UniqueID {
    UniqueID([0; 32])
}

pub struct Brush {
    name: String,
    // Todo: multitip
    tip: Tip,
    // Todo: Curves per tip
    // curves: (),
    // Todo: Set of categories, like Geometric, Ink, Texture, Splatter, ect to aid in searching through a library.
    // categories: (),
}

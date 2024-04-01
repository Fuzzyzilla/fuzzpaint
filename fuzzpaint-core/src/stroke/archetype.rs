bitflags::bitflags! {
    #[derive(Copy, Clone, Eq, PartialEq, Hash, bytemuck::Pod, bytemuck::Zeroable, Debug)]
    /// Description of a point's data fields. Organized such that devices that have later flags are
    /// also likely to have prior flags.
    ///
    /// Note that position nor arc_length are required fields. Arc_length is derived data,
    /// and position may be ignored for strokes which trace a predefined path.
    #[rustfmt::skip]
    #[repr(transparent)]
    pub struct Archetype : u8 {
        /// The point stream reports an (X: f32, Y: f32) position.
        const POSITION =   0b0000_0001;
        /// The point stream reports an f32 timestamp, in seconds from an arbitrary start moment.
        const TIME =       0b0000_0010;
        /// The point stream reports an f32, representing the cumulative length of the path from the start.
        const ARC_LENGTH = 0b0000_0100;
        /// The point stream reports a normalized, non-saturated pressure value.
        const PRESSURE =   0b0000_1000;
        /// The point stream reports a signed noramlized (X: f32, Y: f32) tilt, where positive X is to the right,
        /// positive Y is towards the user.
        const TILT =       0b0001_0000;
        /// Mask for bits that contain two fields
        const HAS_TWO_FIELDS = 0b0001_0001;
        /// The point stream reports a normalized f32 distance, in arbitrary units.
        const DISTANCE =   0b0010_0000;
        /// The point stream reports stylus roll (rotation along it's axis). Units and sign unknown!
        ///
        /// FIXME: Someone with such hardware, please let me know what it's units are :3
        const ROLL =       0b0100_0000;
        /// The point stream reports wheel values in signed, unnormalized, non-wrapping degrees, f32.
        ///
        /// Wheels are a general-purpose value which the user can use in their brushes. It may
        /// correspond to a physical wheel on the pen or pad, a touch slider, ect. which may be interacted
        /// with during the stroke for expressive effects.
        const WHEEL =      0b1000_0000;
    }
}
impl Archetype {
    /// How many elements (f32) does a point of this archetype occupy?
    #[must_use]
    pub const fn elements(self) -> usize {
        // Formerly Self::iter based but the codegen was un-scrumptious

        // Every field specifies one element, count them all
        self.bits().count_ones() as usize
        // These fields specify two elements, count them again
            + (self.bits() & Self::HAS_TWO_FIELDS.bits()).count_ones() as usize
    }
    #[must_use]
    pub const fn len_bytes(self) -> usize {
        self.elements() * std::mem::size_of::<f32>()
    }
    /// Fetch the offset of the given property. None if the property isn't included.
    /// *`other` should have just one bit set.* Otherwise, the returned value is meaningless.
    #[must_use]
    pub const fn offset_of(self, other: Self) -> Option<usize> {
        if self.intersects(other) {
            // Bitmask of every element ordered *before* the requested element
            //   1000
            //    - 1
            // = 0111
            let before_mask = Self::from_bits_retain(other.bits() - 1);
            // Count how many elements *before* the given element are set.
            Some(self.intersection(before_mask).elements())
        } else {
            None
        }
    }
}

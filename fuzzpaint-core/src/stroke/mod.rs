pub mod archetype;
pub use archetype::Archetype;

//U32::MAX us == 71 minutes. If someone draws one continuous stroke for that long, other problems would certainly arise. D:
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(transparent)]
pub struct Microseconds(pub u32);

/// A single dynamically structured point.
#[derive(Clone, Copy)]
pub struct BorrowedPoint<'a> {
    /// Invariant: `data.len == archetype.elements`
    elements: &'a [u32],
    archetype: Archetype,
}
impl<'a> BorrowedPoint<'a> {
    pub fn empty() -> Self {
        BorrowedPoint {
            elements: &[],
            archetype: Archetype::empty(),
        }
    }
    /// Borrow a slice as a point. Returns `None` if the length of the data is not `archetype.elements()`
    ///
    /// The elements in data will be interpreted as a series of packed elements. For each flag of `archetype` in order, sequential `u32`
    /// values will be interpreted as the relevant type as documented in [`Archetype`].
    /// This is a safe function as all such `transmutes` from any `u32` to any of the relevant types is sound.
    #[must_use]
    pub fn new(elements: &'a [u32], archetype: Archetype) -> Option<Self> {
        // Check that lengths make sense
        if elements.len() == archetype.elements() {
            Some(Self {
                elements,
                archetype,
            })
        } else {
            None
        }
    }
    pub fn position(&self) -> Option<[f32; 2]> {
        self.archetype.offset_of(Archetype::POSITION).map(|idx| {
            let data: [u32; 2] = self.elements[idx..idx + 2].try_into().unwrap();
            bytemuck::cast(data)
        })
    }
    pub fn time(&self) -> Option<Microseconds> {
        self.archetype.offset_of(Archetype::TIME).map(|idx| {
            let data = self.elements[idx];
            bytemuck::cast(data)
        })
    }
    pub fn arc_length(&self) -> Option<f32> {
        self.archetype.offset_of(Archetype::ARC_LENGTH).map(|idx| {
            let data = self.elements[idx];
            bytemuck::cast(data)
        })
    }
    pub fn pressure(&self) -> Option<f32> {
        self.archetype.offset_of(Archetype::PRESSURE).map(|idx| {
            let data = self.elements[idx];
            bytemuck::cast(data)
        })
    }
    pub fn tilt(&self) -> Option<[f32; 2]> {
        self.archetype.offset_of(Archetype::TILT).map(|idx| {
            let data: [u32; 2] = self.elements[idx..idx + 2].try_into().unwrap();
            bytemuck::cast(data)
        })
    }
    pub fn distance(&self) -> Option<f32> {
        self.archetype.offset_of(Archetype::DISTANCE).map(|idx| {
            let data = self.elements[idx];
            bytemuck::cast(data)
        })
    }
    pub fn roll(&self) -> Option<f32> {
        self.archetype.offset_of(Archetype::ROLL).map(|idx| {
            let data = self.elements[idx];
            bytemuck::cast(data)
        })
    }
    pub fn wheel(&self) -> Option<f32> {
        self.archetype.offset_of(Archetype::WHEEL).map(|idx| {
            let data = self.elements[idx];
            bytemuck::cast(data)
        })
    }
}
impl std::fmt::Debug for BorrowedPoint<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("BorrowedPoint");
        if let Some(position) = self.position() {
            d.field("position", &position);
        }
        if let Some(time) = self.time() {
            d.field("time", &time);
        }
        if let Some(arc_length) = self.arc_length() {
            d.field("arc_length", &arc_length);
        }
        if let Some(pressure) = self.pressure() {
            d.field("pressure", &pressure);
        }
        if let Some(tilt) = self.tilt() {
            d.field("tilt", &tilt);
        }
        if let Some(distance) = self.distance() {
            d.field("distance", &distance);
        }
        if let Some(roll) = self.roll() {
            d.field("roll", &roll);
        }
        if let Some(wheel) = self.wheel() {
            d.field("wheel", &wheel);
        }
        d.finish()
    }
}

/// A dynamic layout structure containing many packed points based on an archetype.
#[derive(Clone, Copy)]
pub struct StrokeSlice<'a> {
    elements: &'a [u32],
    archetype: Archetype,
    /// Number of points in this stroke.
    /// Invariant: `self.len * self.archetype.elements() == self.elements.len()`
    len: usize,
}
impl<'a> StrokeSlice<'a> {
    /// Create an empty borrow of the given archetype.
    #[must_use]
    pub fn empty(archetype: Archetype) -> Self {
        Self {
            archetype,
            elements: &[],
            len: 0,
        }
    }
    /// Borrow a slice as a stroke. Returns `None` if the length of the data is not divisible by `archetype.elements()`
    ///
    /// The elements in data will be interpreted as a series of packed points. For each flag of `archetype` in order, sequential `u32`
    /// values will be interpreted as the relevant type as documented in [`Archetype`].
    /// This is a safe function as all such `transmutes` from any `u32` to any of the relevant types is sound.
    #[must_use]
    pub fn new(elements: &'a [u32], archetype: Archetype) -> Option<Self> {
        // Special cases: Avoid div by zero
        if elements.is_empty() {
            return Some(Self::empty(archetype));
        }
        if archetype.is_empty() {
            // Arch is empty but data is not, that makes no sense!
            return None;
        }

        // Ensure data len makes sense given the archetype:
        if elements.len() % archetype.elements() != 0 {
            return None;
        }

        let len = elements.len() / archetype.elements();
        Some(Self {
            elements,
            archetype,
            len,
        })
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get the number of *points* in this stroke.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub fn archetype(&self) -> Archetype {
        self.archetype
    }
    /// Get the full data as bytes.
    #[must_use]
    pub fn bytes(&self) -> &'a [u8] {
        bytemuck::cast_slice(self.elements)
    }
    /// Get the full data interpreted as `u32`
    #[must_use]
    pub fn elements(&self) -> &'a [u32] {
        self.elements
    }
    /// Take a sub-stroke of this stroke. Indices are in units of *points*.
    #[must_use = "returns a new instance without modifying `self`"]
    pub fn slice<R: std::ops::RangeBounds<usize>>(self, range: R) -> Option<Self> {
        let start = match range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Excluded(&x) => x.checked_add(1)?,
            std::ops::Bound::Included(&x) => x,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Unbounded => self.len,
            std::ops::Bound::Excluded(&x) => x,
            std::ops::Bound::Included(&x) => x.checked_add(1)?,
        };

        // Mimic behavior of `[T]::get`
        // empty range is OK, starting one-past-end is OK.
        if start > end {
            return None;
        }
        if start > self.len {
            return None;
        }

        let sliced_len = end - start;
        let elems = self.archetype.elements();

        Some(Self {
            // Shoullldd always be Some
            elements: self.elements.get(start * elems..end * elems)?,
            archetype: self.archetype,
            len: sliced_len,
        })
    }
    /// Fetch the first point of this stroke slice. `None` if empty.
    pub fn first(&self) -> Option<BorrowedPoint<'a>> {
        self.get(0)
    }
    /// Fetch the last point of this stroke slice. `None` if empty.
    pub fn last(&self) -> Option<BorrowedPoint<'a>> {
        self.get(self.len().checked_sub(1)?)
    }
    /// Fetch a point from the stroke slice. `None` if out-of-bounds.
    pub fn get(&self, idx: usize) -> Option<BorrowedPoint<'a>> {
        if idx >= self.len() {
            return None;
        }
        let elements = self.slice(idx..=idx).unwrap().elements();
        Some(BorrowedPoint::new(elements, self.archetype()).unwrap())
    }
}
impl std::fmt::Debug for StrokeSlice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("StrokeSlice");
        d.field(
            "points",
            // Grr.. `field_with` would fix this, but unstable. eh.
            &(0..self.len())
                .map(|i| self.get(i).unwrap())
                .collect::<Vec<_>>(),
        );
        d.finish()
    }
}

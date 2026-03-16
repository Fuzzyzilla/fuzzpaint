//! # Structure-of-Arrays representation.
//!
//! Most efficient for compression.
use std::vec::Vec as StdVec;

use super::{ArcLength, Archetype, Distance, Position, Pressure, Roll, Tilt, Time, Wheel};
use core::ptr::NonNull;

#[derive(Default, Clone)]
pub struct Vec {
    archetype: Archetype,

    // These can share a length and capacity. But that's evil IDC
    position: StdVec<Position>,
    time: StdVec<Time>,
    arc_length: StdVec<ArcLength>,
    pressure: StdVec<Pressure>,
    tilt: StdVec<Tilt>,
    distance: StdVec<Distance>,
    roll: StdVec<Roll>,
    wheel: StdVec<Wheel>,
}
impl Vec {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn len(&self) -> usize {
        /// All vecs that are populated have the same length. Choose one.
        [
            (Archetype::POSITION, self.position.len()),
            (Archetype::TIME, self.time.len()),
            (Archetype::ARC_LENGTH, self.arc_length.len()),
            (Archetype::PRESSURE, self.pressure.len()),
            (Archetype::TILT, self.tilt.len()),
            (Archetype::DISTANCE, self.distance.len()),
            (Archetype::ROLL, self.roll.len()),
            (Archetype::WHEEL, self.wheel.len()),
        ]
        .into_iter()
        .find_map(|(bit, len)| self.archetype.intersects(bit).then_some(len))
        // No bits set, zero length.
        .unwrap_or(0)
    }
    pub fn clear(&mut self) {
        self.archetype = Archetype::empty();

        self.position.clear();
        self.time.clear();
        self.arc_length.clear();
        self.pressure.clear();
        self.tilt.clear();
        self.distance.clear();
        self.roll.clear();
        self.wheel.clear();
    }
    /// Set the kinds of data this vector carries. Unsetting then re-setting a
    /// bit results in the data being lost and reset to defaults.
    pub fn set_archetype(&mut self, new_archetype: Archetype) {
        let default = super::Point::default();
        // Delete no longer needed axes:
        {
            let removed = self.archetype.difference(new_archetype);
        }
    }
    /// Push a point. If the point contains axes not contained in the vec, the
    /// previous points are populated with defaulted values.
    pub fn push_back(&mut self, point: super::Point) {
        let default = super::Point::default();
        todo!()
    }
}
pub struct Slice<'a> {
    /// Which of the below fields are populated?
    archetype: Archetype,
    /// True if all pointers refer to the same allocation. If so, this structure
    /// may be safely converted back into an `&[u32]` raw SoA representation. It
    /// is not possible, from pointers alone, to know if they're contiguous.
    all_same_allocation: bool,
    num_points: usize,

    position: NonNull<Position>,
    time: NonNull<Time>,
    arc_length: NonNull<ArcLength>,
    pressure: NonNull<Pressure>,
    tilt: NonNull<Tilt>,
    distance: NonNull<Distance>,
    roll: NonNull<Roll>,
    wheel: NonNull<Wheel>,

    borrow: core::marker::PhantomData<&'a [u32]>,
}
impl Slice<'_> {
    pub fn empty() -> Self {
        Self {
            archetype: Archetype::empty(),
            all_same_allocation: true,
            num_points: 0,

            position: NonNull::dangling(),
            time: NonNull::dangling(),
            arc_length: NonNull::dangling(),
            pressure: NonNull::dangling(),
            tilt: NonNull::dangling(),
            distance: NonNull::dangling(),
            roll: NonNull::dangling(),
            wheel: NonNull::dangling(),

            borrow: core::marker::PhantomData,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.num_points == 0
    }
    pub fn len(&self) -> usize {
        self.num_points
    }
}

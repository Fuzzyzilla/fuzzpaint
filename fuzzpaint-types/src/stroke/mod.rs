//! Arrays of stylus data

pub mod aos;
pub mod archetype;
pub mod soa;
pub use archetype::Archetype;

//U32::MAX us == 71 minutes. If someone draws one continuous stroke for that long, other problems would certainly arise. D:
#[derive(
    Default, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug,
)]
#[repr(transparent)]
pub struct Microseconds(pub u32);

type Position = [f32; 2];
type Time = Microseconds;
type ArcLength = f32;
type Pressure = f32;
type Tilt = [f32; 2];
type Distance = f32;
type Roll = f32;
type Wheel = f32;

pub trait Stroke {
    /// Get the axes contained in this stroke.
    fn archetype(&self) -> Archetype;
    /// Count the number of points in the stroke.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
pub enum EncodeError {
    /// The array was too small. In units of u32 words.
    TooSmall { needed: usize, got: usize },
}

/// A single point.
///
/// Arrays of this type should *not* collected, as it is unnecessarily large for
/// that use. Prefer using one of [`soa::Stroke`] or [`aos::Stroke`] instead.
#[derive(Default, Clone, Copy)]
pub struct Point {
    archetype: Archetype,
    // Arbitrary (but initialized) value when not included in the archetype:
    position: Position,
    time: Time,
    arc_length: ArcLength,
    pressure: Pressure,
    tilt: Tilt,
    distance: Distance,
    roll: Roll,
    wheel: Wheel,
}
impl Point {
    pub fn empty() -> Self {
        Self::default()
    }
    /// Copy all set fields from the other, retaining any fields set in `self`
    /// but missing from `other`.
    ///
    /// This is a useful operation for collecting values from an input device
    /// which may change its capabilities over time:
    /// ```
    /// # use crate::stroke::Point;
    /// # let current_state = Point::empty();
    /// # let new_state = Point::empty();
    /// current_state = current_state.or(new_state)
    /// ```
    pub fn or(mut self, other: &Self) -> Self {
        if let Some(position) = other.position() {
            self.set_position(position);
        }
        if let Some(time) = other.time() {
            self.set_time(time);
        }
        if let Some(arc_length) = other.arc_length() {
            self.set_arc_length(arc_length);
        }
        if let Some(pressure) = other.pressure() {
            self.set_pressure(pressure);
        }
        if let Some(tilt) = other.tilt() {
            self.set_tilt(tilt);
        }
        if let Some(distance) = other.distance() {
            self.set_distance(distance);
        }
        if let Some(roll) = other.roll() {
            self.set_roll(roll);
        }
        if let Some(wheel) = other.wheel() {
            self.set_wheel(wheel);
        }
        self
    }
    pub fn archetype(&self) -> Archetype {
        self.archetype
    }
    /// Set the kinds of data this point carries. Removing and then adding an
    /// archetype will result in the value being re-defaulted.
    pub fn set_archetype(&mut self, archetype: Archetype) {
        let mut new = Self::default();
        // Has all of the axes, defaulted.
        new.archetype = archetype;

        // Remove the ones from self that are now unset.
        self.archetype &= archetype;
        // Re-apply the axes that didn't change.
        new.or(self);

        *self = new;
    }
    pub fn position(&self) -> Option<Position> {
        self.archetype
            .intersects(Archetype::POSITION)
            .then_some(self.position)
    }
    pub fn set_position(&mut self, position: Position) {
        self.position = position;
        self.archetype |= Archetype::POSITION;
    }

    pub fn time(&self) -> Option<Time> {
        self.archetype
            .intersects(Archetype::TIME)
            .then_some(self.time)
    }
    pub fn set_time(&mut self, time: Time) {
        self.time = time;
        self.archetype |= Archetype::TIME;
    }

    pub fn arc_length(&self) -> Option<ArcLength> {
        self.archetype
            .intersects(Archetype::ARC_LENGTH)
            .then_some(self.arc_length)
    }
    pub fn set_arc_length(&mut self, arc_length: ArcLength) {
        self.arc_length = arc_length;
        self.archetype |= Archetype::ARC_LENGTH;
    }

    pub fn pressure(&self) -> Option<Pressure> {
        self.archetype
            .intersects(Archetype::PRESSURE)
            .then_some(self.pressure)
    }
    pub fn set_pressure(&mut self, pressure: Pressure) {
        self.pressure = pressure;
        self.archetype |= Archetype::PRESSURE;
    }

    pub fn tilt(&self) -> Option<Tilt> {
        self.archetype
            .intersects(Archetype::TILT)
            .then_some(self.tilt)
    }
    pub fn set_tilt(&mut self, tilt: Tilt) {
        self.tilt = tilt;
        self.archetype |= Archetype::TILT;
    }

    pub fn distance(&self) -> Option<Distance> {
        self.archetype
            .intersects(Archetype::DISTANCE)
            .then_some(self.distance)
    }
    pub fn set_distance(&mut self, distance: Distance) {
        self.distance = distance;
        self.archetype |= Archetype::DISTANCE;
    }

    pub fn roll(&self) -> Option<Roll> {
        self.archetype
            .intersects(Archetype::ROLL)
            .then_some(self.roll)
    }
    pub fn set_roll(&mut self, roll: Roll) {
        self.roll = roll;
        self.archetype |= Archetype::ROLL;
    }

    pub fn wheel(&self) -> Option<Wheel> {
        self.archetype
            .intersects(Archetype::WHEEL)
            .then_some(self.wheel)
    }
    pub fn set_wheel(&mut self, wheel: Wheel) {
        self.wheel = wheel;
        self.archetype |= Archetype::WHEEL;
    }
}

impl std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Point");
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

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
/// A range of Arclengths, used to refer to a subsection of a stroke with more
/// precision than simply slicing at point boundaries.
///
/// Behavior at boundaries (whether min and max are inclusive or exclusive) is
/// not yet defined.
#[repr(C)]
pub struct ArcLengthRange {
    min: f32,
    max: f32,
}
impl ArcLengthRange {
    /// The (-inf, inf) range, naturally covering the entire stroke.
    pub const EVERYTHING: Self = Self {
        min: f32::NEG_INFINITY,
        max: f32::INFINITY,
    };
    /// Construct from a range of arclengths
    pub fn new(arc_length_range: impl std::ops::RangeBounds<f32>) -> Self {
        Self {
            min: match arc_length_range.start_bound() {
                std::ops::Bound::Excluded(x) | std::ops::Bound::Included(x) => *x,
                std::ops::Bound::Unbounded => 0.0,
            },
            max: match arc_length_range.start_bound() {
                std::ops::Bound::Excluded(x) | std::ops::Bound::Included(x) => *x,
                std::ops::Bound::Unbounded => f32::INFINITY,
            },
        }
    }
    /// Split the range at the given position in the ArcLegnth, returning the
    /// lesser and greater of the ranges. If the split point is greater than the
    /// max, `(Some(self), None)` is returned, and vice-versa.
    pub fn split_at(self, at: f32) -> (Option<Self>, Option<Self>) {
        if at < self.min {
            (None, Some(self))
        } else if at > self.max {
            (Some(self), None)
        } else {
            (Some(Self::new(self.min..at)), Some(Self::new(at..self.max)))
        }
    }
    pub fn length(self) -> f32 {
        // Works in the (-inf, inf) special case too~!
        self.max - self.min
    }
    /// Cut the given range out of self, returning Zero, one, or two remaining
    /// ranges. The first range, if any will be the lesser, and the second will
    /// be the greater.
    pub fn split_range(self, range: Self) -> (Option<Self>, Option<Self>) {
        if range.min < self.min {
            if range.max > self.max {
                (None, None)
            } else {
                todo!();
                (None, Some(Self::new(range.max..self.max)))
            }
        } else {
            todo!()
        }
    }
}
impl std::ops::RangeBounds<f32> for ArcLengthRange {
    fn start_bound(&self) -> std::ops::Bound<&f32> {
        if self.min == f32::NEG_INFINITY {
            std::ops::Bound::Unbounded
        } else {
            std::ops::Bound::Included(&self.min)
        }
    }
    fn end_bound(&self) -> std::ops::Bound<&f32> {
        if self.max == f32::INFINITY {
            std::ops::Bound::Unbounded
        } else {
            std::ops::Bound::Excluded(&self.max)
        }
    }
}

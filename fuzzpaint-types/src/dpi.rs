/// A combined DPI and scale-factor.
#[derive(Clone, Copy, Debug)]
pub struct Dpi {
    /// Physical pixels per physical length.
    pub dots_per: (f32, PhysicalUnit),
    /// Physical pixels per logical pixel, aka the scale factor.
    pub physical_per_logical_px: f32,
}

/// A rectangle with an erased unit.
#[derive(Clone, Copy, Debug)]
pub struct UnitlessRect {
    pub origin: ultraviolet::Vec2,
    pub size: ultraviolet::Vec2,
}

/// A rectangle with a dynamic unit.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub origin: Length<ultraviolet::Vec2>,
    pub size: Length<ultraviolet::Vec2>,
}

/// A physical unit of length.
#[derive(Clone, Copy, Debug)]
pub enum PhysicalUnit {
    Centimeter,
    Inch,
    Point,
}
impl PhysicalUnit {
    /// Get the scale factor from this unit to another.
    const fn scale_factor_to(self, other: Self) -> f32 {
        match (self, other) {
            (Self::Centimeter, Self::Centimeter) => const { 1.0 },
            (Self::Inch, Self::Inch) => const { 1.0 },
            (Self::Point, Self::Point) => const { 1.0 },

            (Self::Inch, Self::Centimeter) => const { 2.54 },
            (Self::Centimeter, Self::Inch) => const { 1.0 / 2.54 },

            (Self::Inch, Self::Point) => const { 72.0 },
            (Self::Point, Self::Inch) => const { 1.0 / 72.0 },

            (Self::Centimeter, Self::Point) => const { 28.346_457 },
            (Self::Point, Self::Centimeter) => const { 0.035_277_776 },
        }
    }
}
/// A type with a unit.
#[derive(Clone, Copy, Debug)]
pub enum Length<T> {
    /// Physical measurements, dependent on the DPI (physical pixels per
    /// physical distance)
    Physical(Physical<T>),
    /// Logical pixels, dependent on the scale factor (physical pixels per
    /// logical pixel). All strokes are in logical pixels, however other vector
    /// elements may use other units.
    LogicalPx(LogicalPx<T>),
    /// Physical pixels, constant over changes to DPI or scale factor.
    PhysicalPx(PhysicalPx<T>),
}
impl<T> Length<T> {
    /// Apply the given transformation function, regardless of the unit.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Length<U> {
        match self {
            Self::Physical(t) => Length::Physical(t.map(f)),
            Self::LogicalPx(t) => Length::LogicalPx(t.map(f)),
            Self::PhysicalPx(t) => Length::PhysicalPx(t.map(f)),
        }
    }
}
impl<U, T> Length<T>
where
    T: std::ops::Mul<f32, Output = U>,
{
    /// Convert into a physical measure of the given unit using the given DPI settings.
    pub fn into_physical(self, dpi: &Dpi, new_unit: PhysicalUnit) -> Physical<U> {
        match self {
            Self::Physical(t) => t.convert(new_unit),
            Self::LogicalPx(LogicalPx(t)) => Physical {
                unit: new_unit,
                value: t
                    * (dpi.physical_per_logical_px * dpi.dots_per.1.scale_factor_to(new_unit)
                        / dpi.dots_per.0),
            },
            Self::PhysicalPx(PhysicalPx(t)) => Physical {
                unit: new_unit,
                value: t * (dpi.dots_per.1.scale_factor_to(new_unit) / dpi.dots_per.0),
            },
        }
    }
}
impl<U, T> Length<T>
where
    T: std::ops::Mul<f32, Output = U> + Into<U>,
{
    /// Convert into a measure in logical pixels using the given DPI settings.
    pub fn into_logical_px(self, dpi: &Dpi) -> LogicalPx<U> {
        match self {
            Self::LogicalPx(t) => t.map(Into::into),
            Self::Physical(t) => LogicalPx(
                t.value * (t.unit.scale_factor_to(dpi.dots_per.1) / dpi.physical_per_logical_px),
            ),
            Self::PhysicalPx(PhysicalPx(t)) => LogicalPx(t * (1.0 / dpi.physical_per_logical_px)),
        }
    }
    /// Convert into a measure in physical pixels using the given DPI settings.
    pub fn into_physical_px(self, dpi: &Dpi) -> PhysicalPx<U> {
        todo!()
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Physical<T> {
    unit: PhysicalUnit,
    value: T,
}
impl<T, U> Physical<T>
where
    T: std::ops::Mul<f32, Output = U>,
{
    /// Convert the unit, scaling as appropriate.
    pub fn convert(self, new_unit: PhysicalUnit) -> Physical<U> {
        Physical {
            unit: new_unit,
            value: self.value * self.unit.scale_factor_to(new_unit),
        }
    }
}
impl<T> Physical<T> {
    /// Apply the given transformation function, regardless of the unit.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Physical<U> {
        Physical {
            unit: self.unit,
            value: f(self.value),
        }
    }
    pub fn into_inner(self) -> T {
        self.value
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LogicalPx<T>(pub T);
impl<T> LogicalPx<T> {
    /// Apply the given transformation function, regardless of the unit.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> LogicalPx<U> {
        LogicalPx(f(self.0))
    }
    pub fn into_inner(self) -> T {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PhysicalPx<T>(pub T);
impl<T> PhysicalPx<T> {
    /// Apply the given transformation function, regardless of the unit.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> PhysicalPx<U> {
        PhysicalPx(f(self.0))
    }
    pub fn into_inner(self) -> T {
        self.0
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[track_caller]
    fn assert_approx_equal(a: f32, b: f32) {
        assert!((a - b).abs() < 0.01, "{a} == {b}");
    }
    #[test]
    fn logical_px() {
        let dpi = Dpi {
            dots_per: (3.0, PhysicalUnit::Inch),
            physical_per_logical_px: 2.0,
        };
        let test = Length::LogicalPx(LogicalPx(7.0));

        assert_approx_equal(test.into_logical_px(&dpi).0, 7.0);
        todo!()
    }
}

pub const CM_PER_IN: f32 = 2.54;
pub const IN_PER_CM: f32 = 1.0 / CM_PER_IN;
/// Constant varies by who you ask - but this is the one defined by W3C.
pub const PT_PER_IN: f32 = 72.0;
pub const IN_PER_PT: f32 = 1.0 / PT_PER_IN;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnitParseError {
    #[error(transparent)]
    Value(#[from] std::num::ParseFloatError),
    #[error("unknown unit")]
    UnrecognizedUnit,
}

/// A physical length allowing specifying sizes and positions in a scale-factor-independent manner.
// Future me: Would anyone ever want to use physical pixels? I can't think of a reason why they would.
#[derive(Clone, Copy, Debug)]
pub enum Length {
    /// Logical pixels. These are [`Resolution`] dependent.
    Logical(f32),
    Inch(f32),
    /// Typographic points, as defined by W3C.
    Point(f32),
    Centimeter(f32),
}
impl Length {
    /// Access the numeric component of the length.
    #[must_use]
    pub fn value(self) -> f32 {
        match self {
            Self::Logical(x) | Self::Inch(x) | Self::Point(x) | Self::Centimeter(x) => x,
        }
    }
    /// Access the numeric component of the length.
    #[must_use]
    pub fn value_mut(&mut self) -> &mut f32 {
        match self {
            Self::Logical(x) | Self::Inch(x) | Self::Point(x) | Self::Centimeter(x) => x,
        }
    }
    /// Fetch the name of the unit.
    #[must_use]
    pub fn unit(self) -> &'static str {
        match self {
            Self::Logical(_) => "px",
            Self::Inch(_) => "in",
            Self::Point(_) => "pt",
            Self::Centimeter(_) => "cm",
        }
    }
    #[must_use]
    /// Convert into logical pixels, under the given resolution.
    pub fn into_logical(self, resolution: Resolution) -> f32 {
        match self {
            Self::Logical(l) => l,
            Self::Point(p) => resolution.into_dpi() * (p * IN_PER_PT),
            Self::Inch(i) => resolution.into_dpi() * i,
            Self::Centimeter(cm) => resolution.into_dpcm() * cm,
        }
    }
    #[must_use]
    /// Convert into inches, under the given resolution.
    pub fn into_inches(self, resolution: Resolution) -> f32 {
        match self {
            Self::Logical(l) => l / resolution.into_dpi(),
            Self::Point(p) => p * IN_PER_PT,
            Self::Inch(i) => i,
            Self::Centimeter(cm) => cm * IN_PER_CM,
        }
    }
    #[must_use]
    /// Convert into centimeters, under the given resolution.
    pub fn into_centimeters(self, resolution: Resolution) -> f32 {
        match self {
            Self::Logical(l) => l / resolution.into_dpcm(),
            Self::Point(p) => p * const { IN_PER_PT * CM_PER_IN },
            Self::Inch(i) => i * CM_PER_IN,
            Self::Centimeter(cm) => cm,
        }
    }
    #[must_use]
    /// Convert into points, under the given resolution.
    pub fn into_points(self, resolution: Resolution) -> f32 {
        match self {
            Self::Logical(l) => l / resolution.into_dpi() * PT_PER_IN,
            Self::Point(p) => p,
            Self::Inch(i) => i * PT_PER_IN,
            Self::Centimeter(cm) => cm * const { IN_PER_CM * PT_PER_IN },
        }
    }
    /// Add another length, under the given resolution, keeping the units of `self`
    #[must_use = "returns a new length and does not modify `self`"]
    pub fn add(self, other: Self, resolution: Resolution) -> Self {
        match self {
            Self::Logical(l) => Self::Logical(l + other.into_logical(resolution)),
            Self::Inch(i) => Self::Inch(i + other.into_inches(resolution)),
            Self::Centimeter(cm) => Self::Centimeter(cm + other.into_centimeters(resolution)),
            Self::Point(pt) => Self::Point(pt + other.into_points(resolution)),
        }
    }
    /// Subtract another length, under the given resolution, keeping the units of `self`
    #[must_use = "returns a new length and does not modify `self`"]
    pub fn subtract(self, other: Self, resolution: Resolution) -> Self {
        self.add(-other, resolution)
    }
}
impl std::fmt::Display for Length {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.value(), self.unit())
    }
}
impl std::ops::Neg for Length {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        *self.value_mut() = -self.value();
        self
    }
}
impl std::ops::Mul<f32> for Length {
    type Output = Self;
    fn mul(mut self, rhs: f32) -> Self::Output {
        *self.value_mut() *= rhs;
        self
    }
}
impl std::ops::Div<f32> for Length {
    type Output = Self;
    fn div(mut self, rhs: f32) -> Self::Output {
        *self.value_mut() /= rhs;
        self
    }
}

impl std::str::FromStr for Length {
    type Err = UnitParseError;
    fn from_str(mut s: &str) -> Result<Self, Self::Err> {
        // parse the unit text first:
        // We only care about ascii since it's just arabic numerals and some predefined latin suffixes.
        // (Alas, we must anticipate unicode input to avoid panics!)
        s = s.trim_ascii_end();
        // Try to split off two chars.
        if s.len() < 2 || !s.is_char_boundary(s.len() - 2) {
            return Err(UnitParseError::UnrecognizedUnit);
        }
        let unit = &s[s.len() - 2..];
        s = &s[..s.len() - 2];

        // Fixme: Full names, case insensitive.
        let mut parsed = match unit {
            "px" => Self::Logical(0.0),
            "in" => Self::Inch(0.0),
            "pt" => Self::Point(0.0),
            "cm" => Self::Centimeter(0.0),
            _ => return Err(UnitParseError::UnrecognizedUnit),
        };

        // Then, parse the floating-point value.
        *parsed.value_mut() = s.trim_ascii().parse()?;
        Ok(parsed)
    }
}

/// Defines the relationship between logical pixels for rendering and physical units.
#[derive(Clone, Copy, Debug)]
pub enum Resolution {
    /// Dots (logical pixels) per inch
    Dpi(f32),
    /// Dots (logical pixels) per centimeter
    Dpcm(f32),
}
impl Resolution {
    #[must_use]
    pub fn value(self) -> f32 {
        match self {
            Self::Dpi(x) | Self::Dpcm(x) => x,
        }
    }
    #[must_use]
    pub fn value_mut(&mut self) -> &mut f32 {
        match self {
            Self::Dpi(x) | Self::Dpcm(x) => x,
        }
    }
    #[must_use]
    pub fn unit(self) -> &'static str {
        match self {
            Resolution::Dpi(_) => "dpi",
            Resolution::Dpcm(_) => "dpcm",
        }
    }
    #[must_use]
    pub fn into_dpi(self) -> f32 {
        match self {
            Resolution::Dpi(i) => i,
            Resolution::Dpcm(cm) => cm * CM_PER_IN,
        }
    }
    #[must_use]
    pub fn into_dpcm(self) -> f32 {
        match self {
            Resolution::Dpi(i) => i * IN_PER_CM,
            Resolution::Dpcm(cm) => cm,
        }
    }
}
impl std::fmt::Display for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.value(), self.unit())
    }
}

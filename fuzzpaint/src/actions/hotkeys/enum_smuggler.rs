//! A hack to smuggle the variant name into and out of a unit enum. This is a crime.
//! (Essentially, imolementing a conversion to and from `&str` for any unit variant that implements [`serde::Serialize`])

use serde::de::IntoDeserializer;

#[derive(Clone, Copy)]
pub struct Enum {
    pub name: &'static str,
    pub index: u32,
    pub variant: &'static str,
}

/// Get the variant name out of an enum value.
#[allow(clippy::missing_errors_doc)]
pub fn smuggle_out<MaybeUnitEnum: serde::Serialize>(
    m: MaybeUnitEnum,
) -> Result<Enum, InvalidSmuggleOperation> {
    m.serialize(SmuggleOut)
}
/// Get an enum value out of it's variant name.
#[allow(clippy::missing_errors_doc)]
pub fn smuggle_in<'a, MaybeUnitEnum: serde::Deserialize<'a>>(
    variant: &'a str,
) -> Result<MaybeUnitEnum, InvalidSmuggleOperation> {
    MaybeUnitEnum::deserialize(SmuggleIn(variant))
}

#[derive(thiserror::Error, Debug)]
#[error("invalid use of `UnitEnumSmuggler`")]
/// Error when the de/serialize implementation attempts to read or write anything other than a single unit variant.
pub struct InvalidSmuggleOperation;
impl serde::ser::Error for InvalidSmuggleOperation {
    fn custom<T>(_: T) -> Self
    where
        T: std::fmt::Display,
    {
        // Dont-care, we don't use this functionality.
        Self
    }
}
impl serde::de::Error for InvalidSmuggleOperation {
    fn duplicate_field(_: &'static str) -> Self {
        Self
    }
    fn invalid_length(_: usize, _: &dyn serde::de::Expected) -> Self {
        Self
    }
    fn invalid_type(_: serde::de::Unexpected, _: &dyn serde::de::Expected) -> Self {
        Self
    }
    fn invalid_value(_: serde::de::Unexpected, _: &dyn serde::de::Expected) -> Self {
        Self
    }
    fn missing_field(_: &'static str) -> Self {
        Self
    }
    fn unknown_field(_: &str, _: &'static [&'static str]) -> Self {
        Self
    }
    fn unknown_variant(_: &str, _: &'static [&'static str]) -> Self {
        Self
    }
    fn custom<T>(_: T) -> Self
    where
        T: std::fmt::Display,
    {
        Self
    }
}

/// Serde Serializer that smuggles the variant name out of an enum with all unit variants.
/// Fails if anything that isn't a single, unit enum variant is attempted to serialize
struct SmuggleOut;
impl serde::ser::Serializer for SmuggleOut {
    type Error = InvalidSmuggleOperation;
    /// The name of the unit variant.
    type Ok = Enum;
    type SerializeMap = serde::ser::Impossible<Enum, InvalidSmuggleOperation>;
    type SerializeSeq = serde::ser::Impossible<Enum, InvalidSmuggleOperation>;
    type SerializeStruct = serde::ser::Impossible<Enum, InvalidSmuggleOperation>;
    type SerializeStructVariant = serde::ser::Impossible<Enum, InvalidSmuggleOperation>;
    type SerializeTuple = serde::ser::Impossible<Enum, InvalidSmuggleOperation>;
    type SerializeTupleStruct = serde::ser::Impossible<Enum, InvalidSmuggleOperation>;
    type SerializeTupleVariant = serde::ser::Impossible<Enum, InvalidSmuggleOperation>;
    fn serialize_unit_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        Ok(Enum {
            name,
            index,
            variant,
        })
    }
    fn serialize_bool(self, _: bool) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_char(self, _: char) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_f32(self, _: f32) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_f64(self, _: f64) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_i128(self, _: i128) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_i16(self, _: i16) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_i32(self, _: i32) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_i64(self, _: i64) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_i8(self, _: i8) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_newtype_struct<T>(self, _: &'static str, _: &T) -> Result<Self::Ok, Self::Error>
    where
        T: serde::Serialize + ?Sized,
    {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_newtype_variant<T>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: serde::Serialize + ?Sized,
    {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_some<T>(self, _: &T) -> Result<Self::Ok, Self::Error>
    where
        T: serde::Serialize + ?Sized,
    {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_str(self, _: &str) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err(InvalidSmuggleOperation)
    }

    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_u128(self, _: u128) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_u16(self, _: u16) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_u32(self, _: u32) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_u64(self, _: u64) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_u8(self, _: u8) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Self::Ok, Self::Error> {
        Err(InvalidSmuggleOperation)
    }

    fn is_human_readable(&self) -> bool {
        true
    }
}

struct SmuggleIn<'a>(&'a str);
impl<'de> serde::de::Deserializer<'de> for SmuggleIn<'de> {
    type Error = InvalidSmuggleOperation;
    fn deserialize_enum<V>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        // *SCREAMING*
        // I cannot figure out the traits involved here for the life of me.
        visitor.visit_enum(self.0.into_deserializer())
    }
    fn deserialize_bool<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_byte_buf<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_bytes<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_char<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_f32<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_f64<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_i16<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_i32<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_i64<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_i8<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_identifier<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_ignored_any<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_map<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_newtype_struct<V>(self, _: &'static str, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_option<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_seq<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_str<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_string<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_struct<V>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        _: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_tuple<V>(self, _: usize, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_tuple_struct<V>(
        self,
        _: &'static str,
        _: usize,
        _: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_u16<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_u32<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_u64<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_u8<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_unit<V>(self, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn deserialize_unit_struct<V>(self, _: &'static str, _: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        Err(InvalidSmuggleOperation)
    }
    fn is_human_readable(&self) -> bool {
        true
    }
}

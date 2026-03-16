type InnerNonZero = std::num::NonZero<u32>;

/// A strongly-typed numeric ID.
///
/// The type parameter acts as a namespace.
#[repr(transparent)]
pub struct ID<T>(
    // WARNING: if you change the repr or inner type, make sure all the deeply
    // unsafe bytemuck stuff below still checks out.

    // Nonzero is used as an optimization for the common case of `Option<ID>`.
    InnerNonZero,
    std::marker::PhantomData<T>,
);

// Trait implementations. We cannot use derives, as they incorrectly bound on T.
unsafe impl<T> Send for ID<T> {}
unsafe impl<T> Sync for ID<T> {}
impl<T> Copy for ID<T> {}
impl<T> Clone for ID<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> std::cmp::PartialEq for ID<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> std::cmp::Eq for ID<T> {}
/// Returns an arbitrary ordering of IDs. The exact details of how IDs order
/// should not be relied on.
impl<T> std::cmp::PartialOrd for ID<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
/// Returns an arbitrary ordering of IDs. The exact details of how IDs order
/// should not be relied on.
impl<T> std::cmp::Ord for ID<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl<T: 'static> std::hash::Hash for ID<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.0, state);
    }
    fn hash_slice<H: std::hash::Hasher>(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        // Defer to NonZeroInner::hash_slice. Ironically, it doesn't even
        // specialize this. But it felt right:tm:
        std::hash::Hash::hash_slice(bytemuck::TransparentWrapper::peel_slice(data), state)
    }
}
impl<T> ID<T> {
    /// Create from a raw integer.
    ///
    /// This should only be used if you are the producer and consumer of these
    /// IDs (e.g. you are implementing the document type) or to re-wrap a
    /// previous call to [`Self::into_nonzero`] with the same ID type. This is
    /// *not* a safety precondition - implementations should handle bogus IDs
    /// gracefully.
    pub const fn new(id: InnerNonZero) -> Self {
        Self(id, std::marker::PhantomData)
    }
    /// Access the ID as a number. It is discouraged to store IDs this way
    /// unless the resulting type erasure is desired or required, prefer storing
    /// `ID<T>` directly.
    pub const fn into_nonzero(self) -> InnerNonZero {
        self.0
    }
}
// We dont impl From<InnerNonZero> as we want to discourage id <-> int round
// trips. They are semantically lossy and weakly-typed.
impl<T> From<ID<T>> for InnerNonZero {
    fn from(value: ID<T>) -> Self {
        value.into_nonzero()
    }
}
impl<T> From<ID<T>> for u32 {
    fn from(value: ID<T>) -> Self {
        value.into_nonzero().get()
    }
}
impl<T: std::any::Any> std::fmt::Debug for ID<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ID<{}>({:X})",
            std::any::type_name::<T>(),
            self.into_nonzero()
        )
    }
}

// SAFETY: repr(transparent) over InnerNonZero.
unsafe impl<T> bytemuck::TransparentWrapper<InnerNonZero> for ID<T> {}

// SAFETY: implimentations deferred to the repr(transparent) inner type.
unsafe impl<T: 'static> bytemuck::Contiguous for ID<T> {
    type Int = <InnerNonZero as bytemuck::Contiguous>::Int;
    const MIN_VALUE: Self::Int = <InnerNonZero as bytemuck::Contiguous>::MIN_VALUE;
    const MAX_VALUE: Self::Int = <InnerNonZero as bytemuck::Contiguous>::MAX_VALUE;
}
unsafe impl<T> bytemuck::CheckedBitPattern for ID<T> {
    type Bits = <InnerNonZero as bytemuck::CheckedBitPattern>::Bits;
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        // Seems silly, but it feels nicer to defer the implimentation.
        InnerNonZero::is_valid_bit_pattern(bits)
    }
}

// SAFETY: repr(transparent) over a type that implements these.
unsafe impl<T: 'static> bytemuck::NoUninit for ID<T> {}
unsafe impl<T> bytemuck::ZeroableInOption for ID<T> {}
unsafe impl<T: 'static> bytemuck::PodInOption for ID<T> {}

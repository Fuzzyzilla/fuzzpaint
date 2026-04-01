pub struct Stroke;
type ID = fuzzpaint_types::id::ID<Stroke>;

pub struct Strokes {
    // Dear future aspen. this is a private implimentation detail. I know it
    // hurts. I know it should be a bump allocator over a virtual allocation.
    // Take a deep breath, you can do it later.
    strokes: Vec<std::sync::Arc<fuzzpaint_types::stroke::aos::Stroke>>,
}
impl Strokes {
    pub fn push(&mut self, stroke: fuzzpaint_types::stroke::aos::Stroke) -> StrokeRef<'_> {
        let arc = std::sync::Arc::new(stroke);
        self.strokes.push(arc.clone());
        let id = u32::try_from(self.strokes.len() + 1).unwrap();
        let id = fuzzpaint_types::id::ID::new(std::num::NonZero::new(id).unwrap());

        StrokeRef {
            id,
            arc,
            _phantom: std::marker::PhantomData,
        }
    }
}
/// A reference to a stroke's data.
#[derive(Clone)]
pub struct StrokeRef<'a> {
    id: ID,
    // Keep private the details of how it's stored, see the impl of Strokes.
    arc: std::sync::Arc<fuzzpaint_types::stroke::aos::Stroke>,
    // Borrows its source, to further seal implementation details.
    _phantom: std::marker::PhantomData<&'a fuzzpaint_types::stroke::aos::Stroke>,
}
impl StrokeRef<'_> {
    /// Downgrade into a handle which can be later upgraded. Useful for storing
    /// strokes in caches without necessarily keeping them allocated.
    pub fn downgrade(&self) -> WeakStroke {
        WeakStroke {
            id: self.id,
            weak: std::sync::Arc::downgrade(&self.arc),
        }
    }
}
impl<'a> std::ops::Deref for StrokeRef<'a> {
    type Target = fuzzpaint_types::stroke::aos::Stroke;
    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

/// A weak reference to a stroke's data. [`Self::upgrade`] can be used to
/// attempt to acquire a strong reference which can be used to access that data.
pub struct WeakStroke {
    id: ID,
    weak: std::sync::Weak<fuzzpaint_types::stroke::aos::Stroke>,
}
impl WeakStroke {
    pub fn upgrade(&self) -> Option<StrokeRef<'_>> {
        // This borrows self on purpose :3. Sealing impl details~
        Some(StrokeRef {
            id: self.id,
            arc: self.weak.upgrade()?,
            _phantom: std::marker::PhantomData,
        })
    }
}

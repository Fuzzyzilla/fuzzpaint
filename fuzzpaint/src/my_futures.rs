use std::{future::Future, pin::Pin, task::Poll};
mod pinvec {
    use std::pin::Pin;
    // Unpin regardless of T's Unpin-ness. (overrides auto-implementation where F: Unpin)
    impl<F> Unpin for PinVec<F> {}
    pub struct PinVec<F> {
        // MUST be private to uphold the Pinning contract
        vec: Vec<F>,
    }
    impl<F> From<Vec<F>> for PinVec<F> {
        fn from(value: Vec<F>) -> Self {
            Self { vec: value }
        }
    }
    impl<F> PinVec<F> {
        pub fn is_empty(&self) -> bool {
            self.vec.is_empty()
        }
        pub fn iter_pinned_mut(&mut self) -> impl Iterator<Item = Pin<&mut F>> {
            self.vec
                .iter_mut()
                .map(|f| unsafe { Pin::new_unchecked(f) })
        }
    }
}
struct Race<F> {
    fs: pinvec::PinVec<F>,
}
impl<F: Future> Future for Race<F> {
    type Output = Option<F::Output>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        if self.fs.is_empty() {
            return Poll::Ready(None);
        }
        for f in self.fs.iter_pinned_mut() {
            match f.poll(cx) {
                Poll::Pending => (),
                Poll::Ready(r) => return Poll::Ready(Some(r)),
            }
        }
        Poll::Pending
    }
}
/// Races a list of futures, yielding the results of the first future to
/// complete, cancelling the rest. Yields `None` if the list is empty. The
/// returned future is not Fused.
pub fn race<F: Future>(
    fs: impl Into<pinvec::PinVec<F>>,
) -> impl Future<Output = Option<F::Output>> {
    Race { fs: fs.into() }
}

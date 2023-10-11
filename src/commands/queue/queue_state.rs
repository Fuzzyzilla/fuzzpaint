struct State {}
/*
// BONK no premature optimization!
/// State where some fields have been overwritten, but the rest are inherited from an older state.
/// Can be flattened back into a State without incurring extra clones when the Arcs are only owned by this object!
struct PartialState {
    base: either::Either<std::sync::Arc<State>, std::sync::Arc<PartialState>>,
}*/

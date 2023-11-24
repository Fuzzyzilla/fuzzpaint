//! # Repositories
//!
//! Global servers which grant shared, immutable access to document resources, like stroke information, point
//! lists, brushes, ect.
//!
//! Can eventually become a multi-layer LRU cache, compressing and dumping cold data onto disk.
//! For now, just store everything in ram :3

#[derive(thiserror::Error, Debug)]
pub enum TryRepositoryError {
    #[error("resource not resident")]
    NotResident,
    #[error("resource not found")]
    NotFound,
}

pub mod points;

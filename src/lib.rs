//! GenAI Vector Database
//! 
//! A production-grade vector database built with Rust for scalable similarity search.

pub mod api;
pub mod faiss_utils;
pub mod indexer;
pub mod ingest;
pub mod metadata_filter;
pub mod metrics;
pub mod minio;
pub mod model;
pub mod query;

pub use model::*;
pub use minio::S3Client;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration values
pub mod defaults {
    pub const DEFAULT_DIMENSION: usize = 256;
    pub const DEFAULT_BATCH_SIZE: usize = 1000;
    pub const DEFAULT_TOP_K: usize = 10;
    pub const DEFAULT_METRIC: &str = "cosine";
}

/// Error types for the vector database
pub mod errors {
    use anyhow::Error;
    
    pub type Result<T> = std::result::Result<T, Error>;
    
    #[derive(Debug, thiserror::Error)]
    pub enum VectorDbError {
        #[error("Index not found: {0}")]
        IndexNotFound(String),
        
        #[error("Invalid dimension: expected {expected}, got {actual}")]
        InvalidDimension { expected: usize, actual: usize },
        
        #[error("Storage error: {0}")]
        StorageError(String),
        
        #[error("Serialization error: {0}")]
        SerializationError(String),
    }
}

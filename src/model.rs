use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

pub const SLICE_ROW_LIMIT: usize = 1_000;  // flush after 1k rows
pub const SLICE_AGE_LIMIT_S: u64 = 30;     // or 30-second age

#[derive(Serialize, Deserialize)]
pub struct CreateIndex {
    pub name: String,
    pub dim: u32,
    pub metric: String, // "cosine" | "euclidean"
    pub nlist: u32,
    pub m: u32,
    pub nbits: u32,
}

#[derive(Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub embedding: Vec<f32>,
    pub meta: serde_json::Value,
    #[serde(default = "Utc::now", with = "chrono::serde::ts_microseconds")]
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize)]
pub struct PutVectors {
    pub index: String,
    pub vectors: Vec<VectorRecord>,
}

#[derive(Serialize, Deserialize)]
pub struct QueryRequest {
    pub index: String,
    pub embedding: Vec<f32>,
    pub topk: usize,
    #[serde(default)]
    pub nprobe: Option<u32>,
    #[serde(default)]
    pub filter: Option<serde_json::Value>,
}

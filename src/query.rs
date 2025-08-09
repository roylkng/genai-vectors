use crate::{minio::S3Client, model::*};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;

pub async fn search(s3: S3Client, req: QueryRequest) -> Result<Value> {
    // 1. Load index manifest to find active shards
    let manifest_key = format!("indexes/{}/manifest.json", req.index);
    
    let manifest_data = match s3.get_object(&manifest_key).await {
        Ok(data) => data,
        Err(_) => {
            // No index exists yet, return empty results
            return Ok(serde_json::json!({
                "results": [],
                "took_ms": 0
            }));
        }
    };

    let manifest: IndexManifest = serde_json::from_slice(&manifest_data)
        .context("Failed to parse index manifest")?;

    let start = std::time::Instant::now();
    let mut all_results = Vec::new();

    // 2. Search each shard
    for shard in &manifest.shards {
        let results = search_shard(&s3, &req, shard).await?;
        all_results.extend(results);
    }

    // 3. Sort by score and take top-k
    all_results.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(req.topk);

    let took_ms = start.elapsed().as_millis();

    Ok(serde_json::json!({
        "results": all_results,
        "took_ms": took_ms
    }))
}

async fn search_shard(s3: &S3Client, req: &QueryRequest, shard: &ShardInfo) -> Result<Vec<SearchResult>> {
    // 1. Load shard metadata (including vectors for simple brute force search)
    let metadata = load_shard_metadata(s3, shard).await?;

    // 2. Simple brute force vector search
    let mut results = Vec::new();
    let query_vector = &req.embedding;

    for (i, id) in metadata.ids.iter().enumerate() {
        if let Some(vector) = metadata.vectors.get(i) {
            let score = match shard.metric.as_str() {
                "cosine" => cosine_similarity(query_vector, vector),
                "euclidean" => euclidean_similarity(query_vector, vector),
                _ => cosine_similarity(query_vector, vector),
            };

            let vector_meta = metadata.metadata.get(id)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));

            results.push(SearchResult {
                id: id.clone(),
                score,
                metadata: vector_meta,
            });
        }
    }

    // Sort by score and take top-k for this shard
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(req.topk);

    Ok(results)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    let distance: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    
    // Convert distance to similarity (higher score = more similar)
    1.0 / (1.0 + distance)
}

async fn load_shard_metadata(s3: &S3Client, shard: &ShardInfo) -> Result<ShardMetadata> {
    let metadata_data = s3.get_object(&shard.metadata_path).await
        .context("Failed to download shard metadata")?;
    
    let metadata: ShardMetadata = serde_json::from_slice(&metadata_data)
        .context("Failed to parse shard metadata")?;
    
    Ok(metadata)
}

#[derive(serde::Deserialize)]
struct IndexManifest {
    index_name: String,
    dim: u32,
    metric: String,
    shards: Vec<ShardInfo>,
    total_vectors: usize,
}

#[derive(serde::Deserialize)]
struct ShardInfo {
    shard_id: String,
    index_path: String,
    metadata_path: String,
    vector_count: usize,
    metric: String,
    created_at: String,
}

#[derive(serde::Deserialize)]
struct ShardMetadata {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>, // Store vectors for brute force search
    metadata: HashMap<String, Value>,
}

#[derive(serde::Serialize)]
struct SearchResult {
    id: String,
    score: f32,
    metadata: Value,
}

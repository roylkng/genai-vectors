use crate::{minio::S3Client, model::*};
use crate::faiss_utils;
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

    // 2. Search each shard using Faiss
    for shard in &manifest.shards {
        let results = search_shard(&s3, &req, shard, &manifest).await?;
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

async fn search_shard(s3: &S3Client, req: &QueryRequest, shard: &ShardInfo, manifest: &IndexManifest) -> Result<Vec<SearchResult>> {
    // Load metadata JSON for this shard
    let metadata_bytes = s3.get_object(&shard.metadata_path).await
        .context("Failed to load shard metadata")?;
    let metadata_map: HashMap<String, Value> = serde_json::from_slice(&metadata_bytes)
        .context("Failed to parse shard metadata")?;

    // Load ID map (numeric ID to original string ID)
    let id_map_key = shard.index_path.replace("index.faiss", "id_map.json");
    let id_map_bytes = s3.get_object(&id_map_key).await
        .context("Failed to load id map")?;
    let id_map: Vec<(i64, String)> = serde_json::from_slice(&id_map_bytes)
        .context("Failed to parse id map")?;
    let id_lookup: HashMap<i64, String> = id_map.into_iter().collect();

    // Load Mock Faiss index
    // Download index file to a temporary location
    let index_bytes = s3.get_object(&shard.index_path).await
        .context("Failed to download index file")?;
    let local_index_path = format!("/tmp/{}.faiss", shard.shard_id);
    std::fs::write(&local_index_path, &index_bytes)
        .context("Failed to write temp index file")?;
    
    let mut index = faiss_utils::load_index(&local_index_path)
        .context("Failed to load mock Faiss index")?;

    // Determine nprobe value (from request, manifest default, or calculated)
    let nprobe = req.nprobe
        .or(manifest.default_nprobe)
        .or_else(|| {
            // Calculate a reasonable nprobe if not specified
            Some(faiss_utils::calculate_optimal_nprobe(100)) // Default assumption
        });

    // Search using Mock Faiss
    let (distances, faiss_ids) = faiss_utils::search_index(
        &mut index, 
        &req.embedding, 
        req.topk, 
        nprobe
    ).context("Failed to search mock Faiss index")?;

    // Convert results back to original format
    let mut results = Vec::new();
    for (distance, faiss_id) in distances.iter().zip(faiss_ids.iter()) {
        if *faiss_id == -1 {
            // Faiss returns -1 for empty slots
            continue;
        }
        
        if let Some(original_id) = id_lookup.get(faiss_id) {
            let score = match shard.metric.as_str() {
                "cosine" => *distance, // Mock Faiss already returns similarity for cosine
                "euclidean" => *distance, // Mock implementation handles this
                _ => *distance,
            };

            let vector_meta = metadata_map.get(original_id)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));

            results.push(SearchResult {
                id: original_id.clone(),
                score,
                metadata: vector_meta,
            });
        }
    }

    // Clean up temp file
    let _ = std::fs::remove_file(&local_index_path);

    Ok(results)
}

#[derive(serde::Deserialize)]
struct IndexManifest {
    index_name: String,
    dim: u32,
    metric: String,
    default_nprobe: Option<u32>,
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

#[derive(serde::Serialize)]
struct SearchResult {
    id: String,
    score: f32,
    metadata: Value,
}

async fn load_shard_metadata(s3: &S3Client, shard: &ShardInfo) -> Result<ShardMetadata> {
    let metadata_data = s3.get_object(&shard.metadata_path).await
        .context("Failed to download shard metadata")?;
    
    let metadata: ShardMetadata = serde_json::from_slice(&metadata_data)
        .context("Failed to parse shard metadata")?;
    
    Ok(metadata)
}

#[derive(serde::Deserialize)]
struct ShardMetadata {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>, // Store vectors for brute force search (not used in Faiss version)
    metadata: HashMap<String, Value>,
}

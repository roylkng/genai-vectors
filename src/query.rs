use crate::{minio::S3Client, model::*};
use crate::faiss_utils::FaissIndex;
// use crate::metadata_filter::MetadataFilter;  // TODO: Re-enable after fixing module path
use crate::metrics::get_metrics_collector;
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;

pub async fn search(s3: S3Client, req: QueryRequest) -> Result<Value> {
    let _measurement = crate::measure_operation!("query.search");
    let search_start = std::time::Instant::now();
    
    // Track query parameters
    get_metrics_collector().track_metric("query.topk", req.topk as f64);
    get_metrics_collector().track_metric("query.vector_dimension", req.embedding.len() as f64);
    
    // 1. Load index manifest to find active shards
    let manifest_key = format!("indexes/{}/manifest.json", req.index);
    
    let manifest_data = match s3.get_object(&manifest_key).await {
        Ok(data) => data,
        Err(_) => {
            // No index exists yet, return empty results
            get_metrics_collector().track_metric("query.index_not_found", 1.0);
            return Ok(serde_json::json!({
                "results": [],
                "took_ms": 0
            }));
        }
    };

    let manifest: IndexManifest = serde_json::from_slice(&manifest_data)
        .context("Failed to parse index manifest")?;

    get_metrics_collector().track_metric("query.shards_count", manifest.shards.len() as f64);

    let start = std::time::Instant::now();
    let mut all_results = Vec::new();

    // 2. Search each shard using Faiss
    for (shard_idx, shard) in manifest.shards.iter().enumerate() {
        let shard_start = std::time::Instant::now();
                let results = search_shard(&s3, &req, shard, &manifest).await?;
        let shard_time = shard_start.elapsed();
        
        get_metrics_collector().track_metric(&format!("query.shard_{}_time_ms", shard_idx), shard_time.as_millis() as f64);
        get_metrics_collector().track_metric(&format!("query.shard_{}_results", shard_idx), results.len() as f64);
        
        all_results.extend(results);
    }

    // 3. Sort by score and take top-k
    all_results.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(req.topk);

    let took_ms = start.elapsed().as_millis();
    let total_search_time = search_start.elapsed();
    
    get_metrics_collector().track_metric("query.total_time_ms", total_search_time.as_millis() as f64);
    get_metrics_collector().track_metric("query.results_returned", all_results.len() as f64);

    Ok(serde_json::json!({
        "results": all_results,
        "took_ms": took_ms
    }))
}

async fn search_shard(
    s3: &S3Client, 
    req: &QueryRequest,
    shard: &ShardInfo, 
    _manifest: &IndexManifest
) -> Result<Vec<SearchResult>> {
    let _measurement = crate::measure_operation!("query.search_shard");
    
    // Load metadata JSON for this shard
    let metadata_start = std::time::Instant::now();
    let metadata_bytes = s3.get_object(&shard.metadata_path).await
        .context("Failed to load shard metadata")?;
    let metadata_map: HashMap<String, Value> = serde_json::from_slice(&metadata_bytes)
        .context("Failed to parse shard metadata")?;
    let metadata_load_time = metadata_start.elapsed();

    // Apply metadata pre-filtering if specified
    let pre_filtered_ids: Option<Vec<String>> = if let Some(_filter_value) = &req.filter {
        // TODO: Re-enable metadata filtering after fixing module path
        // match MetadataFilter::try_from(filter_value.clone()) {
        //     Ok(filter) => {
        //         let filtered = filter.pre_filter_ids(&metadata_map);
        //         get_metrics_collector().track_metric("query.pre_filtered_candidates", filtered.len() as f64);
        //         Some(filtered)
        //     }
        //     Err(e) => {
        //         tracing::warn!("Invalid metadata filter: {}, proceeding without filter", e);
        //         None
        //     }
        // }
        None  // Temporary: disable filtering
    } else {
        None
    };

    // Load ID map (numeric ID to original string ID)
    let id_map_key = shard.index_path.replace("index.faiss", "id_map.json");
    let id_map_bytes = s3.get_object(&id_map_key).await
        .context("Failed to load id map")?;
    let id_map: Vec<(i64, String)> = serde_json::from_slice(&id_map_bytes)
        .context("Failed to parse id map")?;
    let id_lookup: HashMap<i64, String> = id_map.into_iter().collect();
    
    get_metrics_collector().track_metric("query.metadata_load_time_ms", metadata_load_time.as_millis() as f64);
    get_metrics_collector().track_metric("query.id_map_size", id_lookup.len() as f64);

    // Load index using real Faiss
    let index_bytes = s3.get_object(&shard.index_path).await
        .context("Failed to download index file")?;
    let local_index_path = format!("/tmp/{}.faiss", shard.shard_id);
    std::fs::write(&local_index_path, &index_bytes)
        .context("Failed to write temp index file")?;
    
    // Also try to download the config file
    let config_path = format!("/tmp/{}.config.json", shard.shard_id);
    let index_config_key = shard.index_path.replace("index.faiss", "index.config.json");
    if let Ok(config_bytes) = s3.get_object(&index_config_key).await {
        std::fs::write(&config_path, &config_bytes).ok();
    }
    
    let mut index = FaissIndex::load_from_file(&local_index_path)
        .context("Failed to load Faiss index")?;

    // Adjust search parameters if we have pre-filtered candidates
    let search_k = if let Some(ref filtered_ids) = pre_filtered_ids {
        // If we have pre-filtered candidates, we might need to search more to find enough matches
        let expansion_factor = (metadata_map.len() as f64 / filtered_ids.len() as f64).ceil() as usize;
        (req.topk * expansion_factor.max(2)).min(index.ntotal())
    } else {
        req.topk
    };

    // Search using real Faiss
    let (distances, faiss_ids) = index.search(&req.embedding, search_k, req.nprobe.map(|n| n as usize))
        .context("Failed to search Faiss index")?;

    // Convert results back to original format with post-filtering
    let mut results = Vec::new();
    for (distance, faiss_id) in distances.iter().zip(faiss_ids.iter()) {
        if *faiss_id == -1 {
            // Faiss returns -1 for empty slots
            continue;
        }
        
        // Map faiss_id back to original string ID
        if let Some(original_id) = id_lookup.get(faiss_id) {
            // Apply post-filtering if we have pre-filtered candidates
            if let Some(ref filtered_ids) = pre_filtered_ids {
                if !filtered_ids.contains(original_id) {
                    continue; // Skip this result as it doesn't match the filter
                }
            }

            let score = match shard.metric.as_str() {
                "cosine" => *distance,
                "euclidean" => -distance, // Convert distance to similarity score
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

            // Stop when we have enough results
            if results.len() >= req.topk {
                break;
            }
        }
    }

    // Clean up temp files
    let _ = std::fs::remove_file(&local_index_path);
    let _ = std::fs::remove_file(&config_path);

    Ok(results)
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

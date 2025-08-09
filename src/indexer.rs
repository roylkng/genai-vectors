use crate::{minio::S3Client, model::*};
use crate::faiss_utils::{
    build_ivfpq_index, calculate_optimal_nlist, 
    calculate_optimal_pq_params,
};
use crate::metrics::get_metrics_collector;
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;

pub async fn run_once() -> Result<()> {
    let _bucket = std::env::var("VEC_BUCKET")?;
    let s3 = S3Client::from_env().await?;

    // 1. List staged slices and group by index
    let staged_objects = s3.list_objects("staged/").await?;
    let mut index_slices: HashMap<String, Vec<String>> = HashMap::new();

    for object_key in staged_objects {
        if let Some(index_name) = extract_index_name_from_path(&object_key) {
            index_slices.entry(index_name).or_default().push(object_key);
        }
    }

    // 2. Process each index
    for (index_name, slice_paths) in index_slices {
        if !slice_paths.is_empty() {
            process_index_slices(&s3, &index_name, slice_paths).await?;
        }
    }

    Ok(())
}

async fn process_index_slices(s3: &S3Client, index_name: &str, slice_paths: Vec<String>) -> Result<()> {
    let _measurement = crate::measure_operation!("indexer.process_index_slices");
    tracing::info!("Processing {} slices for index {} with real Faiss IVF-PQ", 
                   slice_paths.len(), index_name);

    // Track the number of slices being processed
    get_metrics_collector().track_metric("indexer.slices_count", slice_paths.len() as f64);

    // 1. Load all vectors from slices with optimized batch processing
    let mut all_vectors = Vec::new();
    let mut metadata = HashMap::new();
    let mut vector_ids = Vec::new();

    // Pre-allocate capacity for better performance
    let estimated_capacity = slice_paths.len() * 1000; // Estimate ~1000 vectors per slice
    all_vectors.reserve(estimated_capacity);
    metadata.reserve(estimated_capacity);
    vector_ids.reserve(estimated_capacity);

    let load_start = std::time::Instant::now();
    for slice_path in &slice_paths {
        let slice_data = s3.get_object(slice_path).await?;
        let slice_text = String::from_utf8(slice_data.to_vec())?;
        
        for line in slice_text.lines() {
            if !line.trim().is_empty() {
                let record: VectorRecord = serde_json::from_str(line)?;
                all_vectors.push(record.embedding.clone());
                metadata.insert(record.id.clone(), record.meta);
                vector_ids.push(record.id);
            }
        }
    }
    
    let load_duration = load_start.elapsed();
    get_metrics_collector().track_metric("indexer.vector_loading_time_ms", load_duration.as_millis() as f64);
    get_metrics_collector().track_metric("indexer.vectors_loaded", all_vectors.len() as f64);

    if all_vectors.is_empty() {
        tracing::warn!("No vectors found in slices for index {}", index_name);
        return Ok(());
    }

    // 2. Get index configuration (dimension, metric, etc.)
    let config = get_or_create_index_config(s3, index_name, all_vectors[0].len()).await?;

    // 3. Create multiple shards with Faiss IVF-PQ indexing
    const MAX_VECTORS_PER_SHARD: usize = 50_000;  // Larger shards for better Faiss performance
    let total_vectors = all_vectors.len();
    let num_shards = (total_vectors + MAX_VECTORS_PER_SHARD - 1) / MAX_VECTORS_PER_SHARD;
    
    get_metrics_collector().track_metric("indexer.shards_created", num_shards as f64);
    get_metrics_collector().track_metric("indexer.vectors_per_shard", (total_vectors as f64) / (num_shards as f64));
    
    let _shard_creation_times: Vec<std::time::Duration> = Vec::new();
    for shard_index in 0..num_shards {
        let _shard_start = std::time::Instant::now();
        let start_idx = shard_index * MAX_VECTORS_PER_SHARD;
        let end_idx = std::cmp::min(start_idx + MAX_VECTORS_PER_SHARD, total_vectors);
        
        let shard_vectors = &all_vectors[start_idx..end_idx];
        let shard_ids_slice = &vector_ids[start_idx..end_idx];
        let shard_metadata: HashMap<String, Value> = shard_ids_slice.iter()
            .filter_map(|id| metadata.get(id).map(|meta| (id.clone(), meta.clone())))
            .collect();

        let shard_id = Uuid::new_v4().to_string();
        
        // Build and train index using selected backend (real Faiss or mock)
        let index_start = std::time::Instant::now();
        
        // Calculate optimal parameters based on shard size (real implementation)
        let shard_nlist = calculate_optimal_nlist(shard_vectors.len());
        
        let (optimal_m, optimal_nbits) = calculate_optimal_pq_params(
            config.dim as usize, 
            0.85 // Target 85% compression for real Faiss
        );
        
        let index = build_ivfpq_index(
            config.dim as usize,
            shard_nlist,
            optimal_m,
            optimal_nbits,
            &config.metric,
            &shard_vectors,
        )?;
        let _index_creation_time = index_start.elapsed();
        
        // Generate numeric IDs for Faiss from string IDs
        let faiss_ids: Vec<i64> = (0..shard_ids_slice.len() as i64).collect();
        
        // The vectors are already added during build_ivfpq_index, so no need to add again
        
        // Save index using Faiss binary format
        let local_path = format!("/tmp/{}.faiss", shard_id);
        index.save_to_file(&local_path)?;
        
        let index_object_path = format!("indexes/{}/shards/{}/index.faiss", index_name, shard_id);
        
        // Read the index file and upload as bytes
        let index_data = std::fs::read(&local_path)
            .context("Failed to read Faiss index file")?;
        s3.put_object(&index_object_path, index_data.clone().into()).await?;
        
        tracing::info!("Uploaded shard {}: {} bytes (Faiss binary format)", 
                        shard_id, index_data.len());

        // Clean up temp file
        let _ = std::fs::remove_file(&local_path);

        // Write mapping from hashed numeric ID to original string ID
        let id_map: Vec<(i64, String)> = faiss_ids.iter().cloned()
            .zip(shard_ids_slice.iter().cloned()).collect();
        let id_map_data = serde_json::to_vec(&id_map)?;
        let id_map_path = format!("indexes/{}/shards/{}/id_map.json", index_name, shard_id);
        s3.put_object(&id_map_path, id_map_data.into()).await?;

        // Persist metadata JSON (without vectors since they're in the Faiss index)
        let metadata_path = format!("indexes/{}/shards/{}/metadata.json", index_name, shard_id);
        let metadata_data = serde_json::to_vec(&shard_metadata)?;
        s3.put_object(&metadata_path, metadata_data.into()).await?;

        let shard_info = ShardInfo {
            shard_id: shard_id.clone(),
            index_path: index_object_path,
            metadata_path,
            vector_count: shard_ids_slice.len(),
            metric: config.metric.clone(),
            created_at: Utc::now().format("%Y%m%dT%H%M%S").to_string(),
        };
        
        // Update manifest for each shard
        update_index_manifest(s3, index_name, shard_info, &config).await?;
        
                tracing::info!("Created Real Faiss IVF-PQ shard {}/{} with {} vectors", 
                       shard_index + 1, num_shards, end_idx - start_idx);
    }

    // 5. Clean up processed slices
    for slice_path in slice_paths {
        s3.delete_object(&slice_path).await?;
    }

    tracing::info!("Successfully processed {} vectors for index {} in {} Faiss IVF-PQ shards", 
                   all_vectors.len(), index_name, num_shards);
    Ok(())
}

fn extract_index_name_from_path(path: &str) -> Option<String> {
    // Extract index name from path like "staged/demo-4d/slice-123.jsonl"
    if let Some(parts) = path.strip_prefix("staged/") {
        if let Some(slash_pos) = parts.find('/') {
            return Some(parts[..slash_pos].to_string());
        }
    }
    None
}

async fn get_or_create_index_config(s3: &S3Client, index_name: &str, dimension: usize) -> Result<IndexConfig> {
    let config_key = format!("indexes/{}/config.json", index_name);
    
    match s3.get_object(&config_key).await {
        Ok(data) => {
            let config: IndexConfig = serde_json::from_slice(&data).context("Failed to parse index config")?;
            tracing::info!("Loaded existing index config: {}D, {}, nlist={}, PQ={}x{}", 
                         config.dim, config.metric, config.nlist, config.m, config.nbits);
            Ok(config)
        }
        Err(e) => {
            tracing::warn!("Failed to load index config: {}, creating optimized config based on dataset characteristics", e);
            
            // Estimate total dataset size from previous manifests or current batch
            let estimated_total_vectors = estimate_total_dataset_size(s3, index_name, dimension * 100).await;
            
            // Calculate optimal parameters for real Faiss IVF-PQ based on estimated size
            let optimal_nlist = calculate_optimal_nlist(estimated_total_vectors);
            let (optimal_m, optimal_nbits) = calculate_optimal_pq_params(dimension, 0.85);
            
            // Ensure nlist is feasible for current data size
            let feasible_nlist = std::cmp::min(optimal_nlist, dimension / 4); // Conservative bound
            
            let config = IndexConfig {
                name: index_name.to_string(),
                dim: dimension as u32,
                metric: "cosine".to_string(),
                nlist: feasible_nlist as u32,
                m: optimal_m as u32,
                nbits: optimal_nbits as u32,
            };
            
            let config_data = serde_json::to_vec(&config)?;
            s3.put_object(&config_key, config_data.into()).await?;
            
            tracing::info!(
                "Created optimized Faiss IVF-PQ config: {}D, {}, nlist={}, PQ={}x{} (estimated {} total vectors)", 
                config.dim, config.metric, config.nlist, config.m, config.nbits, estimated_total_vectors
            );
            Ok(config)
        }
    }
}

async fn estimate_total_dataset_size(s3: &S3Client, index_name: &str, default_estimate: usize) -> usize {
    // Try to load existing manifest to get historical data
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    
    match s3.get_object(&manifest_key).await {
        Ok(data) => {
            if let Ok(manifest) = serde_json::from_slice::<IndexManifest>(&data) {
                // Use existing vector count as base estimate, assume 50% growth
                let projected_size = (manifest.total_vectors as f64 * 1.5) as usize;
                tracing::info!("Estimated dataset size from manifest: {} vectors (current: {})", 
                             projected_size, manifest.total_vectors);
                return projected_size.max(1000); // Minimum reasonable size
            }
        }
        Err(_) => {
            // No existing manifest, check for other indexes in the bucket for size patterns
            if let Ok(staged_objects) = s3.list_objects("staged/").await {
                let staged_count = staged_objects.len();
                if staged_count > 0 {
                    // Estimate based on staging activity: ~1000 vectors per slice
                    let estimated = staged_count * 1000;
                    tracing::info!("Estimated dataset size from staged objects: {} vectors ({} slices)", 
                                 estimated, staged_count);
                    return estimated.max(1000);
                }
            }
        }
    }
    
    tracing::info!("Using default dataset size estimate: {} vectors", default_estimate);
    default_estimate
}

async fn update_index_manifest(s3: &S3Client, index_name: &str, new_shard: ShardInfo, config: &IndexConfig) -> Result<()> {
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    
    let mut manifest = match s3.get_object(&manifest_key).await {
        Ok(data) => serde_json::from_slice::<IndexManifest>(&data)?,
        Err(_) => IndexManifest {
            index_name: index_name.to_string(),
            dim: config.dim,
            metric: config.metric.clone(),
            shards: Vec::new(),
            total_vectors: 0,
        },
    };

    manifest.total_vectors += new_shard.vector_count;
    manifest.shards.push(new_shard);

    let manifest_data = serde_json::to_vec(&manifest)?;
    s3.put_object(&manifest_key, manifest_data.into()).await?;

    Ok(())
}

#[derive(serde::Deserialize, serde::Serialize)]
struct IndexConfig {
    name: String,
    dim: u32,
    metric: String,
    nlist: u32,
    m: u32,
    nbits: u32,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct IndexManifest {
    index_name: String,
    dim: u32,
    metric: String,
    shards: Vec<ShardInfo>,
    total_vectors: usize,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ShardInfo {
    shard_id: String,
    index_path: String,
    metadata_path: String,
    vector_count: usize,
    metric: String,
    created_at: String,
}

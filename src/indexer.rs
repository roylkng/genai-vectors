use crate::{minio::S3Client, model::*};
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
    tracing::info!("Processing {} slices for index {} (optimized)", slice_paths.len(), index_name);

    // 1. Load all vectors from slices with optimized batch processing
    let mut all_vectors = Vec::new();
    let mut metadata = HashMap::new();
    let mut vector_ids = Vec::new();

    // Pre-allocate capacity for better performance
    let estimated_capacity = slice_paths.len() * 1000; // Estimate ~1000 vectors per slice
    all_vectors.reserve(estimated_capacity);
    metadata.reserve(estimated_capacity);
    vector_ids.reserve(estimated_capacity);

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

    if all_vectors.is_empty() {
        tracing::warn!("No vectors found in slices for index {}", index_name);
        return Ok(());
    }

    // 2. Get index configuration (dimension, metric, etc.)
    let config = get_or_create_index_config(s3, index_name, all_vectors[0].len()).await?;

    // 3. Create multiple shards with optimized size limit (reduce from 50k to 10k for better performance)
    const MAX_VECTORS_PER_SHARD: usize = 10_000;  // Optimized for better search performance
    let total_vectors = all_vectors.len();
    let num_shards = (total_vectors + MAX_VECTORS_PER_SHARD - 1) / MAX_VECTORS_PER_SHARD;
    
    tracing::info!("Creating {} shards for {} vectors (max {} vectors per shard - optimized)", 
                   num_shards, total_vectors, MAX_VECTORS_PER_SHARD);

    for shard_index in 0..num_shards {
        let start_idx = shard_index * MAX_VECTORS_PER_SHARD;
        let end_idx = std::cmp::min(start_idx + MAX_VECTORS_PER_SHARD, total_vectors);
        
        let shard_vectors = &all_vectors[start_idx..end_idx];
        let shard_ids = &vector_ids[start_idx..end_idx];
        let shard_metadata: HashMap<String, Value> = shard_ids.iter()
            .filter_map(|id| metadata.get(id).map(|meta| (id.clone(), meta.clone())))
            .collect();

        let shard_id = Uuid::new_v4().to_string();
        let shard_info = create_shard(s3, index_name, &shard_id, shard_vectors, shard_ids, &shard_metadata, &config).await?;
        
        // 4. Update manifest for each shard
        update_index_manifest(s3, index_name, shard_info, &config).await?;
        
        tracing::info!("Created shard {}/{} with {} vectors", 
                       shard_index + 1, num_shards, end_idx - start_idx);
    }

    // 5. Clean up processed slices
    for slice_path in slice_paths {
        s3.delete_object(&slice_path).await?;
    }

    tracing::info!("Successfully processed {} vectors for index {} in {} shards", 
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
            tracing::info!("Loaded existing index config: {}D, {}", config.dim, config.metric);
            Ok(config)
        }
        Err(e) => {
            tracing::warn!("Failed to load index config: {}, creating default", e);
            // Create default config
            let config = IndexConfig {
                name: index_name.to_string(),
                dim: dimension as u32,
                metric: "cosine".to_string(),
                nlist: 16,
                m: 4,
                nbits: 8,
            };
            
            let config_data = serde_json::to_vec(&config)?;
            s3.put_object(&config_key, config_data.into()).await?;
            tracing::info!("Created new index config: {}D, {}", config.dim, config.metric);
            Ok(config)
        }
    }
}

async fn create_shard(
    s3: &S3Client,
    index_name: &str,
    shard_id: &str,
    vectors: &[Vec<f32>],
    vector_ids: &[String],
    metadata: &HashMap<String, Value>,
    config: &IndexConfig,
) -> Result<ShardInfo> {
    let timestamp = Utc::now().format("%Y%m%dT%H%M%S");
    
    // Store shard metadata (including vectors for simple brute force search)
    let shard_metadata = ShardMetadata {
        ids: vector_ids.to_vec(),
        vectors: vectors.to_vec(),
        metadata: metadata.clone(),
    };
    let metadata_path = format!("indexes/{}/shards/{}/metadata.json", index_name, shard_id);
    let metadata_data = serde_json::to_vec(&shard_metadata)?;
    s3.put_object(&metadata_path, metadata_data.into()).await?;

    // For now, we don't create a separate index file - everything is in metadata
    let index_path = format!("indexes/{}/shards/{}/index.json", index_name, shard_id);
    let index_data = serde_json::json!({
        "type": "simple_brute_force",
        "created_at": timestamp.to_string()
    });
    s3.put_object(&index_path, serde_json::to_vec(&index_data)?.into()).await?;

    Ok(ShardInfo {
        shard_id: shard_id.to_string(),
        index_path,
        metadata_path,
        vector_count: vector_ids.len(),
        metric: config.metric.clone(),
        created_at: timestamp.to_string(),
    })
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

#[derive(serde::Deserialize, serde::Serialize)]
struct ShardMetadata {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>, // Store vectors for brute force search
    metadata: HashMap<String, Value>,
}

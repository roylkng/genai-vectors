use crate::{minio::S3Client, model::*};
use crate::faiss_utils;
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
    tracing::info!("Processing {} slices for index {} with Faiss IVF-PQ", slice_paths.len(), index_name);

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

    // 3. Create multiple shards with Faiss IVF-PQ indexing
    const MAX_VECTORS_PER_SHARD: usize = 50_000;  // Larger shards for better Faiss performance
    let total_vectors = all_vectors.len();
    let num_shards = (total_vectors + MAX_VECTORS_PER_SHARD - 1) / MAX_VECTORS_PER_SHARD;
    
    tracing::info!("Creating {} Faiss IVF-PQ shards for {} vectors (max {} vectors per shard)", 
                   num_shards, total_vectors, MAX_VECTORS_PER_SHARD);

    for shard_index in 0..num_shards {
        let start_idx = shard_index * MAX_VECTORS_PER_SHARD;
        let end_idx = std::cmp::min(start_idx + MAX_VECTORS_PER_SHARD, total_vectors);
        
        let shard_vectors = &all_vectors[start_idx..end_idx];
        let shard_ids_slice = &vector_ids[start_idx..end_idx];
        let shard_metadata: HashMap<String, Value> = shard_ids_slice.iter()
            .filter_map(|id| metadata.get(id).map(|meta| (id.clone(), meta.clone())))
            .collect();

        let shard_id = Uuid::new_v4().to_string();
        
        // Build and train a Faiss IVF-PQ index for this shard
        let mut index = faiss_utils::build_ivfpq_index(
            config.dim as usize,
            config.nlist,
            config.m,
            config.nbits,
            &config.metric,
            shard_vectors,
        )?;
        
        // Generate numeric IDs for Faiss from string IDs
        let faiss_ids: Vec<i64> = shard_ids_slice.iter()
            .map(|id| faiss_utils::hash_string_to_i64(id))
            .collect();
        
        // Add vectors to the trained index
        faiss_utils::add_vectors(&mut index, shard_vectors, &faiss_ids)?;

        // Save index to a local temp file and upload to S3
        let local_path = format!("/tmp/{}.faiss", shard_id);
        faiss_utils::save_index(&index, &local_path)?;
        let index_object_path = format!("indexes/{}/shards/{}/index.faiss", index_name, shard_id);
        
        // Read the index file and upload as bytes
        let index_data = std::fs::read(&local_path)
            .context("Failed to read Faiss index file")?;
        s3.put_object(&index_object_path, index_data.into()).await?;
        
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
        
        tracing::info!("Created Faiss IVF-PQ shard {}/{} with {} vectors", 
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
            tracing::warn!("Failed to load index config: {}, creating default with optimal Faiss parameters", e);
            
            // Calculate optimal parameters for Faiss IVF-PQ based on estimated dataset size
            let estimated_total_vectors = dimension * 10000; // Rough estimate, will be adjusted
            let optimal_nlist = faiss_utils::calculate_optimal_nlist(estimated_total_vectors);
            
            // Create default config with reasonable IVF-PQ parameters
            let config = IndexConfig {
                name: index_name.to_string(),
                dim: dimension as u32,
                metric: "cosine".to_string(),
                nlist: optimal_nlist,
                m: 8,  // 8 subspaces for PQ
                nbits: 8,  // 8 bits per subspace
                default_nprobe: Some(faiss_utils::calculate_optimal_nprobe(optimal_nlist)),
            };
            
            let config_data = serde_json::to_vec(&config)?;
            s3.put_object(&config_key, config_data.into()).await?;
            tracing::info!("Created new Faiss IVF-PQ index config: {}D, {}, nlist={}, PQ={}x{}, default_nprobe={}", 
                         config.dim, config.metric, config.nlist, config.m, config.nbits, 
                         config.default_nprobe.unwrap_or(0));
            Ok(config)
        }
    }
}

async fn update_index_manifest(s3: &S3Client, index_name: &str, new_shard: ShardInfo, config: &IndexConfig) -> Result<()> {
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    
    let mut manifest = match s3.get_object(&manifest_key).await {
        Ok(data) => serde_json::from_slice::<IndexManifest>(&data)?,
        Err(_) => IndexManifest {
            index_name: index_name.to_string(),
            dim: config.dim,
            metric: config.metric.clone(),
            default_nprobe: config.default_nprobe,
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
    default_nprobe: Option<u32>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct IndexManifest {
    index_name: String,
    dim: u32,
    metric: String,
    default_nprobe: Option<u32>,
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

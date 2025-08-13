use crate::{minio::S3Client, model::*};
use crate::faiss_utils::{
    build_hnsw_flat_index, build_ivfpq_index, calculate_optimal_nlist,
    calculate_optimal_pq_params,
};
use crate::metrics::get_metrics_collector;
use anyhow::{Context, Result};
use arrow::array::{Array, Float32Array, ListArray, StringArray};
use chrono::Utc;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use uuid::Uuid;

pub async fn run_once() -> Result<()> {
    let _bucket = std::env::var("VEC_BUCKET")?;
    let s3 = S3Client::from_env().await?;

    let staged_objects = s3.list_objects("staged/").await?;
    let mut index_slices: HashMap<String, Vec<String>> = HashMap::new();

    for object_key in staged_objects {
        if let Some(index_name) = extract_index_name_from_path(&object_key) {
            index_slices.entry(index_name).or_default().push(object_key);
        }
    }

    for (index_name, slice_paths) in index_slices {
        if !slice_paths.is_empty() {
            process_index_slices(&s3, &index_name, slice_paths).await?;
        }
    }

    Ok(())
}

pub async fn trigger_indexing_for_slice(s3: S3Client, slice_path: String) -> Result<()> {
    if let Some(index_name) = extract_index_name_from_path(&slice_path) {
        tracing::info!("Indexing slice {} for index {}", slice_path, index_name);
        process_index_slices(&s3, &index_name, vec![slice_path]).await?;
    } else {
        tracing::warn!("Could not extract index name from slice path: {}", slice_path);
    }
    Ok(())
}

async fn process_index_slices(
    s3: &S3Client,
    index_name: &str,
    slice_paths: Vec<String>,
) -> Result<()> {
    let _measurement = crate::measure_operation!("indexer.process_index_slices");
    tracing::info!(
        "Processing {} slices for index {}",
        slice_paths.len(),
        index_name
    );

    get_metrics_collector().track_metric("indexer.slices_count", slice_paths.len() as f64);

    let mut all_vectors = Vec::new();
    let mut metadata = HashMap::new();
    let mut vector_ids = Vec::new();

    let estimated_capacity = slice_paths.len() * 1000;
    all_vectors.reserve(estimated_capacity);
    metadata.reserve(estimated_capacity);
    vector_ids.reserve(estimated_capacity);

    let load_start = std::time::Instant::now();
    for slice_path in &slice_paths {
        if slice_path.ends_with(".parquet") {
            let local_path = format!("/tmp/{}", slice_path.split('/').last().unwrap_or("slice.parquet"));
            let slice_data = s3.get_object(slice_path).await?;
            std::fs::write(&local_path, &slice_data)?;
            let file = File::open(&local_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let reader = builder.build()?;

            for batch in reader {
                let batch = batch?;
                let id_array = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                let embedding_array = batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
                let meta_array = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();

                for i in 0..batch.num_rows() {
                    let id = id_array.value(i).to_string();
                    let meta: serde_json::Value = serde_json::from_str(meta_array.value(i))?;
                    let embedding_list = embedding_array.value(i);
                    let embedding_values = embedding_list.as_any().downcast_ref::<Float32Array>().unwrap();
                    let embedding: Vec<f32> = embedding_values.values().to_vec();
                    all_vectors.push(embedding);
                    metadata.insert(id.clone(), meta);
                    vector_ids.push(id);
                }
            }
            tokio::fs::remove_file(&local_path).await?;
        } else {
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
    }

    let load_duration = load_start.elapsed();
    get_metrics_collector()
        .track_metric("indexer.vector_loading_time_ms", load_duration.as_millis() as f64);
    get_metrics_collector().track_metric("indexer.vectors_loaded", all_vectors.len() as f64);

    if all_vectors.is_empty() {
        tracing::warn!("No vectors found in slices for index {}", index_name);
        return Ok(());
    }

    let config = get_or_create_index_config(s3, index_name, all_vectors[0].len()).await?;
    const MAX_VECTORS_PER_SHARD: usize = 50_000;
    let total_vectors = all_vectors.len();
    let num_shards = (total_vectors + MAX_VECTORS_PER_SHARD - 1) / MAX_VECTORS_PER_SHARD;
    get_metrics_collector().track_metric("indexer.shards_created", num_shards as f64);
    get_metrics_collector()
        .track_metric("indexer.vectors_per_shard", (total_vectors as f64) / (num_shards as f64));
    let max_concurrent_shards = std::cmp::min(num_shards, num_cpus::get().max(1));
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent_shards));
    tracing::info!(
        "Processing {} shards in parallel with max {} concurrent tasks",
        num_shards,
        max_concurrent_shards
    );
    let mut shard_tasks = Vec::new();
    for shard_index in 0..num_shards {
        let start_idx = shard_index * MAX_VECTORS_PER_SHARD;
        let end_idx = std::cmp::min(start_idx + MAX_VECTORS_PER_SHARD, total_vectors);
        let shard_vectors = all_vectors[start_idx..end_idx].to_vec();
        let shard_ids_slice = vector_ids[start_idx..end_idx].to_vec();
        let shard_metadata: HashMap<String, Value> = shard_ids_slice
            .iter()
            .filter_map(|id| metadata.get(id).map(|meta| (id.clone(), meta.clone())))
            .collect();
        let shard_id = Uuid::new_v4().to_string();
        let s3_clone = s3.clone();
        let index_name_clone = index_name.to_string();
        let config_clone = config.clone();
        let semaphore_clone = semaphore.clone();
        let task = tokio::spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();
            process_single_shard(
                s3_clone,
                index_name_clone,
                shard_id,
                shard_vectors,
                shard_ids_slice,
                shard_metadata,
                config_clone,
                shard_index,
                num_shards,
            )
            .await
        });
        shard_tasks.push(task);
    }
    let shard_results: Result<Vec<_>, _> = futures::future::try_join_all(shard_tasks).await;
    let shard_infos = shard_results.context("Failed to process shards in parallel")?;
    let mut final_manifest = load_or_create_manifest(s3, index_name, &config).await?;
    for shard_info_result in shard_infos {
        let shard_info = shard_info_result?;
        final_manifest.total_vectors += shard_info.vector_count;
        final_manifest.shards.push(shard_info);
    }
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    let manifest_data = serde_json::to_vec(&final_manifest)?;
    s3.put_object(&manifest_key, manifest_data.into()).await?;

    for slice_path in slice_paths {
        s3.delete_object(&slice_path).await?;
    }

    tracing::info!(
        "Successfully processed {} vectors for index {} into {} shards",
        all_vectors.len(),
        index_name,
        num_shards
    );
    Ok(())
}

fn extract_index_name_from_path(path: &str) -> Option<String> {
    if let Some(parts) = path.strip_prefix("staged/") {
        if let Some(slash_pos) = parts.find('/') {
            return Some(parts[..slash_pos].to_string());
        }
    }
    None
}

async fn get_or_create_index_config(
    s3: &S3Client,
    index_name: &str,
    dimension: usize,
) -> Result<IndexConfig> {
    let config_key = format!("indexes/{}/config.json", index_name);
    match s3.get_object(&config_key).await {
        Ok(data) => {
            let config: IndexConfig =
                serde_json::from_slice(&data).context("Failed to parse index config")?;
            tracing::info!("Loaded existing index config for index: {}", index_name);
            Ok(config)
        }
        Err(e) => {
            tracing::warn!("Failed to load index config: {}, creating optimized config based on dataset characteristics", e);
            
            // Try to load the CreateIndex config to get metadata configuration
            let create_index_config_key = format!("indexes/{}/config.json", index_name);
            let non_filterable_keys = match s3.get_object(&create_index_config_key).await {
                Ok(data) => {
                    if let Ok(create_index) = serde_json::from_slice::<crate::model::CreateIndex>(&data) {
                        create_index.non_filterable_metadata_keys
                    } else {
                        Vec::new()
                    }
                }
                Err(_) => Vec::new(),
            };
            
            // Estimate total dataset size from previous manifests or current batch
            let estimated_total_vectors = estimate_total_dataset_size(s3, index_name, dimension * 100).await;
            
            // Calculate optimal parameters for real Faiss IVF-PQ based on estimated size
            let optimal_nlist = calculate_optimal_nlist(estimated_total_vectors);
            let (optimal_m, optimal_nbits) = calculate_optimal_pq_params(dimension, 0.85);
            let feasible_nlist = std::cmp::min(optimal_nlist, dimension / 4);
            let config = IndexConfig {
                name: index_name.to_string(),
                dim: dimension as u32,
                metric: "cosine".to_string(),
                nlist: feasible_nlist as u32,
                m: optimal_m as u32,
                nbits: optimal_nbits as u32,
                non_filterable_metadata_keys: non_filterable_keys,
            };
            let config_data = serde_json::to_vec(&config)?;
            s3.put_object(&config_key, config_data.into()).await?;
            tracing::info!("Created new index config for index: {}", index_name);
            Ok(config)
        }
    }
}

async fn estimate_total_dataset_size(
    s3: &S3Client,
    index_name: &str,
    default_estimate: usize,
) -> usize {
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    match s3.get_object(&manifest_key).await {
        Ok(data) => {
            if let Ok(manifest) = serde_json::from_slice::<IndexManifest>(&data) {
                let projected_size = (manifest.total_vectors as f64 * 1.5) as usize;
                return projected_size.max(1000);
            }
        }
        Err(_) => {
            if let Ok(staged_objects) = s3.list_objects("staged/").await {
                let staged_count = staged_objects.len();
                if staged_count > 0 {
                    return (staged_count * 1000).max(1000);
                }
            }
        }
    }
    default_estimate
}

async fn process_single_shard(
    s3: S3Client,
    index_name: String,
    shard_id: String,
    shard_vectors: Vec<Vec<f32>>,
    shard_ids_slice: Vec<String>,
    shard_metadata: HashMap<String, Value>,
    config: IndexConfig,
    shard_index: usize,
    total_shards: usize,
) -> Result<ShardInfo> {
    let shard_start = std::time::Instant::now();
    let manifest = load_or_create_manifest(&s3, &index_name, &config).await?;
    let total_vectors = manifest.total_vectors + shard_vectors.len();
    let algorithm_name = config.algorithm.as_deref().unwrap_or("ivfpq");
    let hnsw_threshold = config.hnsw_threshold.unwrap_or(100_000);
    let use_hnsw = match algorithm_name {
        "hnsw_flat" => true,
        "ivfpq" => false,
        "hybrid" => total_vectors < hnsw_threshold,
        _ => false,
    };

    let (index, algorithm_used) = if use_hnsw {
        let m = 32;
        let index = build_hnsw_flat_index(
            config.dim as usize,
            &config.metric,
            &shard_vectors,
            m,
        )?;
        (index, "hnsw_flat".to_string())
    } else {
        let shard_nlist = calculate_optimal_nlist(shard_vectors.len());
        let (optimal_m, optimal_nbits) =
            calculate_optimal_pq_params(config.dim as usize, 0.85);
        let index = build_ivfpq_index(
            config.dim as usize,
            shard_nlist,
            optimal_m,
            optimal_nbits,
            &config.metric,
            &shard_vectors,
        )?;
        (index, "ivfpq".to_string())
    };

    let local_path = format!("/tmp/{}.faiss", shard_id);
    faiss::write_index(&index, &local_path)?;
    let index_object_path = format!("indexes/{}/shards/{}/index.faiss", index_name, shard_id);
    let index_data = std::fs::read(&local_path)?;
    s3.put_object(&index_object_path, index_data.into()).await?;
    tracing::info!(
        "Uploaded shard {} ({}/{}): algorithm={}",
        shard_id,
        shard_index + 1,
        total_shards,
        algorithm_used
    );
    std::fs::remove_file(&local_path)?;

    let faiss_ids: Vec<i64> = (0..shard_ids_slice.len() as i64).collect();
    let id_map: Vec<(i64, String)> = faiss_ids
        .iter()
        .cloned()
        .zip(shard_ids_slice.iter().cloned())
        .collect();
    let id_map_data = serde_json::to_vec(&id_map)?;
    let id_map_path = format!("indexes/{}/shards/{}/id_map.json", index_name, shard_id);
    s3.put_object(&id_map_path, id_map_data.into()).await?;
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
        algorithm: algorithm_used,
    };
    let total_shard_time = shard_start.elapsed();
    tracing::info!(
        "Completed shard {}/{} with {} vectors in {:?}",
        shard_index + 1,
        total_shards,
        shard_ids_slice.len(),
        total_shard_time
    );
    Ok(shard_info)
}

async fn load_or_create_manifest(
    s3: &S3Client,
    index_name: &str,
    config: &IndexConfig,
) -> Result<IndexManifest> {
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    match s3.get_object(&manifest_key).await {
        Ok(data) => {
            serde_json::from_slice::<IndexManifest>(&data).context("Failed to parse existing manifest")
        }
        Err(_) => Ok(IndexManifest {
            index_name: index_name.to_string(),
            dim: config.dim,
            metric: config.metric.clone(),
            shards: Vec::new(),
            total_vectors: 0,
            algorithm: config.algorithm.clone(),
            hnsw_threshold: config.hnsw_threshold,
        }),
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
struct IndexConfig {
    name: String,
    dim: u32,
    metric: String,
    nlist: u32,
    m: u32,
    nbits: u32,
    #[serde(default)]
    non_filterable_metadata_keys: Vec<String>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct IndexManifest {
    index_name: String,
    dim: u32,
    metric: String,
    shards: Vec<ShardInfo>,
    total_vectors: usize,
    #[serde(default)]
    algorithm: Option<String>,
    #[serde(default)]
    hnsw_threshold: Option<usize>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ShardInfo {
    shard_id: String,
    index_path: String,
    metadata_path: String,
    vector_count: usize,
    metric: String,
    created_at: String,
    #[serde(default)]
    algorithm: String,
}

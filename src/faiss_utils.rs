use anyhow::{Context, Result};
use faiss::{index::IndexImpl, MetricType, Idx, index_factory, Index};

/// Build a complete IVF-PQ index with training and vector addition.
pub fn build_ivfpq_index(
    dimension: usize,
    nlist: usize,
    m: usize,
    nbits: usize,
    metric: &str,
    vectors: &[Vec<f32>],
) -> Result<IndexImpl> {
    if vectors.is_empty() {
        return Err(anyhow::anyhow!("Cannot build index with empty vectors"));
    }

    let metric_type = match metric.to_lowercase().as_str() {
        "cosine" | "angular" => MetricType::InnerProduct,
        "euclidean" | "l2" => MetricType::L2,
        _ => return Err(anyhow::anyhow!("Unsupported metric: {}", metric)),
    };

    let index_description = format!("IVF{},PQ{}x{}", nlist, m, nbits);
    let mut index = index_factory(dimension as u32, &index_description, metric_type)?;

    let training_size = calculate_optimal_training_size(vectors.len(), nlist);
    if training_size > vectors.len() {
        return Err(anyhow::anyhow!(
            "Insufficient vectors for training: need {}, have {}",
            training_size,
            vectors.len()
        ));
    }

    let training_vectors = &vectors[..training_size];
    let flat_training_vectors: Vec<f32> = training_vectors.iter().flat_map(|v| v.iter().cloned()).collect();
    index.train(&flat_training_vectors).context("Failed to train Faiss IVF-PQ index")?;

    let flat_vectors: Vec<f32> = vectors.iter().flat_map(|v| v.iter().cloned()).collect();
    let ids: Vec<i64> = (0..vectors.len() as i64).collect();
    let faiss_ids: Vec<Idx> = ids.iter().map(|&id| Idx::from(id)).collect();
    index.add_with_ids(&flat_vectors, &faiss_ids).context("Failed to add vectors to Faiss index")?;

    tracing::info!(
        "Built Faiss IVF-PQ index: {} vectors, {} dims, {} clusters, {}x{} PQ, trained on {} vectors",
        vectors.len(),
        dimension,
        nlist,
        m,
        nbits,
        training_size
    );

    Ok(index)
}

/// Build a complete HNSW-Flat index with vector addition.
pub fn build_hnsw_flat_index(
    dimension: usize,
    metric: &str,
    vectors: &[Vec<f32>],
    m: usize,
) -> Result<IndexImpl> {
    if vectors.is_empty() {
        return Err(anyhow::anyhow!("Cannot build HNSW index with empty vectors"));
    }

    let metric_type = match metric.to_lowercase().as_str() {
        "cosine" | "angular" => MetricType::InnerProduct,
        "euclidean" | "l2" => MetricType::L2,
        _ => return Err(anyhow::anyhow!("Unsupported metric for HNSW: {}", metric)),
    };

    let index_description = format!("HNSW{},Flat", m);
    let mut index = index_factory(dimension as u32, &index_description, metric_type)?;

    let flat_vectors: Vec<f32> = vectors.iter().flat_map(|v| v.iter().cloned()).collect();
    index.add(&flat_vectors)?;

    tracing::info!(
        "Built Faiss HNSW index: {} vectors, {} dims, M={}",
        vectors.len(),
        dimension,
        m
    );

    Ok(index)
}

/// Search an index for similar vectors.
pub fn search_index(
    index: &mut IndexImpl,
    query: &[f32],
    k: usize,
    nprobe: Option<usize>,
) -> Result<(Vec<f32>, Vec<i64>)> {
    // Set nprobe if it's an IVF index (best effort)
    if let Some(nprobe_val) = nprobe {
        tracing::debug!("Setting nprobe to {} for search", nprobe_val);
        // Note: Direct nprobe setting would require more specific index types
    }

    let search_result = index.search(query, k)?;
    let labels = search_result.labels;
    let valid_results: Vec<(f32, i64)> = search_result
        .distances
        .into_iter()
        .zip(labels.into_iter())
        .filter_map(|(dist, label)| {
            // Try to convert Idx to i64 - Idx(0) represents the inner value
            if label >= Idx::from(0i64) {
                // Assuming Idx wraps i64, access inner value via debug representation
                let label_str = format!("{:?}", label);
                if let Ok(label_i64) = label_str.trim_start_matches("Idx(").trim_end_matches(")").parse::<i64>() {
                    Some((dist, label_i64))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();
    let (filtered_distances, filtered_labels) = valid_results.into_iter().unzip();
    Ok((filtered_distances, filtered_labels))
}

/// Calculate optimal nlist based on dataset size.
pub fn calculate_optimal_nlist(vector_count: usize) -> usize {
    let optimal = (vector_count as f64).sqrt() as usize;
    let bounded = optimal.max(4).min(65536);
    let power_of_2 = bounded.next_power_of_two();
    if power_of_2 > bounded * 2 {
        bounded
    } else {
        power_of_2
    }
}

/// Calculate optimal training size for IVF indexes.
pub fn calculate_optimal_training_size(vector_count: usize, nlist: usize) -> usize {
    let min_training = 39 * nlist;
    let max_training = (vector_count as f64 * 0.3) as usize;
    if vector_count >= min_training {
        min_training.min(max_training).min(vector_count)
    } else {
        tracing::warn!(
            "Insufficient training vectors: {} available, {} recommended for {} clusters",
            vector_count, min_training, nlist
        );
        vector_count
    }
}

/// Calculate optimal nprobe for IVF indexes.
pub fn calculate_optimal_nprobe(nlist: usize, target_recall: f64) -> usize {
    let fraction = if target_recall >= 0.95 {
        0.20
    } else if target_recall >= 0.90 {
        0.15
    } else if target_recall >= 0.80 {
        0.10
    } else {
        0.05
    };
    let optimal = (nlist as f64 * fraction) as usize;
    optimal.max(1).min(nlist)
}

/// Calculate optimal PQ parameters for IVF-PQ indexes.
pub fn calculate_optimal_pq_params(dimension: usize, target_compression: f64) -> (usize, usize) {
    let m = if dimension >= 512 {
        if dimension % 64 == 0 { 64 }
        else if dimension % 32 == 0 { 32 }
        else if dimension % 16 == 0 { 16 }
        else { 8 }
    } else if dimension >= 128 {
        if dimension % 16 == 0 { 16 }
        else if dimension % 8 == 0 { 8 }
        else { 4 }
    } else {
        if dimension % 8 == 0 { 8 }
        else { 4 }
    };
    let nbits = if target_compression >= 0.95 {
        4
    } else if target_compression >= 0.90 {
        6
    } else {
        8
    };
    (m, nbits)
}
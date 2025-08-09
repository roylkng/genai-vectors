use anyhow::{Context, Result};

// Mock Faiss implementation for development
// In production, you would use the real Faiss library

pub struct MockIndex {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    pub ids: Vec<i64>,
    pub metric: String,
    pub nlist: u32,
    pub m: u32,
    pub nbits: u32,
    pub trained: bool,
}

impl MockIndex {
    pub fn new(dim: usize, nlist: u32, m: u32, nbits: u32, metric: String) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            metric,
            nlist,
            m,
            nbits,
            trained: false,
        }
    }

    pub fn train(&mut self, _training_data: &[f32]) -> Result<()> {
        self.trained = true;
        tracing::info!("Mock Faiss index trained");
        Ok(())
    }

    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[i64]) -> Result<()> {
        if !self.trained {
            return Err(anyhow::anyhow!("Index must be trained before adding vectors"));
        }

        let num_vectors = vectors.len() / self.dim;
        if vectors.len() != num_vectors * self.dim {
            return Err(anyhow::anyhow!("Vector data length doesn't match dimension"));
        }

        for i in 0..num_vectors {
            let start = i * self.dim;
            let end = start + self.dim;
            self.vectors.push(vectors[start..end].to_vec());
            self.ids.push(ids[i]);
        }

        tracing::debug!("Added {} vectors to mock index", num_vectors);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<(Vec<f32>, Vec<i64>)> {
        if query.len() != self.dim {
            return Err(anyhow::anyhow!("Query dimension mismatch"));
        }

        let mut results: Vec<(f32, i64)> = Vec::new();

        for (i, vector) in self.vectors.iter().enumerate() {
            let score = match self.metric.as_str() {
                "cosine" => cosine_similarity(query, vector),
                "euclidean" => euclidean_similarity(query, vector),
                _ => cosine_similarity(query, vector),
            };
            results.push((score, self.ids[i]));
        }

        // Sort by score (higher is better for similarity)
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        let distances: Vec<f32> = results.iter().map(|(score, _)| *score).collect();
        let ids: Vec<i64> = results.iter().map(|(_, id)| *id).collect();

        Ok((distances, ids))
    }

    pub fn set_nprobe(&mut self, _nprobe: i32) -> Result<()> {
        // Mock implementation - just log it
        tracing::debug!("Mock index nprobe set to {}", _nprobe);
        Ok(())
    }

    pub fn d(&self) -> i32 {
        self.dim as i32
    }
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

/// Build an IVF-PQ index with the given hyperparameters and train it on `train_vectors`.
///
/// `dim` is the dimensionality of the vectors.
/// `nlist` specifies the number of coarse centroids (typically sqrt(N) where N is dataset size).
/// `m` and `nbits` configure the product quantizer (m subspaces, nbits bits per subspace).
/// `metric` should be "cosine" or "euclidean".
pub fn build_ivfpq_index(
    dim: usize,
    nlist: u32,
    m: u32,
    nbits: u32,
    metric: &str,
    train_vectors: &[Vec<f32>],
) -> Result<MockIndex> {
    tracing::info!("Creating mock Faiss IVF-PQ index: {}D, {}, nlist={}, PQ={}x{}", 
                   dim, metric, nlist, m, nbits);
    
    let mut index = MockIndex::new(dim, nlist, m, nbits, metric.to_string());

    // Flatten training vectors for training
    let mut training_data: Vec<f32> = Vec::with_capacity(dim * train_vectors.len());
    for vector in train_vectors {
        if vector.len() != dim {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}", 
                dim, vector.len()
            ));
        }
        training_data.extend_from_slice(vector);
    }
    
    tracing::info!("Training mock Faiss index on {} vectors", train_vectors.len());
    index.train(&training_data)?;
    
    Ok(index)
}

/// Add vectors with explicit IDs to the index.
pub fn add_vectors(
    index: &mut MockIndex,
    vectors: &[Vec<f32>],
    ids: &[i64],
) -> Result<()> {
    if vectors.len() != ids.len() {
        return Err(anyhow::anyhow!(
            "Vector and ID count mismatch: {} vectors, {} IDs", 
            vectors.len(), ids.len()
        ));
    }
    
    let dim = index.dim;
    let mut flat_vectors: Vec<f32> = Vec::with_capacity(dim * vectors.len());
    
    for vector in vectors {
        if vector.len() != dim {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}", 
                dim, vector.len()
            ));
        }
        flat_vectors.extend_from_slice(vector);
    }
    
    tracing::debug!("Adding {} vectors to mock Faiss index", vectors.len());
    index.add_with_ids(&flat_vectors, ids)?;
    
    Ok(())
}

/// Save an index to the given file path.
pub fn save_index(index: &MockIndex, path: &str) -> Result<()> {
    tracing::debug!("Saving mock Faiss index to: {}", path);
    
    // Serialize the mock index as JSON
    let index_data = serde_json::json!({
        "type": "mock_ivf_pq",
        "dim": index.dim,
        "metric": index.metric,
        "nlist": index.nlist,
        "m": index.m,
        "nbits": index.nbits,
        "vectors": index.vectors,
        "ids": index.ids,
        "trained": index.trained
    });
    
    std::fs::write(path, serde_json::to_vec(&index_data)?)
        .context("Failed to save mock index")?;
    Ok(())
}

/// Load an index from the given file path.
pub fn load_index(path: &str) -> Result<MockIndex> {
    tracing::debug!("Loading mock Faiss index from: {}", path);
    
    let data = std::fs::read(path)
        .context("Failed to read mock index file")?;
    let index_data: serde_json::Value = serde_json::from_slice(&data)
        .context("Failed to parse mock index JSON")?;
    
    let mut index = MockIndex::new(
        index_data["dim"].as_u64().unwrap() as usize,
        index_data["nlist"].as_u64().unwrap() as u32,
        index_data["m"].as_u64().unwrap() as u32,
        index_data["nbits"].as_u64().unwrap() as u32,
        index_data["metric"].as_str().unwrap().to_string(),
    );
    
    index.trained = index_data["trained"].as_bool().unwrap_or(false);
    
    // Restore vectors and IDs
    if let Some(vectors_arr) = index_data["vectors"].as_array() {
        for vector_val in vectors_arr {
            if let Some(vector_arr) = vector_val.as_array() {
                let vector: Vec<f32> = vector_arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                index.vectors.push(vector);
            }
        }
    }
    
    if let Some(ids_arr) = index_data["ids"].as_array() {
        for id_val in ids_arr {
            if let Some(id) = id_val.as_i64() {
                index.ids.push(id);
            }
        }
    }
    
    Ok(index)
}

/// Search the index for the top-k most similar vectors.
/// Returns (distances, ids) where both are vectors of length k.
pub fn search_index(
    index: &mut MockIndex,
    query_vector: &[f32],
    k: usize,
    nprobe: Option<u32>,
) -> Result<(Vec<f32>, Vec<i64>)> {
    let dim = index.dim;
    if query_vector.len() != dim {
        return Err(anyhow::anyhow!(
            "Query vector dimension mismatch: expected {}, got {}", 
            dim, query_vector.len()
        ));
    }
    
    // Set nprobe if specified
    if let Some(nprobe_val) = nprobe {
        index.set_nprobe(nprobe_val as i32)?;
    }
    
    tracing::debug!("Searching mock Faiss index for top-{} results with nprobe={:?}", k, nprobe);
    index.search(query_vector, k)
}

/// Calculate optimal nlist (number of clusters) based on dataset size.
/// Rule of thumb: nlist ≈ sqrt(N) where N is the number of vectors.
pub fn calculate_optimal_nlist(num_vectors: usize) -> u32 {
    let sqrt_n = (num_vectors as f64).sqrt() as u32;
    // Clamp between reasonable bounds
    sqrt_n.max(16).min(65536)
}

/// Calculate optimal nprobe based on nlist.
/// Rule of thumb: nprobe ≈ nlist/8 to nlist/4 for good recall/speed tradeoff.
pub fn calculate_optimal_nprobe(nlist: u32) -> u32 {
    (nlist / 8).max(1).min(nlist)
}

/// Hash a string ID to a 64-bit integer for use with Faiss.
/// Uses a simple hash function to convert string IDs to numeric IDs.
pub fn hash_string_to_i64(id: &str) -> i64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_optimal_nlist() {
        assert_eq!(calculate_optimal_nlist(1000), 31);
        assert_eq!(calculate_optimal_nlist(10000), 100);
        assert_eq!(calculate_optimal_nlist(1000000), 1000);
        assert_eq!(calculate_optimal_nlist(100), 16); // minimum
    }

    #[test]
    fn test_calculate_optimal_nprobe() {
        assert_eq!(calculate_optimal_nprobe(64), 8);
        assert_eq!(calculate_optimal_nprobe(16), 2);
        assert_eq!(calculate_optimal_nprobe(8), 1); // minimum
    }

    #[test]
    fn test_hash_string_to_i64() {
        let id1 = "vector_123";
        let id2 = "vector_456";
        
        let hash1 = hash_string_to_i64(id1);
        let hash2 = hash_string_to_i64(id2);
        
        // Hashes should be different
        assert_ne!(hash1, hash2);
        
        // Same input should produce same hash
        assert_eq!(hash1, hash_string_to_i64(id1));
    }

    #[test]
    fn test_mock_index_basic_flow() {
        let dim = 4;
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        
        let mut index = build_ivfpq_index(dim, 16, 8, 8, "cosine", &vectors).unwrap();
        
        let ids = vec![1, 2, 3];
        add_vectors(&mut index, &vectors, &ids).unwrap();
        
        let query = vec![1.0, 0.1, 0.0, 0.0];
        let (scores, result_ids) = search_index(&mut index, &query, 2, None).unwrap();
        
        assert_eq!(result_ids.len(), 2);
        assert_eq!(result_ids[0], 1); // Should be most similar to first vector
    }
}

use anyhow::{Context, Result};
use faiss::{Index, IndexImpl, MetricType};
use std::path::Path;

/// Real Faiss IVF-PQ Index wrapper for production vector search
pub struct FaissIndex {
    index: Box<dyn Index>,
    dimension: usize,
    metric_type: MetricType,
    nlist: usize,
    m: usize,
    nbits: usize,
}

impl FaissIndex {
    /// Create a new IVF-PQ index with the specified parameters
    pub fn new(
        dimension: usize,
        nlist: usize,
        m: usize,
        nbits: usize,
        metric: &str,
    ) -> Result<Self> {
        let metric_type = match metric.to_lowercase().as_str() {
            "cosine" | "angular" => MetricType::InnerProduct,
            "euclidean" | "l2" => MetricType::L2,
            _ => return Err(anyhow::anyhow!("Unsupported metric: {}", metric)),
        };

        // Create IVF-PQ index: IndexIVFPQ(quantizer, d, nlist, m, nbits)
        let quantizer = faiss::index_factory(dimension, "Flat", Some(metric_type))?;
        let index = faiss::IndexIVFPQ::new(quantizer, dimension, nlist, m, nbits, metric_type)?;

        Ok(FaissIndex {
            index: Box::new(index),
            dimension,
            metric_type,
            nlist,
            m,
            nbits,
        })
    }

    /// Train the index on a sample of vectors
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(anyhow::anyhow!("Training vectors cannot be empty"));
        }

        // Flatten vectors for Faiss API
        let flat_vectors: Vec<f32> = training_vectors
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect();

        // Train the index
        self.index.train(training_vectors.len(), &flat_vectors)?;
        
        tracing::info!(
            "Trained Faiss IVF-PQ index: {} vectors, {} dimensions, {} clusters, {}x{} PQ",
            training_vectors.len(),
            self.dimension,
            self.nlist,
            self.m,
            self.nbits
        );

        Ok(())
    }

    /// Add vectors to the index with their IDs
    pub fn add_vectors(&mut self, vectors: &[Vec<f32>], ids: &[i64]) -> Result<()> {
        if vectors.len() != ids.len() {
            return Err(anyhow::anyhow!(
                "Vector count ({}) must match ID count ({})",
                vectors.len(),
                ids.len()
            ));
        }

        if vectors.is_empty() {
            return Ok(());
        }

        // Verify all vectors have the correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dimension {
                return Err(anyhow::anyhow!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    vector.len(),
                    self.dimension
                ));
            }
        }

        // Flatten vectors for Faiss API
        let flat_vectors: Vec<f32> = vectors
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect();

        // Add vectors with IDs to the index
        self.index.add_with_ids(vectors.len(), &flat_vectors, ids)?;

        tracing::debug!("Added {} vectors to Faiss index", vectors.len());
        Ok(())
    }

    /// Search the index for the k nearest neighbors
    pub fn search(&self, query_vector: &[f32], k: usize, nprobe: Option<usize>) -> Result<(Vec<f32>, Vec<i64>)> {
        if query_vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query vector dimension {} does not match index dimension {}",
                query_vector.len(),
                self.dimension
            ));
        }

        // Set nprobe if specified
        if let Some(nprobe_val) = nprobe {
            if let Some(ivf_index) = self.index.as_any().downcast_ref::<faiss::IndexIVF>() {
                ivf_index.set_nprobe(nprobe_val);
            }
        }

        // Perform search
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![0i64; k];

        self.index.search(1, query_vector, k, &mut distances, &mut labels)?;

        // Filter out invalid results (Faiss returns -1 for missing results)
        let valid_results: Vec<(f32, i64)> = distances
            .into_iter()
            .zip(labels.into_iter())
            .filter(|(_, label)| *label >= 0)
            .collect();

        let (filtered_distances, filtered_labels): (Vec<f32>, Vec<i64>) = valid_results.into_iter().unzip();

        Ok((filtered_distances, filtered_labels))
    }

    /// Save the index to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        faiss::write_index(&*self.index, path.as_ref())?;
        tracing::info!("Saved Faiss index to {}", path.as_ref().display());
        Ok(())
    }

    /// Load an index from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let index = faiss::read_index(path.as_ref())?;
        
        // Extract index parameters (this is simplified - in practice you'd store these in metadata)
        let dimension = index.d() as usize;
        
        Ok(FaissIndex {
            index,
            dimension,
            metric_type: MetricType::L2, // Default, should be stored in metadata
            nlist: 0, // These would be stored in metadata
            m: 0,
            nbits: 0,
        })
    }

    /// Get the number of vectors in the index
    pub fn ntotal(&self) -> usize {
        self.index.ntotal() as usize
    }

    /// Check if the index is trained
    pub fn is_trained(&self) -> bool {
        self.index.is_trained()
    }

    /// Get index dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Build a complete IVF-PQ index with training and vector addition
pub fn build_ivfpq_index(
    dimension: usize,
    nlist: usize,
    m: usize,
    nbits: usize,
    metric: &str,
    vectors: &[Vec<f32>],
) -> Result<FaissIndex> {
    if vectors.is_empty() {
        return Err(anyhow::anyhow!("Cannot build index with empty vectors"));
    }

    // Create the index
    let mut index = FaissIndex::new(dimension, nlist, m, nbits, metric)?;

    // Use a subset of vectors for training (Faiss recommendation: use 30x nlist vectors)
    let training_size = (30 * nlist).min(vectors.len());
    let training_vectors = &vectors[..training_size];

    // Train the index
    index.train(training_vectors)
        .context("Failed to train Faiss IVF-PQ index")?;

    // Generate sequential IDs for the vectors
    let ids: Vec<i64> = (0..vectors.len() as i64).collect();

    // Add all vectors to the index
    index.add_vectors(vectors, &ids)
        .context("Failed to add vectors to Faiss index")?;

    tracing::info!(
        "Built Faiss IVF-PQ index: {} vectors, {} dims, {} clusters, {}x{} PQ",
        vectors.len(),
        dimension,
        nlist,
        m,
        nbits
    );

    Ok(index)
}

/// Calculate optimal nlist based on dataset size
pub fn calculate_optimal_nlist(vector_count: usize) -> usize {
    // Faiss recommendation: nlist = sqrt(N) for good performance
    let optimal = (vector_count as f64).sqrt() as usize;
    // Ensure reasonable bounds
    optimal.max(16).min(65536)
}

/// Calculate optimal nprobe based on nlist and desired recall
pub fn calculate_optimal_nprobe(nlist: usize, target_recall: f64) -> usize {
    // Higher nprobe = better recall but slower search
    // Typical range: 1-20% of nlist
    let fraction = if target_recall >= 0.95 {
        0.20 // Search 20% of clusters for high recall
    } else if target_recall >= 0.90 {
        0.15 // Search 15% of clusters for good recall
    } else if target_recall >= 0.80 {
        0.10 // Search 10% of clusters for balanced performance
    } else {
        0.05 // Search 5% of clusters for fast search
    };
    
    let optimal = (nlist as f64 * fraction) as usize;
    optimal.max(1).min(nlist)
}

/// Calculate optimal PQ parameters based on dimension and memory constraints
pub fn calculate_optimal_pq_params(dimension: usize, target_compression: f64) -> (usize, usize) {
    // m: number of subquantizers (should divide dimension evenly)
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

    // nbits: bits per subquantizer (affects compression vs accuracy)
    let nbits = if target_compression >= 0.95 {
        4 // High compression, lower accuracy
    } else if target_compression >= 0.90 {
        6 // Balanced compression and accuracy
    } else {
        8 // Lower compression, higher accuracy
    };

    (m, nbits)
}

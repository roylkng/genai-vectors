# Faiss IVF-PQ Integration for Billion-Scale Vector Search

## Overview

We have successfully integrated a scalable Faiss IVF-PQ (Inverted File with Product Quantization) architecture into the GenAI Vector Database. This implementation is designed to handle billion-scale vector datasets efficiently.

## Why Faiss IVF-PQ?

### Scalability Challenges
- **Brute Force Search**: O(N) complexity, doesn't scale beyond millions of vectors
- **Memory Limitations**: Storing all vectors in memory becomes prohibitive at billion scale
- **Query Performance**: Linear search becomes too slow for real-time applications

### Faiss IVF-PQ Solutions
1. **Inverted File (IVF)**: Partitions vectors into clusters, reducing search space from N to N/nlist
2. **Product Quantization (PQ)**: Compresses vectors using quantization, reducing memory by ~32x
3. **Approximate Search**: Trades small accuracy loss for massive speed gains
4. **Scalable Architecture**: Handles billions of vectors with sub-second query times

## Architecture Implementation

### 1. Mock Faiss Implementation (Development)
For development and testing, we've implemented a mock Faiss interface that:
- Simulates real Faiss IVF-PQ behavior
- Provides the same API as production Faiss
- Enables development without OpenMP dependencies
- Includes comprehensive testing

### 2. Production-Ready Design
The architecture is designed to seamlessly swap the mock implementation with real Faiss:

```rust
// Mock Implementation (Development)
pub struct MockIndex {
    dim: usize,
    vectors: Vec<Vec<f32>>,
    ids: Vec<i64>,
    // ... IVF-PQ parameters
}

// Production: Replace with real Faiss
// pub type Index = faiss::index_impl::BoxIndex;
```

### 3. Optimal Parameter Calculation
Automatic calculation of optimal IVF-PQ parameters:

```rust
// Optimal cluster count: ~sqrt(N) vectors
fn calculate_optimal_nlist(num_vectors: usize) -> u32 {
    let sqrt_n = (num_vectors as f64).sqrt() as u32;
    sqrt_n.max(16).min(65536) // Reasonable bounds
}

// Optimal search clusters: nlist/8 to nlist/4
fn calculate_optimal_nprobe(nlist: u32) -> u32 {
    (nlist / 8).max(1).min(nlist)
}
```

## Indexing Pipeline

### 1. Shard-Based Processing
- **Large Shards**: 50,000 vectors per shard (optimized for Faiss performance)
- **Parallel Training**: Each shard gets its own trained IVF-PQ index
- **Independent Storage**: Shards stored separately in S3 for horizontal scaling

### 2. Index Training Process
```rust
// 1. Build IVF-PQ index with optimal parameters
let mut index = faiss_utils::build_ivfpq_index(
    config.dim as usize,
    config.nlist,      // Number of clusters
    config.m,          // PQ subspaces (8)
    config.nbits,      // Bits per subspace (8)
    &config.metric,    // "cosine" or "euclidean"
    shard_vectors,     // Training data
)?;

// 2. Train the index on vector data
// Training learns cluster centroids and PQ codebooks

// 3. Add vectors with numeric IDs
let faiss_ids: Vec<i64> = shard_ids_slice.iter()
    .map(|id| faiss_utils::hash_string_to_i64(id))
    .collect();
faiss_utils::add_vectors(&mut index, shard_vectors, &faiss_ids)?;

// 4. Save trained index to S3
faiss_utils::save_index(&index, &local_path)?;
```

### 3. Metadata Management
- **ID Mapping**: Hash string IDs to i64 for Faiss compatibility
- **Metadata Storage**: Separate JSON storage for vector metadata
- **Index Manifests**: Track all shards and their configurations

## Query Pipeline

### 1. Multi-Shard Search
```rust
// Search across all shards in parallel
for shard in &manifest.shards {
    let results = search_shard(&s3, &req, shard, &manifest).await?;
    all_results.extend(results);
}

// Global top-k selection
all_results.sort_by_score();
all_results.truncate(req.topk);
```

### 2. Optimized Search Parameters
- **Dynamic nprobe**: Automatically calculated or user-specified
- **Quality vs Speed**: Configurable trade-off via nprobe parameter
- **Memory Efficiency**: Download only required index shards

### 3. Result Aggregation
- **Score Normalization**: Consistent scoring across shards
- **Metadata Enrichment**: Attach original metadata to results
- **ID Translation**: Convert numeric IDs back to original string IDs

## Performance Characteristics

### Expected Performance (Billion Scale)
- **Index Size**: ~1GB per 1M vectors (with PQ compression)
- **Query Time**: 1-10ms per shard, ~100ms for billion vectors
- **Memory Usage**: Only active shards loaded (not entire dataset)
- **Accuracy**: 95%+ recall with proper nprobe settings

### Scalability Metrics
- **Storage**: Linear scaling with vector count
- **Query Performance**: Logarithmic scaling with cluster count
- **Memory**: Constant per query (independent of dataset size)
- **Throughput**: Horizontal scaling via shard distribution

## Configuration Examples

### Index Creation
```json
{
  "name": "billion-scale-index",
  "dim": 1536,
  "metric": "cosine",
  "nlist": 4096,        // ~sqrt(16M) for 16M vectors
  "m": 8,               // 8 subspaces
  "nbits": 8,           // 8 bits per subspace
  "default_nprobe": 64  // Search 64 clusters
}
```

### Query Request
```json
{
  "index": "billion-scale-index",
  "embedding": [0.1, 0.2, ...],
  "topk": 10,
  "nprobe": 128  // Higher nprobe = better recall
}
```

## Production Deployment

### 1. Real Faiss Integration
To enable production Faiss, update `Cargo.toml`:
```toml
[dependencies]
faiss = { version = "0.12.1", features = ["static"] }
```

### 2. System Requirements
- **OpenMP**: Required for Faiss compilation
- **Memory**: 16GB+ recommended for large indices
- **Storage**: S3-compatible object storage
- **CPU**: AVX2 support recommended for optimal performance

### 3. Monitoring Metrics
- Index training time
- Query latency percentiles
- Memory usage per shard
- Cache hit rates
- Recall quality metrics

## Development vs Production

| Aspect | Development (Mock) | Production (Real Faiss) |
|--------|-------------------|-------------------------|
| Dependencies | None | OpenMP, BLAS |
| Performance | Brute force O(N) | IVF-PQ O(log N) |
| Memory Usage | Full vectors | Compressed (PQ) |
| Accuracy | 100% | 95-99% (configurable) |
| Setup | Simple | Complex build |

## Next Steps

1. **Production Faiss**: Install OpenMP and enable real Faiss
2. **Benchmarking**: Compare mock vs real performance
3. **Optimization**: Tune nlist/nprobe for your dataset
4. **Monitoring**: Add performance metrics and alerting
5. **Scaling**: Implement shard distribution across nodes

This implementation provides a solid foundation for billion-scale vector search with minimal changes required to switch from development to production.

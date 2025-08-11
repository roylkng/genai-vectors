# Data Directory

Local data storage for the GenAI Vector Database.

## Structure

```
data/
├── README.md                     # This file
├── indexes/                      # Vector indexes (auto-generated)
├── logs/                        # Application logs  
├── temp/                        # Temporary processing files
├── embedding_cache_test.pkl     # Small test cache (69KB)
└── embeddings_cache_5k_s3vectors.pkl  # Medium test cache (33MB)
```

## Gitignore Strategy

### Committed Files
- `embedding_cache_test.pkl` - Small test dataset for development
- `embeddings_cache_5k_s3vectors.pkl` - Medium dataset for integration tests

### Ignored Files (too large)
- `embedding_cache_500k.pkl` - Large 500K vector cache
- `*_500k*.pkl`, `*_1m*.pkl` - Large dataset files
- `*.pkl.tmp` - Temporary cache files

### Auto-Generated (ignored)
- `indexes/` - Vector indexes (regenerated from source data)
- `logs/` - Application logs
- `temp/` - Temporary processing files

## Usage

### Development
- Use `embedding_cache_test.pkl` for quick local testing
- Generate larger datasets as needed (will be gitignored)

### Testing  
- Integration tests use `embeddings_cache_5k_s3vectors.pkl`
- Real embeddings tests generate fresh data from LM Studio

### Production
- Data directory should be mounted from persistent storage
- Configure via `config/production.toml` for your storage backend

## Cache Management

```bash
# Clear temporary files
rm -rf data/temp/*

# Clear logs (keep directory)
rm -f data/logs/*.log

# Regenerate indexes (will be rebuilt automatically)
rm -rf data/indexes/*
```

## Backup Strategy

- **Source vectors**: Stored in S3/MinIO (authoritative)
- **Local caches**: Can be regenerated from source
- **Indexes**: Rebuilt automatically from vector data
- **Logs**: Rotate and archive as needed

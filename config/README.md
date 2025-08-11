# Configuration

Environment-specific configuration files for the GenAI Vector Database.

## Configuration Files

### `development.toml`
- Development environment configuration
- Debug logging enabled
- Local MinIO settings

### `production.toml` 
- Production environment configuration
- Optimized for performance and reliability
- AWS S3 settings

### `local.toml` (gitignored)
- Local developer overrides
- Personal API keys and endpoints
- Auto-generated from environment

## Configuration Structure

```toml
[server]
host = "0.0.0.0"
port = 8080
log_level = "info"

[storage]
bucket_name = "vectors"
region = "us-east-1" 
endpoint = "http://localhost:9000"  # MinIO for local dev

[vector]
default_dimension = 768
default_batch_size = 1000
default_top_k = 10
default_metric = "cosine"

[cache]
vector_cache_dir = "./data"
enable_cache = true
cache_size_mb = 1024

[indexing]
shard_size = 10000
max_shards_per_index = 100
enable_background_indexing = true
```

## Environment Variables

The system also supports environment variable overrides:

- `AWS_ACCESS_KEY_ID` - S3/MinIO access key
- `AWS_SECRET_ACCESS_KEY` - S3/MinIO secret key  
- `AWS_ENDPOINT_URL` - S3/MinIO endpoint URL
- `VEC_BUCKET` - Default bucket name
- `RUST_LOG` - Logging level

## Usage

The configuration is automatically loaded based on the environment:

```bash
# Development (uses development.toml)
cargo run api

# Production (uses production.toml)  
RUST_ENV=production cargo run --release api

# With local overrides (local.toml takes precedence)
cargo run api  # local.toml settings applied
```

# GenAI Vector Database

A production-grade vector database built with Rust, designed for scalable similarity search using S3 as storage backend.

## ✨ Features

- **🚀 High Performance**: Built with Rust for maximum speed and efficiency
- **📈 Scalable Storage**: Uses AWS S3 for distributed vector storage
- **🎯 Multiple Metrics**: Supports cosine similarity and Euclidean distance
- **⚡ Batch Operations**: Efficient batch insert/update operations
- **🔍 Metadata Filtering**: Rich metadata-based filtering capabilities
- **🐳 Production Ready**: Docker support with comprehensive testing
- **💾 Smart Caching**: Vector caching for improved performance

## 📁 Project Structure

```
genai-vectors/
├── src/                    # Rust source code
│   ├── main.rs            # Application entry point
│   ├── lib.rs             # Library root
│   ├── model.rs           # Data models and types
│   ├── minio.rs           # S3/MinIO client implementation
│   ├── query.rs           # Query processing and search
│   ├── api.rs             # REST API endpoints
│   ├── indexer.rs         # Index management
│   └── ingest.rs          # Data ingestion
├── tests/                 # Integration tests
│   ├── conftest.py        # Test configuration
│   ├── test_small_scale.py    # Small-scale tests (~1K vectors)
│   └── test_large_scale.py    # Large-scale tests (~500K vectors)
├── config/                # Configuration files
│   ├── development.toml   # Development settings
│   ├── production.toml    # Production settings
│   └── local.toml         # Local overrides (git-ignored)
├── data/                  # Data and cache directory
│   ├── embedding_cache_*.pkl  # Cached vectors (preserved)
│   ├── indexes/           # Index storage
│   └── logs/              # Log files
├── scripts/               # Utility scripts
│   └── setup.sh          # Environment setup
├── docs/                  # Documentation
├── charts/                # Helm charts
├── k8s/                   # Kubernetes manifests
├── Cargo.toml            # Rust dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Local development setup
└── requirements-test.txt  # Python test dependencies
```

## 🚀 Quick Start

### Prerequisites
- Rust 1.75+
- Python 3.8+ (for tests)
- AWS CLI configured
- Docker (optional)

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd genai-vectors
./scripts/setup.sh
```

2. **Configure credentials** (choose one method):

**Option A: Environment Variables (Recommended)**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

**Option B: .env File**
```bash
# Copy example and edit
cp .env.example .env
# Edit .env with your credentials
```

**Option C: Configuration File**
```bash
# Edit config/local.toml
[aws]
access_key_id = "your_access_key"
secret_access_key = "your_secret_key"

[storage]
region = "us-east-1"
bucket_name = "my-vectors"
```

3. **Test configuration**:
```bash
# Test configuration loading
source .venv/bin/activate
python tests/constants.py
```

### Running the Application

**Option 1: Cargo (Development)**
```bash
cargo run
```

**Option 2: Docker Compose (Full Stack)**
```bash
docker-compose up -d
```

**Option 3: Docker (Production)**
```bash
docker build -t genai-vectors .
docker run -p 8080:8080 genai-vectors
```

## 🧪 Testing

The project includes comprehensive tests using cached vectors for performance:

### Small-Scale Tests (~1K vectors)
```bash
python -m pytest tests/test_small_scale.py -v
```

### Large-Scale Tests (~500K vectors)
```bash
python -m pytest tests/test_large_scale.py -v
```

### All Tests
```bash
# Fast tests only
python -m pytest tests/ -v -m "not slow"

# All tests including slow ones
python -m pytest tests/ -v
```

### Test Configuration
Tests use cached vectors stored in `data/` directory:
- `embedding_cache_test.pkl` - Small test dataset
- `embeddings_cache_5k_s3vectors.pkl` - 5K vectors for medium tests
- `embedding_cache_500k.pkl` - 500K vectors for performance tests

## 📊 API Usage

### Insert Vectors
```bash
curl -X POST http://localhost:8080/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "index": "my_index",
    "vectors": [
      {
        "id": "vec1",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {"category": "A", "value": 42}
      }
    ]
  }'
```

### Search Vectors
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "index": "my_index",
    "embedding": [0.1, 0.2, 0.3],
    "topk": 10,
    "filters": {"category": "A"}
  }'
```

## ⚙️ Configuration

The project uses a flexible configuration system that supports:

1. **Environment Variables** (highest priority)
2. **Configuration Files** (.env, config/*.toml)
3. **Default Values** (lowest priority)

### Configuration Priority

```bash
Environment Variables > .env file > config/local.toml > config/development.toml > defaults
```

### Configuration Files

**TOML Configuration** (`config/local.toml`):
```toml
[aws]
access_key_id = "your_access_key"
secret_access_key = "your_secret_key"

[server]
host = "0.0.0.0"
port = 8080
log_level = "info"

[storage]
bucket_name = "my-vectors"
region = "us-east-1"
# endpoint = "http://localhost:9000"  # For MinIO

[vector]
default_dimension = 256
default_batch_size = 1000
default_metric = "cosine"

[cache]
vector_cache_dir = "./data"
enable_cache = true
cache_size_mb = 1024
```

**Environment Variables**:
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
AWS_ENDPOINT_URL=http://localhost:9000  # For MinIO

# Server Configuration  
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
LOG_LEVEL=info

# Storage Configuration
STORAGE_BUCKET_NAME=my-vectors
STORAGE_REGION=us-east-1

# Override any config with: SECTION_KEY format
VECTOR_DEFAULT_DIMENSION=512
CACHE_ENABLE_CACHE=false
```

### Configuration Validation

Test your configuration:
```bash
source .venv/bin/activate
python tests/constants.py
```

## 🐳 Deployment

### Kubernetes (Recommended)
```bash
# Using Helm
helm install genai-vectors ./charts/vector-store

# Using kubectl
kubectl apply -f k8s/
```

### Docker Swarm
```bash
docker stack deploy -c docker-compose.yml genai-vectors
```

## 📈 Performance

- **Ingestion**: Up to 10K vectors/second
- **Query**: Sub-100ms for 1M+ vectors
- **Storage**: Efficient sharding and compression
- **Scalability**: Horizontal scaling via S3 sharding

## 🛠️ Development

### Building
```bash
cargo build --release
```

### Running Tests
```bash
# Rust tests
cargo test

# Integration tests
python -m pytest tests/ -v

# Benchmarks
cargo bench
```

### Code Quality
```bash
# Format
cargo fmt

# Lint
cargo clippy

# Python formatting
black tests/
flake8 tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔍 Troubleshooting

### Common Issues

**AWS Credentials**
```bash
# Check credentials
aws sts get-caller-identity

# Set region
export AWS_REGION=us-east-1
```

**Docker Issues**
```bash
# Clean build
docker-compose down -v
docker-compose up --build
```

**Test Failures**
```bash
# Check cached vectors
ls -la data/

# Regenerate if needed
python -c "import tests.conftest; print('Cache loaded')"
```

For more help, please check the [docs/](docs/) directory or open an issue.

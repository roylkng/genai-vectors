# GenAI Vector Database

A production-grade vector database built with Rust, featuring **100% AWS S3 Vectors API compatibility** for seamless integration with existing applications.

## ✨ Key Features

- **🎯 100% S3 Vectors API Compatible**: All 13 AWS S3 Vectors operations supported
- **🚀 High Performance**: Built with Rust + FAISS for maximum speed
- **📈 Scalable Storage**: Uses S3/MinIO for distributed storage
- **⚡ Production Ready**: Comprehensive testing and monitoring
- **🐳 Easy Deployment**: Docker + Kubernetes support

## 🏆 API Compatibility Status

**✅ All 13 S3 Vectors Operations Working (100%)**

| Operation | Status | Description |
|-----------|--------|-------------|
| create-vector-bucket | ✅ | Create vector storage bucket |
| list-vector-buckets | ✅ | List all vector buckets |
| get-vector-bucket | ✅ | Get bucket information |
| delete-vector-bucket | ✅ | Delete vector bucket |
| create-index | ✅ | Create vector index |
| list-indexes | ✅ | List indexes in bucket |
| get-index | ✅ | Get index information |
| delete-index | ✅ | Delete vector index |
| put-vectors | ✅ | Insert/update vectors |
| list-vectors | ✅ | List vectors in index |
| get-vectors | ✅ | Retrieve specific vectors |
| delete-vectors | ✅ | Delete vectors |
| query-vectors | ✅ | Similarity search |

## � Project Structure

```
genai-vectors/
├── src/                          # Rust source code
│   ├── api.rs                   # AWS S3 Vectors API implementation
│   ├── faiss_utils.rs          # FAISS integration utilities
│   ├── ingest.rs               # Vector ingestion pipeline
│   ├── query.rs                # Vector similarity search
│   ├── minio.rs                # S3/MinIO client
│   └── ...
├── tests/                       # Comprehensive test suite
│   ├── integration/            # Integration tests
│   │   ├── test_real_embeddings_s3.py    # Real LM Studio integration
│   │   ├── test_s3_compatibility.py      # S3 API compatibility
│   │   └── run_real_embeddings_test.sh   # Test automation
│   ├── test_small_scale.py    # Unit tests
│   └── test_large_scale.py    # Performance tests
├── scripts/                    # Utility scripts
├── config/                     # Configuration files
├── charts/                     # Helm charts for Kubernetes
├── docs/                      # Documentation
└── Makefile                   # Build and deployment commands
```

## 🧪 Testing

### Run Full Integration Test (with LM Studio)
```bash
cd tests/integration
./run_real_embeddings_test.sh
```

### Run S3 API Compatibility Test
```bash
python tests/integration/test_s3_compatibility.py
```

Expected result: **13/13 commands passed (100.0%)**

### 4. Run the Service
```bash
AWS_ENDPOINT_URL=http://localhost:9000 
AWS_ACCESS_KEY_ID=minioadmin 
AWS_SECRET_ACCESS_KEY=minioadmin 
./target/release/genai-vectors api
```

### 5. Test S3 API Compatibility
```bash
# Using Python/boto3 (application integration)
python test_s3_compatibility.py

# Using AWS CLI (manual validation)
./test_aws_cli.sh

# Using curl (direct HTTP)
./test_s3_api.sh
```

Expected output (any test method):
```
🏆 Overall: 13/13 commands passed (100.0%)
🎉 All S3 vectors API commands working perfectly!
```

## 📊 Usage Examples

### Using boto3 Client
```python
import boto3

# Create S3 vectors client
s3_vectors = boto3.client(
    's3vectors',
    endpoint_url='http://localhost:8080',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    region_name='us-east-1'
)

# Create bucket and index
s3_vectors.create_vector_bucket(vectorBucketName='my-vectors')
s3_vectors.create_index(
    vectorBucketName='my-vectors',
    indexName='embeddings',
    dimension=1536,
    distanceMetric='COSINE'
)

# Insert vectors
s3_vectors.put_vectors(
    vectorBucketName='my-vectors',
    indexName='embeddings',
    vectors=[{
        'key': 'doc1',
        'data': {'float32': [0.1, 0.2, ...]},
        'metadata': {'category': 'document'}
    }]
)

# Search similar vectors
results = s3_vectors.query_vectors(
    vectorBucketName='my-vectors',
    indexName='embeddings',
    queryVector={'float32': [0.1, 0.2, ...]},
    topK=10
)
```

### Direct REST API
```bash
# Create index
curl -X POST "http://localhost:8080/s3-vectors/CreateIndex" 
  -H "Content-Type: application/json" 
  -d '{
    "vectorBucketName": "my-vectors",
    "indexName": "embeddings",
    "dimension": 1536,
    "distanceMetric": "COSINE"
  }'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   boto3 Client  │───▶│  Vector API     │───▶│   FAISS Engine  │
│   (S3 Vectors)  │    │  (Rust Service) │    │   (IVF-PQ)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   MinIO/S3      │
                       │   (Storage)     │
                       └─────────────────┘
```

**Core Components:**
- **API Layer**: Complete S3 Vectors API implementation
- **Vector Engine**: FAISS-based similarity search
- **Storage**: MinIO/S3 for persistent vector storage
- **Format**: Parquet for efficient serialization

## 🐳 Deployment

### Docker
```bash
docker build -t genai-vectors .
docker run -p 8080:8080 
  -e AWS_ENDPOINT_URL=http://minio:9000 
  -e AWS_ACCESS_KEY_ID=minioadmin 
  -e AWS_SECRET_ACCESS_KEY=minioadmin 
  genai-vectors
```

### Kubernetes
```bash
helm install genai-vectors ./charts/vector-store
```

## 📈 Performance

- **Throughput**: 10K+ vectors/second ingestion
- **Latency**: <100ms similarity search
- **Scalability**: Horizontal scaling via S3 sharding
- **Memory**: Efficient PQ compression

## � Documentation

- **[Production Guide](PRODUCTION_GUIDE.md)**: Comprehensive deployment guide
- **[API Reference](https://docs.aws.amazon.com/s3vectors/)**: AWS S3 Vectors API docs
- **[Test Suite](test_s3_compatibility.py)**: Complete compatibility validation

## 🧪 Testing

### Automated Testing Scripts
```bash
# Test with Python/boto3 (recommended for application integration)
python test_s3_compatibility.py

# Test with AWS CLI (recommended for manual validation)
./test_aws_cli.sh

# Test with curl (alternative for debugging)
./test_s3_api.sh
```

### Additional Tests
```bash
# Rust unit tests
cargo test

# Integration tests
cd tests && python -m pytest -v
```

**All scripts validate 13/13 S3 vectors operations with 100% compatibility.**

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ENDPOINT_URL` | Yes | - | MinIO/S3 endpoint |
| `AWS_ACCESS_KEY_ID` | Yes | - | Access credentials |
| `AWS_SECRET_ACCESS_KEY` | Yes | - | Secret credentials |
| `AWS_REGION` | No | `us-east-1` | AWS region |
| `SERVER_PORT` | No | `8080` | API server port |
| `LOG_LEVEL` | No | `info` | Logging level |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_s3_compatibility.py`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Status**: Production Ready ✅  
**API Compatibility**: 100% (13/13 operations)  
**Last Updated**: August 2025

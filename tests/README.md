# Tests

This directory contains comprehensive tests for the GenAI Vectors system.

## Structure

```
tests/
├── README.md                     # This file
├── integration/                  # Integration tests
│   ├── test_real_embeddings_s3.py       # End-to-end test with real LM Studio embeddings
│   ├── test_s3_compatibility.py         # AWS S3 Vectors API compatibility test
│   └── run_real_embeddings_test.sh      # Test setup and execution script
├── conftest.py                   # Pytest configuration
├── constants.py                  # Test constants and configuration
├── sample.json                   # Sample test data
├── test_large_scale.py          # Large-scale performance tests
├── test_small_scale.py          # Small-scale unit tests
└── e2e.rs                       # Rust end-to-end tests
```

## Integration Tests

### Real Embeddings Test (`test_real_embeddings_s3.py`)
- **Purpose**: End-to-end test using real embeddings from LM Studio
- **Requirements**: 
  - LM Studio running at `http://127.0.0.1:1234`
  - `text-embedding-nomic-embed-text-v1.5` model loaded
  - Vector service running at `http://localhost:8080`
  - MinIO/S3 storage accessible
- **What it tests**:
  - Real embedding generation (768-dimensional vectors)
  - Complete AWS S3 Vectors API workflow
  - Document indexing and similarity search
  - Metadata filtering and vector retrieval

### S3 Compatibility Test (`test_s3_compatibility.py`)
- **Purpose**: Validates 100% AWS S3 Vectors API compatibility
- **Tests**: All 13 AWS S3 Vectors operations
- **Expected Result**: 13/13 commands passed (100.0%)

## Running Tests

### Prerequisites
1. Start the vector service:
   ```bash
   export AWS_ACCESS_KEY_ID=minioadmin
   export AWS_SECRET_ACCESS_KEY=minioadmin  
   export AWS_ENDPOINT_URL=http://localhost:9000
   export VEC_BUCKET=vectors
   cargo run --release api
   ```

2. Start MinIO (if using Kubernetes):
   ```bash
   kubectl port-forward -n genai-vectors service/minio 9000:9000
   ```

3. Start LM Studio (for real embeddings test):
   - Load `text-embedding-nomic-embed-text-v1.5` model
   - Ensure it's accessible at `http://127.0.0.1:1234`

### Run Integration Tests

```bash
# Run the comprehensive setup and test script
cd tests/integration
./run_real_embeddings_test.sh

# Or run individual tests
python test_real_embeddings_s3.py
python test_s3_compatibility.py
```

### Run Unit Tests

```bash
# Python tests
pytest test_small_scale.py test_large_scale.py

# Rust tests  
cargo test
cargo test e2e --test e2e
```

## Test Results Interpretation

### Real Embeddings Test
- **100% Success**: All pipeline components working
- **Partial Success (80-99%)**: Minor issues, check similarity search results
- **<80% Success**: Major issues requiring investigation

### S3 Compatibility Test
- **13/13 Passed**: Full AWS S3 Vectors API compatibility
- **10-12/13 Passed**: Minor compatibility issues
- **<10/13 Passed**: Major compatibility problems

## Development Workflow

1. **Before committing**: Run S3 compatibility test
2. **Before releasing**: Run full real embeddings test
3. **Performance testing**: Use large-scale tests
4. **CI/CD**: All tests should pass in pipeline

## Troubleshooting

### Common Issues
- **Port conflicts**: Ensure ports 8080, 9000, 1234 are available
- **MinIO connection**: Verify Kubernetes port-forward is active
- **LM Studio**: Check model is loaded and API is responsive
- **Environment variables**: Ensure AWS credentials are set

### Debug Commands
```bash
# Check service status
curl http://localhost:8080/CreateVectorBucket -X POST -H "Content-Type: application/json" -d '{"vectorBucketName":"test"}'

# Check LM Studio
curl http://127.0.0.1:1234/v1/models

# Check MinIO
curl http://localhost:9000/minio/health/live
```

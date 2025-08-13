# Documentation

Comprehensive documentation for the GenAI Vector Database.

## Documents

### `aws-vector-test-coverage.md`
- Test coverage analysis for AWS S3 Vectors API compatibility
- Details which operations are tested and validated

### `GITIGNORE_STRATEGY.md`
- Strategy for managing data files and cache in git
- Guidelines for what should and shouldn't be committed

## Additional Documentation

For more specific documentation, see:

- [`/tests/README.md`](../tests/README.md) - Testing guide
- [`/scripts/README.md`](../scripts/README.md) - Scripts documentation  
- [`/config/README.md`](../config/README.md) - Configuration guide
- [`/charts/README.md`](../charts/README.md) - Kubernetes deployment
- [`PRODUCTION_GUIDE.md`](../PRODUCTION_GUIDE.md) - Production deployment guide

## API Documentation

The system implements 100% AWS S3 Vectors API compatibility. See the main README for:
- Complete API operation list
- Usage examples
- Integration patterns

## Architecture

### Core Components
- **API Layer** (`src/api.rs`) - AWS S3 Vectors API implementation
- **Storage Layer** (`src/minio.rs`) - S3/MinIO integration
- **Vector Engine** (`src/faiss_utils.rs`) - FAISS-based similarity search
- **Ingestion Pipeline** (`src/ingest.rs`) - Vector processing and indexing
- **Query Engine** (`src/query.rs`) - High-performance vector search

### Data Flow
1. Vectors submitted via S3 Vectors API
2. Processed through ingestion pipeline
3. Indexed using FAISS for fast similarity search
4. Stored in S3/MinIO for persistence
5. Queried through optimized search algorithms

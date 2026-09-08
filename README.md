# GenAI Vectors

An experimental vector-search service in Rust that explores an **S3 Vectors-style API over FAISS-backed indexes and object storage**.

The project is deliberately infrastructure-oriented: clients talk to a familiar bucket/index/vector API while the service owns indexing, metadata, persistence, and query execution behind that boundary.

> **Status:** research / prototype. The repository contains a compatibility harness covering the 13 S3 Vectors operations implemented here. It is not presented as a drop-in production replacement for AWS S3 Vectors, and no performance claim is made without a reproducible benchmark.

## Why this exists

Vector databases usually couple their API, storage engine, and persistence model. This project explores a different split:

```text
boto3 / REST client
        │
        ▼
S3 Vectors-style API
        │
        ├── bucket + index metadata
        ├── vector ingestion
        ├── metadata filtering
        └── similarity queries
        │
        ▼
FAISS indexes  +  Parquet slices  +  S3 / MinIO
```

The interesting engineering questions are less about wrapping FAISS and more about preserving a useful API contract while keeping vector data, index artifacts, and object storage independently replaceable.

## Technical highlights

- **Rust + Axum** service boundary.
- **FAISS** similarity search with IVF-PQ and HNSW-oriented indexing paths.
- **Parquet** storage for vector slices.
- **S3 / MinIO** persistence through the AWS Rust SDK.
- **boto3-oriented compatibility tests** for bucket, index, vector CRUD, and query operations.
- **Metadata filtering** support in the query path.
- Docker, Kubernetes, Helm, and local integration-test workflows are included in the repository.

## Implemented API surface

The compatibility harness exercises these 13 operations:

| Bucket | Index | Vector |
| --- | --- | --- |
| `create-vector-bucket` | `create-index` | `put-vectors` |
| `list-vector-buckets` | `list-indexes` | `list-vectors` |
| `get-vector-bucket` | `get-index` | `get-vectors` |
| `delete-vector-bucket` | `delete-index` | `delete-vectors` |
|  |  | `query-vectors` |

Run the compatibility harness against a running service rather than treating this table as an external certification:

```bash
python tests/integration/test_s3_compatibility.py
```

## Repository map

```text
genai-vectors/
├── src/                 # Rust API, ingestion, query, indexing and storage code
├── tests/               # unit, scale and integration tests
├── scripts/             # setup and operational helpers
├── charts/              # Helm deployment assets
├── config/              # configuration
├── Dockerfile
├── Makefile
└── PRODUCTION_GUIDE.md  # deployment-oriented notes from the prototype
```

## Local development

Prerequisites depend on how FAISS is installed on your system. Once the Rust dependencies and FAISS toolchain are available:

```bash
cargo build --release
cargo test
```

For an S3-compatible local backend, configure MinIO or another endpoint:

```bash
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
```

Start the service using the CLI exposed by the current build, then run:

```bash
python tests/integration/test_s3_compatibility.py
```

The integration directory also contains a real-embedding workflow that can be exercised with a local embedding endpoint.

## Example client

```python
import boto3

client = boto3.client(
    "s3vectors",
    endpoint_url="http://localhost:8080",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    region_name="us-east-1",
)

client.create_vector_bucket(vectorBucketName="demo")
client.create_index(
    vectorBucketName="demo",
    indexName="embeddings",
    dimension=1536,
    distanceMetric="COSINE",
)

client.put_vectors(
    vectorBucketName="demo",
    indexName="embeddings",
    vectors=[{
        "key": "doc-1",
        "data": {"float32": [0.1, 0.2, 0.3]},
        "metadata": {"kind": "document"},
    }],
)
```

## What I would validate before production use

This repository is useful as a systems prototype, but a production deployment would still need evidence for:

- durability and recovery semantics under partial failures;
- concurrency and consistency guarantees;
- large-scale recall / latency benchmarks on representative data;
- index rebuild and compaction behavior;
- authentication, authorization, tenancy and abuse controls;
- compatibility against the exact AWS API version a client depends on;
- sustained upgrade and dependency-management policy.

That boundary is intentional. A credible infrastructure project should make the unproven parts visible.

## License

MIT. See `LICENSE`.
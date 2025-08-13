use axum::{Router, routing::{post, get, put, delete}, extract::{State, Path, Query}, Json, serve, response::{IntoResponse, Response}, http::StatusCode};
use crate::{model::*, ingest::Ingestor, minio::S3Client};
use std::sync::Arc;
use tokio::net::TcpListener;
use serde::{Deserialize};
use anyhow::Context;
use serde_json::json;

mod buckets;
mod vectors;
mod indices;

// Standard S3 API handlers for boto3 compatibility

/// Handle GET / - List all buckets (S3 ListBuckets operation)
async fn s3_list_buckets(State(state): State<AppState>) -> impl IntoResponse {
    buckets::list(state).await
}

/// Handle PUT / - Create bucket from root with bucket name in request body
async fn s3_create_bucket_root(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket_name = payload.get("Bucket")
        .or_else(|| payload.get("bucket"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    buckets::create(bucket_name, state).await
}

/// Handle PUT /:bucket - Create bucket (S3 CreateBucket operation)
async fn s3_create_bucket(
    Path(bucket): Path<String>,
    State(state): State<AppState>
) -> impl IntoResponse {
    buckets::create(bucket, state).await
}

/// Handle GET /:bucket - Get bucket info (S3 HeadBucket operation)
async fn s3_get_bucket(
    Path(bucket): Path<String>,
    State(state): State<AppState>
) -> impl IntoResponse {
    buckets::get(bucket, state).await
}

/// Handle DELETE /:bucket - Delete bucket (S3 DeleteBucket operation)
async fn s3_delete_bucket(
    Path(bucket): Path<String>,
    State(state): State<AppState>
) -> impl IntoResponse {
    buckets::delete(bucket, state).await
}

// S3 Vectors API compatibility structures
#[derive(Deserialize, Debug)]
struct S3VectorBucketQuery {
    #[serde(rename = "create-vector-bucket")]
    create_vector_bucket: Option<String>,
    #[serde(rename = "list-vector-buckets")]
    list_vector_buckets: Option<String>,
    #[serde(rename = "get-vector-bucket")]
    get_vector_bucket: Option<String>,
    #[serde(rename = "delete-vector-bucket")]
    delete_vector_bucket: Option<String>,
    #[serde(rename = "create-index")]
    create_index: Option<String>,
    #[serde(rename = "list-indexes")]
    list_indexes: Option<String>,
    #[serde(rename = "get-index")]
    get_index: Option<String>,
    #[serde(rename = "delete-index")]
    delete_index: Option<String>,
    #[serde(rename = "put-vectors")]
    put_vectors: Option<String>,
    #[serde(rename = "list-vectors")]
    list_vectors: Option<String>,
    #[serde(rename = "get-vectors")]
    get_vectors: Option<String>,
    #[serde(rename = "delete-vectors")]
    delete_vectors: Option<String>,
    #[serde(rename = "query-vectors")]
    query_vectors: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct S3CreateIndexRequest {
    pub vector_bucket_name: String,
    pub index_name: String,
    pub data_type: String,
    pub dimension: u32,
    pub distance_metric: String,
    #[serde(default)]
    pub metadata_configuration: Option<MetadataConfiguration>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct MetadataConfiguration {
    #[serde(default)]
    pub non_filterable_metadata_keys: Vec<String>,
}

#[derive(Deserialize)]
pub struct S3PutVectorsRequest {
    #[serde(rename = "indexName")]
    pub index_name: Option<String>,
    #[serde(rename = "vectorBucketName")]
    pub vector_bucket_name: Option<String>,
    #[serde(rename = "indexArn")]
    pub index_arn: Option<String>,
    pub vectors: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct S3GetVectorsRequest {
    #[serde(rename = "indexName")]
    pub index_name: Option<String>,
    #[serde(rename = "vectorBucketName")]
    pub vector_bucket_name: Option<String>,
    #[serde(rename = "indexArn")]
    pub index_arn: Option<String>,
    pub keys: Vec<String>,
    #[serde(rename = "returnData", default)]
    pub return_data: bool,
    #[serde(rename = "returnMetadata", default)]
    pub return_metadata: bool,
}

#[derive(Deserialize)]
pub struct S3DeleteVectorsRequest {
    #[serde(rename = "indexName")]
    pub index_name: Option<String>,
    #[serde(rename = "vectorBucketName")]
    pub vector_bucket_name: Option<String>,
    #[serde(rename = "indexArn")]
    pub index_arn: Option<String>,
    pub keys: Vec<String>,
}

#[derive(Deserialize)]
pub struct S3ListVectorsRequest {
    #[serde(rename = "indexName")]
    pub index_name: Option<String>,
    #[serde(rename = "vectorBucketName")]
    pub vector_bucket_name: Option<String>,
    #[serde(rename = "indexArn")]
    pub index_arn: Option<String>,
    #[serde(rename = "maxResults")]
    pub max_results: Option<u32>,
    #[serde(rename = "nextToken")]
    pub next_token: Option<String>,
}

#[derive(Deserialize)]
pub struct S3QueryVectorsRequest {
    #[serde(rename = "indexName")]
    pub index_name: Option<String>,
    #[serde(rename = "vectorBucketName")]
    pub vector_bucket_name: Option<String>,
    #[serde(rename = "indexArn")]
    pub index_arn: Option<String>,
    pub vector: Vec<f64>,
    #[serde(rename = "topK")]
    pub top_k: Option<u32>,
    #[serde(rename = "returnData", default)]
    pub return_data: bool,
    #[serde(rename = "returnMetadata", default)]
    pub return_metadata: bool,
    #[serde(rename = "metadataFilter")]
    pub metadata_filter: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct AppState {
    pub s3: S3Client,
    pub ingest: Arc<Ingestor>,
}

// Handler for S3-style path-based operations (e.g., GET /:bucket_name?operation=value)
async fn s3_vectors_handler(
    Path(operation): Path<String>,
    Query(query): Query<S3VectorBucketQuery>,
    State(state): State<AppState>,
    body: String,
) -> Response {
    tracing::info!("S3 vectors handler - path: {}, query: {:?}", operation, query);
    
    let body = if body.is_empty() {
        serde_json::json!({})
    } else {
        match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(json) => json,
            Err(_) => serde_json::json!({})
        }
    };

    match operation.as_str() {
        "CreateVectorBucket" => {
            buckets::create(operation.clone(), state).await
        }
        "ListVectorBuckets" => {
            buckets::list(state).await
        }
        "GetVectorBucket" => {
            buckets::get(operation.clone(), state).await
        }
        "DeleteVectorBucket" => {
            buckets::delete(operation.clone(), state).await
        }
        "CreateIndex" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            indices::create(bucket_name.to_string(), body, state).await
        }
        "ListIndexes" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            indices::list(bucket_name.to_string(), state).await
        }
        "GetIndex" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            indices::get(bucket_name.to_string(), body, state).await
        }
        "DeleteIndex" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            indices::delete(bucket_name.to_string(), body, state).await
        }
        "PutVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            vectors::put(bucket_name.to_string(), body, state).await
        }
        "ListVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            vectors::list(bucket_name.to_string(), body, state).await
        }
        "GetVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            vectors::get(bucket_name.to_string(), body, state).await
        }
        "DeleteVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            vectors::delete(bucket_name.to_string(), body, state).await
        }
        "QueryVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            vectors::query(bucket_name.to_string(), body, state).await
        },
        // Fallback: legacy operations not supported
        _ => {
            tracing::warn!("Unknown S3 vectors operation - path: {}", operation);
            (StatusCode::BAD_REQUEST, format!("Invalid S3 vectors operation: {}", operation)).into_response()
        }
    }
}

// Handler for RPC-style calls with operation in the body
async fn s3_rpc_handler(
    State(state): State<AppState>,
    body: String,
) -> Response {
    tracing::info!("S3 RPC handler - body: {}", body);
    
    let body = if body.is_empty() {
        serde_json::json!({})
    } else {
        match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(json) => json,
            Err(e) => {
                tracing::error!("Failed to parse RPC body: {}", e);
                return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response();
            }
        }
    };
    
    // Check for operation in the body
    if let Some(operation) = body.get("operation").and_then(|v| v.as_str()) {
        match operation {
            "CreateVectorBucket" => {
                let bucket_name = body.get("bucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                buckets::create(bucket_name.to_string(), state).await
            }
            "ListVectorBuckets" => {
                buckets::list(state).await
            }
            "GetVectorBucket" => {
                let bucket_name = body.get("bucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                buckets::get(bucket_name.to_string(), state).await
            }
            "DeleteVectorBucket" => {
                let bucket_name = body.get("bucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                buckets::delete(bucket_name.to_string(), state).await
            }
            "CreateIndex" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                indices::create(bucket_name.to_string(), body, state).await
            }
            "ListIndexes" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                indices::list(bucket_name.to_string(), state).await
            }
            "GetIndex" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                indices::get(bucket_name.to_string(), body, state).await
            }
            "DeleteIndex" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                indices::delete(bucket_name.to_string(), body, state).await
            }
            "PutVectors" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                vectors::put(bucket_name.to_string(), body, state).await
            }
            "ListVectors" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                vectors::list(bucket_name.to_string(), body, state).await
            }
            "GetVectors" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                vectors::get(bucket_name.to_string(), body, state).await
            }
            "DeleteVectors" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                vectors::delete(bucket_name.to_string(), body, state).await
            }
            "QueryVectors" => {
                let bucket_name = body.get("vectorBucketName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default-bucket");
                vectors::query(bucket_name.to_string(), body, state).await
            }
            _ => {
                tracing::warn!("Unknown RPC operation: {}", operation);
                (StatusCode::BAD_REQUEST, format!("Unknown operation: {}", operation)).into_response()
            }
        }
    } else {
        tracing::warn!("Missing operation in RPC request");
        (StatusCode::BAD_REQUEST, "Missing operation field".to_string()).into_response()
    }
}

// Legacy API handlers for existing endpoints
async fn create_index(
    State(state): State<AppState>,
    Json(body): Json<CreateIndex>,
) -> impl IntoResponse {
    tracing::info!("Legacy create index request");
    
    // Convert to S3 format and call the new handler
    let s3_body = json!({
        "vectorBucketName": "vectors",
        "indexName": body.name,
        "dimension": body.dim,
        "distanceMetric": body.metric,
        "dataType": "FLOAT32"
    });
    
    indices::create("vectors".to_string(), s3_body, state).await
}

async fn put_vectors(
    State(state): State<AppState>,
    Json(body): Json<PutVectors>,
) -> impl IntoResponse {
    tracing::info!("Legacy put vectors request");
    
    // Convert to S3 format and call the new handler
    let s3_body = json!({
        "vectorBucketName": "vectors",
        "indexName": body.index,
        "vectors": body.vectors
    });
    
    vectors::put("vectors".to_string(), s3_body, state).await
}

async fn query(
    State(state): State<AppState>,
    Json(body): Json<QueryRequest>,
) -> impl IntoResponse {
    tracing::info!("Legacy query request");
    
    // Convert to S3 format and call the new handler
    let s3_body = json!({
        "vectorBucketName": "vectors",
        "indexName": body.index,
        "vector": body.embedding,
        "topK": body.topk,
        "returnData": true,
        "returnMetadata": true,
        "metadataFilter": body.filter
    });
    
    vectors::query("vectors".to_string(), s3_body, state).await
}

// GET /health - Health check
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "healthy"})).into_response()
}

pub async fn run() -> anyhow::Result<()> {
    let bucket = std::env::var("VEC_BUCKET").unwrap_or_else(|_| "vectors".to_string());
    let s3 = S3Client::from_env().await?;
    let ingest = Arc::new(Ingestor::new(s3.clone(), bucket));

    let state = AppState {
        s3,
        ingest,
    };

    let app = Router::new()
        // Health check
        .route("/health", get(health))
        // Standard S3 API endpoints for boto3 compatibility
        .route("/", get(s3_list_buckets))
        .route("/:bucket", put(s3_create_bucket).get(s3_get_bucket).delete(s3_delete_bucket))
        // Original API endpoints
        .route("/indexes", post(create_index))
        .route("/vectors", post(put_vectors))
        .route("/query", post(query))
        // S3 Vectors API compatibility endpoints - using the actual paths boto3 calls
        .route("/CreateVectorBucket", post(buckets::create_direct))
        .route("/ListVectorBuckets", post(buckets::list_direct))
        .route("/GetVectorBucket", post(buckets::get_direct))
        .route("/DeleteVectorBucket", post(buckets::delete_direct))
        .route("/CreateIndex", post(indices::create_direct))
        .route("/ListIndexes", post(indices::list_direct))
        .route("/GetIndex", post(indices::get_direct))
        .route("/DeleteIndex", post(indices::delete_direct))
        .route("/PutVectors", post(vectors::put_direct))
        .route("/ListVectors", post(vectors::list_direct))
        .route("/GetVectors", post(vectors::get_direct))
        .route("/DeleteVectors", post(vectors::delete_direct))
        .route("/QueryVectors", post(vectors::query_direct))
        // RPC and fallback handlers
        .route("/", post(s3_rpc_handler))
        .route("/:bucket", post(s3_vectors_handler)) // For path-based ops
        .with_state(state);

    let addr = "0.0.0.0:8081";
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("API listening on {addr}");
    serve(listener, app).await?;
    Ok(())
}

// Helper functions for metadata validation

async fn load_index_configuration(s3: &S3Client, index_name: &str) -> anyhow::Result<IndexConfiguration> {
    let config_key = format!("indexes/{}/config.json", index_name);
    
    let data = s3.get_object(&config_key).await
        .context("Failed to load index configuration")?;
    
    let create_index: CreateIndex = serde_json::from_slice(&data)
        .context("Failed to parse index configuration")?;
    
    Ok(IndexConfiguration {
        non_filterable_metadata_keys: create_index.non_filterable_metadata_keys,
    })
}

#[derive(Debug, Clone)]
struct IndexConfiguration {
    non_filterable_metadata_keys: Vec<String>,
}

fn validate_vector_metadata(metadata: &serde_json::Value, config: &IndexConfiguration) -> anyhow::Result<()> {
    if let serde_json::Value::Object(map) = metadata {
        let mut filterable_size = 0;
        let mut non_filterable_size = 0;
        
        for (key, value) in map {
            let value_size = calculate_metadata_value_size(value);
            
            if config.non_filterable_metadata_keys.contains(key) {
                non_filterable_size += key.len() + value_size;
            } else {
                filterable_size += key.len() + value_size;
            }
        }
        
        // Check size limits (2KB for filterable, 40KB for non-filterable)
        const FILTERABLE_LIMIT: usize = 2 * 1024; // 2KB
        const NON_FILTERABLE_LIMIT: usize = 40 * 1024; // 40KB
        
        if filterable_size > FILTERABLE_LIMIT {
            return Err(anyhow::anyhow!(
                "Filterable metadata size ({} bytes) exceeds limit of {} bytes", 
                filterable_size, FILTERABLE_LIMIT
            ));
        }
        
        if non_filterable_size > NON_FILTERABLE_LIMIT {
            return Err(anyhow::anyhow!(
                "Non-filterable metadata size ({} bytes) exceeds limit of {} bytes", 
                non_filterable_size, NON_FILTERABLE_LIMIT
            ));
        }
    }
    
    Ok(())
}

fn calculate_metadata_value_size(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Null => 4, // "null"
        serde_json::Value::Bool(_) => 5, // "false" (worst case)
        serde_json::Value::Number(n) => n.to_string().len(),
        serde_json::Value::String(s) => s.len() + 2, // Include quotes in size estimation
        serde_json::Value::Array(arr) => {
            2 + arr.iter().map(calculate_metadata_value_size).sum::<usize>() // [] + content
        }
        serde_json::Value::Object(obj) => {
            2 + obj.iter().map(|(k, v)| k.len() + 3 + calculate_metadata_value_size(v)).sum::<usize>() // {} + "key": + content
        }
    }
}

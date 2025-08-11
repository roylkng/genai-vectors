use axum::{Router, routing::{post, get}, extract::{State, Path, Query}, Json, serve, response::{IntoResponse, Response}, http::StatusCode};
use crate::{model::*, ingest::Ingestor, minio::S3Client, model::FilterableKey};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::net::TcpListener;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

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

#[derive(Deserialize)]
struct S3CreateIndexRequest {
    #[serde(rename = "indexName")]
    index_name: String,
    #[serde(rename = "vectorBucketName")]
    vector_bucket_name: String,
    #[serde(rename = "dataType")]
    data_type: String,
    dimension: u32,
    #[serde(rename = "distanceMetric")]
    distance_metric: String,
    algorithm: Option<String>,
    #[serde(rename = "hnswThreshold")]
    hnsw_threshold: Option<usize>,
    #[serde(default)]
    filterable_keys: Vec<FilterableKey>,
}

#[derive(Deserialize)]
struct S3GetVectorsRequest {
    #[serde(rename = "indexName")]
    index_name: String,
    #[serde(rename = "vectorBucketName")]
    vector_bucket_name: String,
    keys: Vec<String>,
    #[serde(rename = "returnData", default)]
    return_data: bool,
    #[serde(rename = "returnMetadata", default)]
    return_metadata: bool,
}

#[derive(Deserialize)]
struct S3DeleteVectorsRequest {
    #[serde(rename = "indexName")]
    index_name: String,
    #[serde(rename = "vectorBucketName")]
    vector_bucket_name: String,
    keys: Vec<String>,
}

#[derive(Deserialize)]
struct S3ListVectorsRequest {
    #[serde(rename = "indexName")]
    index_name: String,
    #[serde(rename = "vectorBucketName")]
    vector_bucket_name: String,
    #[serde(rename = "maxResults")]
    max_results: Option<u32>,
    #[serde(rename = "nextToken")]
    next_token: Option<String>,
}

// Legacy struct - keeping for backward compatibility
#[derive(Deserialize)]
struct S3CreateIndexRequestLegacy {
    #[serde(rename = "IndexName")]
    index_name: String,
    #[serde(rename = "IndexConfiguration")]
    index_configuration: S3IndexConfiguration,
}

#[derive(Deserialize)]
struct S3IndexConfiguration {
    #[serde(rename = "Dimension")]
    dimension: u32,
    #[serde(rename = "DistanceMetric")]
    distance_metric: String,
    #[serde(rename = "Algorithm")]
    algorithm: String,
    #[serde(rename = "Parameters")]
    parameters: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct S3PutVectorsRequest {
    #[serde(rename = "indexName")]
    index_name: String,
    #[serde(rename = "vectorBucketName")]
    vector_bucket_name: String,
    vectors: Vec<S3Vector>,
}

#[derive(Deserialize, Clone)]
struct S3Vector {
    key: String,
    data: S3VectorData,
    metadata: serde_json::Value,
}

#[derive(Deserialize, Clone)]
struct S3VectorData {
    float32: Vec<f32>,
}

#[derive(Deserialize)]
struct S3QueryVectorsRequest {
    #[serde(rename = "indexName")]
    index_name: String,
    #[serde(rename = "vectorBucketName")]
    vector_bucket_name: String,
    #[serde(rename = "queryVector")]
    query_vector: S3VectorData,
    #[serde(rename = "topK")]
    top_k: usize,
    #[serde(rename = "searchConfiguration")]
    search_configuration: Option<S3SearchConfiguration>,
}

#[derive(Deserialize)]
struct S3SearchConfiguration {
    #[serde(rename = "ProbeCount")]
    probe_count: Option<u32>,
}

#[derive(Clone)]
struct AppState {
    s3: S3Client,
    ingest: Arc<Ingestor>,
}

#[derive(Deserialize)]
struct IndexManifest {
    dim: u32,
    metric: String,
    total_vectors: usize,
}

// S3 Vectors API Endpoints
async fn s3_vectors_handler(
    Path(operation): Path<String>,
    Query(query): Query<S3VectorBucketQuery>,
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>
) -> Response {
    match operation.as_str() {
        "CreateVectorBucket" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_create_vector_bucket(bucket_name.to_string(), state).await.into_response()
        },
        "ListVectorBuckets" => s3_list_vector_buckets(state).await.into_response(),
        "GetVectorBucket" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_get_vector_bucket(bucket_name.to_string(), state).await.into_response()
        },
        "DeleteVectorBucket" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_delete_vector_bucket(bucket_name.to_string(), state).await.into_response()
        },
        "CreateIndex" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_create_index(bucket_name.to_string(), body, state).await.into_response()
        },
        "ListIndexes" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_list_indexes(bucket_name.to_string(), state).await.into_response()
        },
        "GetIndex" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_get_index(bucket_name.to_string(), body, state).await.into_response()
        },
        "DeleteIndex" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_delete_index(bucket_name.to_string(), body, state).await.into_response()
        },
        "PutVectors" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_put_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "ListVectors" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_list_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "GetVectors" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_get_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "DeleteVectors" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_delete_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "QueryVectors" => {
            let bucket_name = body.get("vectorBucketName").and_then(|v| v.as_str()).unwrap_or("default-bucket");
            s3_query_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        _ => (StatusCode::BAD_REQUEST, format!("Invalid S3 vectors operation: {}", operation)).into_response(),
    }
}

async fn s3_create_vector_bucket(bucket: String, _state: AppState) -> impl IntoResponse {
    Json(json!({ "VectorBucket": bucket })).into_response()
}

async fn s3_get_vector_bucket(bucket: String, _state: AppState) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "bucket": bucket, "exists": true })))
}

async fn s3_delete_vector_bucket(bucket: String, _state: AppState) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "bucket": bucket, "deleted": true })))
}

async fn s3_list_indexes(bucket: String, state: AppState) -> impl IntoResponse {
    let objects = match state.s3.list_objects("indexes/").await {
        Ok(objects) => objects,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list indexes: {}", e)).into_response(),
    };

    let mut index_names = std::collections::HashSet::new();
    for obj_key in objects {
        if let Some(name) = obj_key.strip_prefix("indexes/").and_then(|s| s.split('/').next()) {
            if !name.is_empty() {
                index_names.insert(name.to_string());
            }
        }
    }

    let mut indexes = Vec::new();
    for index_name in index_names {
        let manifest_key = format!("indexes/{}/manifest.json", index_name);
        if let Ok(manifest_data) = state.s3.get_object(&manifest_key).await {
            if let Ok(manifest) = serde_json::from_slice::<IndexManifest>(&manifest_data) {
                indexes.push(json!({
                    "name": index_name,
                    "dimension": manifest.dim,
                    "metric": manifest.metric,
                    "vector_count": manifest.total_vectors
                }));
            }
        }
    }
    
    (StatusCode::OK, Json(json!({ "bucket": bucket, "indexes": indexes }))).into_response()
}

async fn s3_get_index(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    let index_name = match body.get("indexName").and_then(|v| v.as_str()) {
        Some(name) => name,
        None => return (StatusCode::BAD_REQUEST, "indexName is required".to_string()).into_response(),
    };

    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    let manifest_data = match state.s3.get_object(&manifest_key).await {
        Ok(data) => data,
        Err(_) => return (StatusCode::NOT_FOUND, "Index not found".to_string()).into_response(),
    };

    let manifest: IndexManifest = match serde_json::from_slice(&manifest_data) {
        Ok(m) => m,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to parse manifest: {}", e)).into_response(),
    };

    let response = json!({
        "index": {
            "vectorBucketName": bucket,
            "indexName": index_name,
            "indexArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}/index/{}", bucket, index_name),
            "creationTime": "2024-01-01T00:00:00Z",
            "dataType": "FLOAT32",
            "dimension": manifest.dim,
            "distanceMetric": manifest.metric.to_uppercase(),
        }
    });
    
    (StatusCode::OK, Json(response)).into_response()
}

async fn s3_delete_index(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    let index_name = match body.get("indexName").and_then(|v| v.as_str()) {
        Some(name) => name,
        None => return (StatusCode::BAD_REQUEST, "indexName is required".to_string()).into_response(),
    };

    let prefix = format!("indexes/{}/", index_name);
    let objects = match state.s3.list_objects(&prefix).await {
        Ok(objects) => objects,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list objects for deletion: {}", e)).into_response(),
    };

    for obj_key in objects {
        if let Err(e) = state.s3.delete_object(&obj_key).await {
            tracing::error!("Failed to delete object {}: {}", obj_key, e);
        }
    }

    (StatusCode::OK, Json(json!({ "bucket": bucket, "index": index_name, "deleted": true }))).into_response()
}

async fn s3_list_vectors(bucket: String, body: serde_json::Value, _state: AppState) -> impl IntoResponse {
    let index_name = body.get("index").and_then(|v| v.as_str()).unwrap_or("default");
    (StatusCode::OK, Json(json!({ "bucket": bucket, "index": index_name, "vectors": [], "count": 0 })))
}

async fn s3_get_vectors(bucket: String, body: serde_json::Value, _state: AppState) -> impl IntoResponse {
    let index_name = body.get("index").and_then(|v| v.as_str()).unwrap_or("default");
    (StatusCode::OK, Json(json!({ "bucket": bucket, "index": index_name, "vectors": [], "found": 0 })))
}

async fn s3_delete_vectors(bucket: String, body: serde_json::Value, _state: AppState) -> impl IntoResponse {
    let index_name = body.get("index").and_then(|v| v.as_str()).unwrap_or("default");
    let vector_ids = body.get("ids").and_then(|v| v.as_array()).map(|arr| arr.len()).unwrap_or(0);
    (StatusCode::OK, Json(json!({ "bucket": bucket, "index": index_name, "deleted": vector_ids })))
}

async fn s3_list_vector_buckets(_state: AppState) -> impl IntoResponse {
    Json(json!({ "VectorBuckets": [{"Name": "vectors", "CreationDate": "2024-01-01T00:00:00Z"}] })).into_response()
}

async fn s3_create_index(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    let req: S3CreateIndexRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    let create_index_req = CreateIndex {
        name: req.index_name.clone(),
        dim: req.dimension,
        metric: req.distance_metric.to_lowercase(),
        nlist: 16,
        m: 8,
        nbits: 8,
        default_nprobe: Some(8),
        algorithm: req.algorithm.clone(),
        hnsw_threshold: req.hnsw_threshold,
        filterable_keys: req.filterable_keys,
    };
    
    let config_key = format!("indexes/{}/config.json", create_index_req.name);
    let config_data = match serde_json::to_vec(&create_index_req) {
        Ok(data) => data,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization error: {}", e)).into_response(),
    };
    
    if let Err(e) = state.s3.put_object(&config_key, config_data.into()).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create index: {}", e)).into_response();
    }
    
    Json(json!({
        "IndexName": req.index_name,
        "IndexArn": format!("arn:aws:s3:::{}/index/{}", bucket, req.index_name)
    })).into_response()
}

async fn s3_put_vectors(_bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    let req: S3PutVectorsRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    let vector_count = req.vectors.len();
    let vectors: Vec<VectorRecord> = req.vectors.into_iter().map(|v| VectorRecord {
        id: v.key,
        embedding: v.data.float32,
        meta: v.metadata,
        created_at: chrono::Utc::now(),
    }).collect();

    let put_vectors_req = PutVectors {
        index: req.index_name.clone(),
        vectors,
    };

    if let Err(e) = state.ingest.append(put_vectors_req.vectors, &put_vectors_req.index).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Ingestion failed: {}", e)).into_response();
    }

    Json(json!({ "Status": "Success", "VectorCount": vector_count })).into_response()
}

async fn s3_query_vectors(_bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    let req: S3QueryVectorsRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    let query_req = QueryRequest {
        index: req.index_name,
        embedding: req.query_vector.float32,
        topk: req.top_k,
        nprobe: req.search_configuration.and_then(|sc| sc.probe_count),
        filter: None,
    };
    
    match crate::query::search(state.s3, query_req).await {
        Ok(resp) => {
            let empty_vec = vec![];
            let results = resp.get("results").and_then(|r| r.as_array()).unwrap_or(&empty_vec);
            let s3_results: Vec<serde_json::Value> = results.iter().map(|result| {
                json!({
                    "Id": result.get("id").unwrap_or(&json!("unknown")),
                    "Score": result.get("score").unwrap_or(&json!(0.0)),
                    "Metadata": result.get("metadata").unwrap_or(&json!({}))
                })
            }).collect();
            
            Json(json!({ "Results": s3_results, "RequestId": Uuid::new_v4().to_string() })).into_response()
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Query failed: {}", e)).into_response(),
    }
}

pub async fn run() -> anyhow::Result<()> {
    let bucket = std::env::var("VEC_BUCKET").unwrap_or_else(|_| "vectors".to_string());
    let s3 = S3Client::from_env().await?;
    let ingest = Arc::new(Ingestor::new(s3.clone(), bucket));

    let state = AppState { s3, ingest };

    let app = Router::new()
        .route("/:operation", post(s3_vectors_handler))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("API listening on {addr}");
    serve(listener, app).await?;
    Ok(())
}

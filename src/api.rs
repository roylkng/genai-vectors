use axum::{Router, routing::post, extract::{State, Path, Query}, Json, serve, response::{IntoResponse, Response}, http::StatusCode};
use crate::{model::*, ingest::Ingestor, minio::S3Client, model::FilterableKey};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::net::TcpListener;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

// Metadata validation constants (AWS S3 Vectors limits)
const MAX_FILTERABLE_METADATA_SIZE: usize = 2048; // 2KB
const MAX_TOTAL_METADATA_SIZE: usize = 40960; // 40KB

// Metadata schema validation function
fn validate_metadata_schema(metadata: &serde_json::Value, filterable_keys: &[FilterableKey]) -> Result<(), String> {
    let metadata_str = metadata.to_string();
    let total_size = metadata_str.len();
    
    // Check total metadata size limit
    if total_size > MAX_TOTAL_METADATA_SIZE {
        return Err(format!("Total metadata size {} bytes exceeds limit of {} bytes", total_size, MAX_TOTAL_METADATA_SIZE));
    }
    
    // Extract filterable metadata and check size
    let mut filterable_size = 0;
    if let serde_json::Value::Object(map) = metadata {
        for filterable_key in filterable_keys {
            if let Some(value) = map.get(&filterable_key.name) {
                let value_str = value.to_string();
                filterable_size += value_str.len();
                
                // Validate type compatibility
                match filterable_key.key_type.as_str() {
                    "string" => {
                        if !value.is_string() {
                            return Err(format!("Key '{}' should be string but got {}", filterable_key.name, value_str));
                        }
                    },
                    "number" | "int64" | "float64" => {
                        if !value.is_number() {
                            return Err(format!("Key '{}' should be number but got {}", filterable_key.name, value_str));
                        }
                    },
                    "boolean" => {
                        if !value.is_boolean() {
                            return Err(format!("Key '{}' should be boolean but got {}", filterable_key.name, value_str));
                        }
                    },
                    "array" => {
                        if !value.is_array() {
                            return Err(format!("Key '{}' should be array but got {}", filterable_key.name, value_str));
                        }
                    },
                    _ => {
                        return Err(format!("Unsupported filterable key type: {}", filterable_key.key_type));
                    }
                }
            }
        }
    }
    
    // Check filterable metadata size limit
    if filterable_size > MAX_FILTERABLE_METADATA_SIZE {
        return Err(format!("Filterable metadata size {} bytes exceeds limit of {} bytes", filterable_size, MAX_FILTERABLE_METADATA_SIZE));
    }
    
    Ok(())
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
    #[serde(default)]
    non_filterable_keys: Vec<String>,
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
    Query(_query): Query<S3VectorBucketQuery>,
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

async fn s3_delete_vector_bucket(bucket: String, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-vector-bucket request for bucket: {}", bucket);
    
    // First, list all indexes in the bucket to delete them
    match state.s3.list_objects("indexes/").await {
        Ok(objects) => {
            let mut index_names = std::collections::HashSet::new();
            for obj_key in &objects {
                if let Some(name) = obj_key.strip_prefix("indexes/").and_then(|s| s.split('/').next()) {
                    if !name.is_empty() {
                        index_names.insert(name.to_string());
                    }
                }
            }
            
            // Delete all indexes first
            for index_name in index_names {
                let prefix = format!("indexes/{}/", index_name);
                if let Ok(index_objects) = state.s3.list_objects(&prefix).await {
                    for obj_key in index_objects {
                        if let Err(e) = state.s3.delete_object(&obj_key).await {
                            tracing::error!("Failed to delete index object {}: {}", obj_key, e);
                        }
                    }
                }
            }
            
            // Delete any remaining objects in the bucket (WAL files, etc.)
            for obj_key in objects {
                if let Err(e) = state.s3.delete_object(&obj_key).await {
                    tracing::error!("Failed to delete object {}: {}", obj_key, e);
                }
            }
            
            // Delete other bucket contents like WAL files
            if let Ok(wal_objects) = state.s3.list_objects("wal/").await {
                for obj_key in wal_objects {
                    if let Err(e) = state.s3.delete_object(&obj_key).await {
                        tracing::error!("Failed to delete WAL object {}: {}", obj_key, e);
                    }
                }
            }
            
            // Delete any staged objects
            if let Ok(staged_objects) = state.s3.list_objects("staged/").await {
                for obj_key in staged_objects {
                    if let Err(e) = state.s3.delete_object(&obj_key).await {
                        tracing::error!("Failed to delete staged object {}: {}", obj_key, e);
                    }
                }
            }
            
            let response = json!({
                "bucket": bucket,
                "deleted": true,
                "status": "success",
                "message": "All vector data and indexes deleted from bucket"
            });
            
            (StatusCode::OK, Json(response))
        },
        Err(e) => {
            tracing::error!("Failed to list objects for bucket deletion: {}", e);
            let error_response = json!({
                "bucket": bucket,
                "deleted": false,
                "status": "error",
                "error": format!("Failed to delete bucket contents: {}", e)
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
        }
    }
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

    // First try to get manifest.json (for indexed data), then fall back to config.json (for new indexes)
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    let config_key = format!("indexes/{}/config.json", index_name);
    
    // Try manifest first (for indexes with processed data)
    if let Ok(manifest_data) = state.s3.get_object(&manifest_key).await {
        if let Ok(manifest) = serde_json::from_slice::<IndexManifest>(&manifest_data) {
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
            return (StatusCode::OK, Json(response)).into_response();
        }
    }
    
    // Fall back to config.json (for newly created indexes)
    if let Ok(config_data) = state.s3.get_object(&config_key).await {
        if let Ok(config) = serde_json::from_slice::<CreateIndex>(&config_data) {
            let response = json!({
                "index": {
                    "vectorBucketName": bucket,
                    "indexName": index_name,
                    "indexArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}/index/{}", bucket, index_name),
                    "creationTime": "2024-01-01T00:00:00Z",
                    "dataType": "FLOAT32",
                    "dimension": config.dim,
                    "distanceMetric": config.metric.to_uppercase(),
                }
            });
            return (StatusCode::OK, Json(response)).into_response();
        }
    }

    (StatusCode::NOT_FOUND, "Index not found".to_string()).into_response()
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

async fn s3_list_vectors(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 list-vectors request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    let max_results = body.get("maxResults")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as usize;
    
    // Try to load the manifest to get vector information
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    
    match state.s3.get_object(&manifest_key).await {
        Ok(manifest_data) => {
            if let Ok(manifest_str) = String::from_utf8(manifest_data.to_vec()) {
                if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                    let total_vectors = manifest.get("total_vectors")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    
                    let shards = manifest.get("shards")
                        .and_then(|v| v.as_array());
                    
                    let empty_vec = vec![];
                    let shards = shards.unwrap_or(&empty_vec);
                    
                    let mut vectors_info = Vec::new();
                    let mut count = 0;
                    
                    // List vectors from shards (limiting to max_results)
                    for shard in shards {
                        if count >= max_results { break; }
                        
                        if let Some(shard_id) = shard.get("shard_id").and_then(|v| v.as_str()) {
                            if let Some(vector_count) = shard.get("vector_count").and_then(|v| v.as_u64()) {
                                let shard_info = json!({
                                    "shard_id": shard_id,
                                    "vector_count": vector_count,
                                    "created_at": shard.get("created_at").and_then(|v| v.as_str()).unwrap_or("unknown")
                                });
                                vectors_info.push(shard_info);
                                count += 1;
                            }
                        }
                    }
                    
                    let response = json!({
                        "bucket": bucket,
                        "index": index_name,
                        "vectors": vectors_info,
                        "count": vectors_info.len(),
                        "total_vectors": total_vectors
                    });
                    
                    return (StatusCode::OK, Json(response));
                }
            }
        },
        Err(e) => {
            tracing::warn!("Failed to load manifest for index {}: {}", index_name, e);
        }
    }
    
    // Fallback response if manifest not found
    let response = json!({
        "bucket": bucket,
        "index": index_name,
        "vectors": [],
        "count": 0,
        "total_vectors": 0
    });
    
    (StatusCode::OK, Json(response))
}

async fn s3_get_vectors(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 get-vectors request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    let keys = body.get("keys")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();
    
    let return_data = body.get("returnData")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    let return_metadata = body.get("returnMetadata")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    if keys.is_empty() {
        let response = json!({
            "bucket": bucket,
            "index": index_name,
            "vectors": [],
            "found": 0
        });
        return (StatusCode::OK, Json(response));
    }
    
    // Try to load the manifest to get shard information
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    
    match state.s3.get_object(&manifest_key).await {
        Ok(manifest_data) => {
            if let Ok(manifest_str) = String::from_utf8(manifest_data.to_vec()) {
                if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                    let shards = manifest.get("shards")
                        .and_then(|v| v.as_array());
                    
                    let empty_vec2 = vec![];
                    let shards = shards.unwrap_or(&empty_vec2);
                    
                    let mut found_vectors = Vec::new();
                    
                    // Search through shards for the requested keys
                    for shard in shards {
                        if let Some(shard_id) = shard.get("shard_id").and_then(|v| v.as_str()) {
                            // Try to load the ID mapping for this shard
                            let id_map_path = format!("indexes/{}/shards/{}/id_map.json", index_name, shard_id);
                            
                            if let Ok(id_map_data) = state.s3.get_object(&id_map_path).await {
                                if let Ok(id_map_str) = String::from_utf8(id_map_data.to_vec()) {
                                    if let Ok(id_map) = serde_json::from_str::<Vec<(i64, String)>>(&id_map_str) {
                                        // Check if any requested keys are in this shard
                                        for key in &keys {
                                            if let Some((faiss_id, _)) = id_map.iter().find(|(_, string_id)| string_id == key) {
                                                // Try to load metadata for this vector
                                                if return_metadata {
                                                    let metadata_path = format!("indexes/{}/shards/{}/metadata.json", index_name, shard_id);
                                                    if let Ok(metadata_data) = state.s3.get_object(&metadata_path).await {
                                                        if let Ok(metadata_str) = String::from_utf8(metadata_data.to_vec()) {
                                                            if let Ok(metadata_map) = serde_json::from_str::<std::collections::HashMap<String, serde_json::Value>>(&metadata_str) {
                                                                let mut vector_response = json!({
                                                                    "key": key,
                                                                    "faiss_id": faiss_id
                                                                });
                                                                
                                                                if let Some(metadata) = metadata_map.get(*key) {
                                                                    vector_response["metadata"] = metadata.clone();
                                                                }
                                                                
                                                                if return_data {
                                                                    // Note: Getting actual vector data would require loading the Faiss index
                                                                    // This is complex and would require the actual Faiss implementation
                                                                    vector_response["data"] = json!({
                                                                        "float32": [] // Placeholder - would need Faiss index loading
                                                                    });
                                                                }
                                                                
                                                                found_vectors.push(vector_response);
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    // Just return key information
                                                    let mut vector_response = json!({
                                                        "key": key,
                                                        "faiss_id": faiss_id
                                                    });
                                                    
                                                    if return_data {
                                                        vector_response["data"] = json!({
                                                            "float32": [] // Placeholder
                                                        });
                                                    }
                                                    
                                                    found_vectors.push(vector_response);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    let response = json!({
                        "bucket": bucket,
                        "index": index_name,
                        "vectors": found_vectors,
                        "found": found_vectors.len(),
                        "requested": keys.len()
                    });
                    
                    return (StatusCode::OK, Json(response));
                }
            }
        },
        Err(e) => {
            tracing::warn!("Failed to load manifest for index {}: {}", index_name, e);
        }
    }
    
    // Fallback response if manifest not found
    let response = json!({
        "bucket": bucket,
        "index": index_name,
        "vectors": [],
        "found": 0,
        "requested": keys.len()
    });
    
    (StatusCode::OK, Json(response))
}

async fn s3_delete_vectors(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-vectors request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    let keys = body.get("keys")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();
    
    if keys.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "No vector keys provided for deletion"
        }))).into_response();
    }
    
    let mut deleted_count = 0;
    let mut errors = Vec::new();
    
    // Try to load the manifest to get shard information
    let manifest_key = format!("indexes/{}/manifest.json", index_name);
    
    match state.s3.get_object(&manifest_key).await {
        Ok(manifest_data) => {
            if let Ok(manifest_str) = String::from_utf8(manifest_data.to_vec()) {
                if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                    let shards = manifest.get("shards")
                        .and_then(|v| v.as_array());
                    
                    let empty_vec = vec![];
                    let shards = shards.unwrap_or(&empty_vec);
                    
                    // Search through shards to find and remove the vectors
                    for shard in shards {
                        if let Some(shard_id) = shard.get("shard_id").and_then(|v| v.as_str()) {
                            // Try to load and update the ID mapping for this shard
                            let id_map_path = format!("indexes/{}/shards/{}/id_map.json", index_name, shard_id);
                            
                            if let Ok(id_map_data) = state.s3.get_object(&id_map_path).await {
                                if let Ok(id_map_str) = String::from_utf8(id_map_data.to_vec()) {
                                    if let Ok(mut id_map) = serde_json::from_str::<Vec<(i64, String)>>(&id_map_str) {
                                        // Remove vectors from ID mapping
                                        let original_len = id_map.len();
                                        id_map.retain(|(_, string_id)| !keys.contains(&string_id.as_str()));
                                        let removed_from_shard = original_len - id_map.len();
                                        
                                        if removed_from_shard > 0 {
                                            // Update the ID mapping file
                                            match serde_json::to_vec(&id_map) {
                                                Ok(updated_id_map) => {
                                                    if let Err(e) = state.s3.put_object(&id_map_path, updated_id_map.into()).await {
                                                        errors.push(format!("Failed to update ID map for shard {}: {}", shard_id, e));
                                                    } else {
                                                        deleted_count += removed_from_shard;
                                                    }
                                                },
                                                Err(e) => {
                                                    errors.push(format!("Failed to serialize updated ID map for shard {}: {}", shard_id, e));
                                                }
                                            }
                                            
                                            // Also update metadata file to remove deleted vector metadata
                                            let metadata_path = format!("indexes/{}/shards/{}/metadata.json", index_name, shard_id);
                                            if let Ok(metadata_data) = state.s3.get_object(&metadata_path).await {
                                                if let Ok(metadata_str) = String::from_utf8(metadata_data.to_vec()) {
                                                    if let Ok(mut metadata_map) = serde_json::from_str::<std::collections::HashMap<String, serde_json::Value>>(&metadata_str) {
                                                        // Remove metadata for deleted vectors
                                                        for key in &keys {
                                                            metadata_map.remove(*key);
                                                        }
                                                        
                                                        match serde_json::to_vec(&metadata_map) {
                                                            Ok(updated_metadata) => {
                                                                if let Err(e) = state.s3.put_object(&metadata_path, updated_metadata.into()).await {
                                                                    errors.push(format!("Failed to update metadata for shard {}: {}", shard_id, e));
                                                                }
                                                            },
                                                            Err(e) => {
                                                                errors.push(format!("Failed to serialize updated metadata for shard {}: {}", shard_id, e));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Update the manifest to reflect the deletion
                    if deleted_count > 0 {
                        if let Ok(mut manifest_value) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                            if let Some(total_vectors) = manifest_value.get_mut("total_vectors") {
                                if let Some(current_total) = total_vectors.as_u64() {
                                    *total_vectors = json!(current_total.saturating_sub(deleted_count as u64));
                                }
                            }
                            
                            match serde_json::to_vec(&manifest_value) {
                                Ok(updated_manifest) => {
                                    if let Err(e) = state.s3.put_object(&manifest_key, updated_manifest.into()).await {
                                        errors.push(format!("Failed to update manifest: {}", e));
                                    }
                                },
                                Err(e) => {
                                    errors.push(format!("Failed to serialize updated manifest: {}", e));
                                }
                            }
                        }
                    }
                }
            }
        },
        Err(e) => {
            tracing::warn!("Failed to load manifest for index {}: {}", index_name, e);
            errors.push(format!("Index manifest not found: {}", e));
        }
    }
    
    let response = if errors.is_empty() {
        json!({
            "bucket": bucket,
            "index": index_name,
            "deleted": deleted_count,
            "requested": keys.len(),
            "status": "success"
        })
    } else {
        json!({
            "bucket": bucket,
            "index": index_name,
            "deleted": deleted_count,
            "requested": keys.len(),
            "status": "partial_success",
            "errors": errors
        })
    };
    
    (StatusCode::OK, Json(response)).into_response()
}

async fn s3_list_vector_buckets(state: AppState) -> impl IntoResponse {
    // List actual buckets from MinIO
    match state.s3.list_buckets().await {
        Ok(buckets) => {
            let vector_buckets: Vec<_> = buckets.into_iter().map(|bucket| {
                json!({
                    "Name": bucket,
                    "CreationDate": "2024-01-01T00:00:00Z"
                })
            }).collect();
            
            Json(json!({
                "VectorBuckets": vector_buckets
            })).into_response()
        },
        Err(e) => {
            tracing::error!("Failed to list buckets: {}", e);
            let error_response = json!({
                "VectorBuckets": [],
                "error": format!("Failed to list buckets: {}", e)
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
        }
    }
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
        non_filterable_keys: req.non_filterable_keys,
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
    
    // AWS S3 Vectors limit: max 500 vectors per request
    if req.vectors.len() > 500 {
        return (StatusCode::BAD_REQUEST, format!("Too many vectors: {} (max 500 allowed)", req.vectors.len())).into_response();
    }
    
    // Load index config to validate metadata schema
    let config_key = format!("indexes/{}/config.json", req.index_name);
    if let Ok(config_data) = state.s3.get_object(&config_key).await {
        if let Ok(config) = serde_json::from_slice::<CreateIndex>(&config_data) {
            // Validate metadata against schema
            for vector in &req.vectors {
                if let Err(e) = validate_metadata_schema(&vector.metadata, &config.filterable_keys) {
                    return (StatusCode::BAD_REQUEST, format!("Metadata validation failed for vector '{}': {}", vector.key, e)).into_response();
                }
            }
        }
    }
    
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
    // Extract metadata filter before consuming body
    let metadata_filter = body.get("metadataFilter").cloned();
    
    let req: S3QueryVectorsRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    let query_req = QueryRequest {
        index: req.index_name,
        embedding: req.query_vector.float32,
        topk: req.top_k,
        nprobe: req.search_configuration.and_then(|sc| sc.probe_count),
        filter: metadata_filter,
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

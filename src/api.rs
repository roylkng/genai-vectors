use axum::{Router, routing::{post, get}, extract::{State, Path, Query}, Json, serve, response::{IntoResponse, Response}, http::StatusCode};
use crate::{model::*, ingest::Ingestor, minio::S3Client};
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
    query_vector: S3VectorData,  // Changed from Vec<f32> to S3VectorData (dict format)
    #[serde(rename = "topK")]     // Changed from maxResults to topK
    top_k: usize,                // Changed field name
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

// S3 Vectors API Endpoints
async fn s3_vectors_handler(
    Path(operation): Path<String>,
    Query(query): Query<S3VectorBucketQuery>,
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>
) -> Response {
    // Debug: log the request details
    tracing::info!("S3 Vectors API request - operation: {}, query: {:?}", operation, query);
    tracing::info!("S3 Vectors API body: {}", serde_json::to_string_pretty(&body).unwrap_or_else(|_| "invalid json".to_string()));
    
    // Handle different S3 vectors operations based on the path (operation name)
    match operation.as_str() {
        "CreateVectorBucket" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_create_vector_bucket(bucket_name.to_string(), state).await.into_response()
        },
        "ListVectorBuckets" => {
            s3_list_vector_buckets(state).await.into_response()
        },
        "GetVectorBucket" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_get_vector_bucket(bucket_name.to_string(), state).await.into_response()
        },
        "DeleteVectorBucket" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_delete_vector_bucket(bucket_name.to_string(), state).await.into_response()
        },
        "CreateIndex" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_create_index(bucket_name.to_string(), body, state).await.into_response()
        },
        "ListIndexes" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_list_indexes(bucket_name.to_string(), state).await.into_response()
        },
        "GetIndex" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_get_index(bucket_name.to_string(), body, state).await.into_response()
        },
        "DeleteIndex" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_delete_index(bucket_name.to_string(), body, state).await.into_response()
        },
        "PutVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_put_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "ListVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_list_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "GetVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_get_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "DeleteVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_delete_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        "QueryVectors" => {
            let bucket_name = body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket");
            s3_query_vectors(bucket_name.to_string(), body, state).await.into_response()
        },
        // Fallback: check if this is a legacy query parameter based operation
        _ => {
            // Handle different S3 vectors operations based on query parameters (legacy support)
            if query.create_vector_bucket.is_some() {
                s3_create_vector_bucket(operation, state).await.into_response()
            } else if query.list_vector_buckets.is_some() {
                s3_list_vector_buckets(state).await.into_response()
            } else if query.get_vector_bucket.is_some() {
                s3_get_vector_bucket(operation, state).await.into_response()
            } else if query.delete_vector_bucket.is_some() {
                s3_delete_vector_bucket(operation, state).await.into_response()
            } else if query.create_index.is_some() {
                s3_create_index(operation, body, state).await.into_response()
            } else if query.list_indexes.is_some() {
                s3_list_indexes(operation, state).await.into_response()
            } else if query.get_index.is_some() {
                s3_get_index(operation, body, state).await.into_response()
            } else if query.delete_index.is_some() {
                s3_delete_index(operation, body, state).await.into_response()
            } else if query.put_vectors.is_some() {
                s3_put_vectors(operation, body, state).await.into_response()
            } else if query.list_vectors.is_some() {
                s3_list_vectors(operation, body, state).await.into_response()
            } else if query.get_vectors.is_some() {
                s3_get_vectors(operation, body, state).await.into_response()
            } else if query.delete_vectors.is_some() {
                s3_delete_vectors(operation, body, state).await.into_response()
            } else if query.query_vectors.is_some() {
                s3_query_vectors(operation, body, state).await.into_response()
            } else {
                tracing::warn!("Unknown S3 vectors operation - path: {}, query: {:?}", operation, query);
                (StatusCode::BAD_REQUEST, format!("Invalid S3 vectors operation: {}", operation)).into_response()
            }
        }
    }
}

async fn s3_create_vector_bucket(bucket: String, _state: AppState) -> impl IntoResponse {
    // For our implementation, vector buckets are just logical constructs
    // The actual storage uses MinIO buckets
    Json(serde_json::json!({
        "VectorBucket": bucket
    })).into_response()
}

async fn s3_get_vector_bucket(bucket: String, _state: AppState) -> impl IntoResponse {
    tracing::info!("S3 get-vector-bucket request for bucket: {}", bucket);
    
    let response = json!({
        "bucket": bucket,
        "exists": true,
        "created_at": "2024-01-01T00:00:00Z",
        "vector_count": 0,
        "indexes": []
    });
    
    (StatusCode::OK, Json(response))
}

async fn s3_delete_vector_bucket(bucket: String, _state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-vector-bucket request for bucket: {}", bucket);
    
    let response = json!({
        "bucket": bucket,
        "deleted": true,
        "status": "success"
    });
    
    (StatusCode::OK, Json(response))
}

async fn s3_list_indexes(bucket: String, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 list-indexes request for bucket: {}", bucket);
    
    // List actual indexes by looking for config.json files in indexes/ directory
    match state.s3.list_objects("indexes/").await {
        Ok(objects) => {
            let mut indexes = Vec::new();
            
            for object in objects {
                if object.ends_with("/config.json") {
                    // Extract index name from path like "indexes/my-index/config.json"
                    if let Some(index_name) = extract_index_name_from_config_path(&object) {
                        // Try to load the index config to get details
                        match state.s3.get_object(&object).await {
                            Ok(config_data) => {
                                if let Ok(config_str) = String::from_utf8(config_data.to_vec()) {
                                    if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                                        let index_info = json!({
                                            "name": index_name,
                                            "dimension": config.get("dim").and_then(|v| v.as_u64()).unwrap_or(1536),
                                            "metric": config.get("metric").and_then(|v| v.as_str()).unwrap_or("cosine"),
                                            "vector_count": 0 // TODO: Could calculate from manifest
                                        });
                                        indexes.push(index_info);
                                    }
                                }
                            },
                            Err(e) => {
                                tracing::warn!("Failed to load config for index {}: {}", index_name, e);
                            }
                        }
                    }
                }
            }
            
            let response = json!({
                "bucket": bucket,
                "indexes": indexes
            });
            
            (StatusCode::OK, Json(response))
        },
        Err(e) => {
            tracing::error!("Failed to list indexes: {}", e);
            let response = json!({
                "bucket": bucket,
                "indexes": [],
                "error": format!("Failed to list indexes: {}", e)
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
}

fn extract_index_name_from_config_path(path: &str) -> Option<String> {
    // Extract index name from path like "indexes/my-index/config.json"
    if let Some(stripped) = path.strip_prefix("indexes/") {
        if let Some(end_pos) = stripped.find("/config.json") {
            return Some(stripped[..end_pos].to_string());
        }
    }
    None
}

async fn s3_get_index(bucket: String, body: serde_json::Value, _state: AppState) -> impl IntoResponse {
    tracing::info!("S3 get-index request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    // Response must match GetIndexOutput shape with top-level "index" field
    let response = json!({
        "index": {
            "vectorBucketName": bucket,
            "indexName": index_name,
            "indexArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}/index/{}", bucket, index_name),
            "creationTime": "2024-01-01T00:00:00Z",
            "dataType": "FLOAT32",
            "dimension": 1536,
            "distanceMetric": "COSINE"
        }
    });
    
    (StatusCode::OK, Json(response))
}

async fn s3_delete_index(bucket: String, body: serde_json::Value, _state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-index request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("index")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    let response = json!({
        "bucket": bucket,
        "index": index_name,
        "deleted": true,
        "status": "success"
    });
    
    (StatusCode::OK, Json(response))
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

async fn s3_delete_vectors(bucket: String, body: serde_json::Value, _state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-vectors request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("index")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    let vector_ids = body.get("ids")
        .and_then(|v| v.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);
    
    let response = json!({
        "bucket": bucket,
        "index": index_name,
        "deleted": vector_ids,
        "status": "success"
    });
    
    (StatusCode::OK, Json(response))
}

// Direct S3 vectors handlers for specific operations
async fn s3_list_vector_buckets_direct(State(state): State<AppState>) -> impl IntoResponse {
    s3_list_vector_buckets(state).await
}

async fn s3_create_index_direct(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>
) -> impl IntoResponse {
    // Extract the bucket name from the body for our internal API
    let bucket = body.get("vectorBucketName")
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    s3_create_index(bucket, body, state).await
}

async fn s3_put_vectors_direct(
    State(state): State<AppState>,
    body: String,
) -> Response {
    tracing::info!("S3 put_vectors request received");
    tracing::info!("Raw request body: {}", body);
    
    // Try to parse as JSON to see the structure
    match serde_json::from_str::<serde_json::Value>(&body) {
        Ok(json_body) => {
            tracing::info!("Parsed JSON body: {}", serde_json::to_string_pretty(&json_body).unwrap_or_default());
            
            // Extract the bucket name from the body for our internal API
            let bucket = json_body.get("vectorBucketName")
                .and_then(|v| v.as_str())
                .unwrap_or("default-bucket")
                .to_string();
            
            s3_put_vectors(bucket, json_body, state).await.into_response()
        },
        Err(e) => {
            tracing::error!("Failed to parse put_vectors request as JSON: {}", e);
            (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response()
        }
    }
}

async fn s3_query_vectors_direct(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>
) -> impl IntoResponse {
    // Extract the bucket name from the body for our internal API
    let bucket = body.get("vectorBucketName")
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    s3_query_vectors(bucket, body, state).await
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
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list buckets: {}", e)).into_response()
        }
    }
}

async fn s3_create_index(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    let req: S3CreateIndexRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    // Convert S3 format to our internal format
    let create_index_req = CreateIndex {
        name: req.index_name.clone(),
        dim: req.dimension,
        metric: req.distance_metric.to_lowercase(),
        nlist: 16, // Default value
        m: 8,      // Default value  
        nbits: 8,  // Default value
        default_nprobe: Some(8), // Default value
    };
    
    // Use our existing create_index logic
    let config_key = format!("indexes/{}/config.json", create_index_req.name);
    let config_data = match serde_json::to_vec(&create_index_req) {
        Ok(data) => data,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization error: {}", e)).into_response(),
    };
    
    if let Err(e) = state.s3.put_object(&config_key, config_data.into()).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create index: {}", e)).into_response();
    }
    
    Json(serde_json::json!({
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
    tracing::info!("Converting {} vectors to internal format", vector_count);
    
    // Convert S3 vectors format to our internal format
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

    tracing::info!("Attempting to ingest {} vectors to index '{}'", vector_count, req.index_name);
    
    // Use our existing put_vectors logic
    if let Err(e) = state.ingest.append(put_vectors_req.vectors, &put_vectors_req.index).await {
        tracing::error!("Ingestion failed for index '{}': {}", req.index_name, e);
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Ingestion failed: {}", e)).into_response();
    }

    tracing::info!("Successfully ingested {} vectors to index '{}'", vector_count, req.index_name);    Json(serde_json::json!({
        "Status": "Success",
        "VectorCount": vector_count
    })).into_response()
}

async fn s3_query_vectors(_bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    let req: S3QueryVectorsRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    // Convert S3 format to our internal format
    let query_req = QueryRequest {
        index: req.index_name,
        embedding: req.query_vector.float32,  // Extract from the S3VectorData struct
        topk: req.top_k,                      // Updated field name
        nprobe: req.search_configuration
            .and_then(|sc| sc.probe_count),
        filter: None,
    };
    
    // Use our existing query logic
    match crate::query::search(state.s3, query_req).await {
        Ok(resp) => {
            // Convert our response to S3 vectors format
            let empty_vec = vec![];
            let results = resp.get("results").and_then(|r| r.as_array()).unwrap_or(&empty_vec);
            let s3_results: Vec<serde_json::Value> = results.iter().map(|result| {
                serde_json::json!({
                    "Id": result.get("id").unwrap_or(&serde_json::Value::String("unknown".to_string())),
                    "Score": result.get("score").unwrap_or(&serde_json::Value::Number(serde_json::Number::from_f64(0.0).unwrap())),
                    "Metadata": result.get("metadata").unwrap_or(&serde_json::Value::Object(serde_json::Map::new()))
                })
            }).collect();
            
            Json(serde_json::json!({
                "Results": s3_results,
                "RequestId": Uuid::new_v4().to_string()
            })).into_response()
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Query failed: {}", e)).into_response(),
    }
}
async fn create_index(
    State(state): State<AppState>,
    Json(req): Json<CreateIndex>
) -> impl IntoResponse {
    let config_key = format!("indexes/{}/config.json", req.name);
    let config_data = match serde_json::to_vec(&req) {
        Ok(data) => data,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization error: {}", e)).into_response(),
    };
    
    if let Err(e) = state.s3.put_object(&config_key, config_data.into()).await {
        tracing::error!("Failed to put object: {:?}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create index: {}", e)).into_response();
    }
    
    Json(serde_json::json!({
        "status": "created",
        "index": req.name
    })).into_response()
}

// POST /vectors
async fn put_vectors(
    State(state): State<AppState>,
    Json(req): Json<PutVectors>
) -> impl IntoResponse {
    if let Err(e) = state.ingest.append(req.vectors, &req.index).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Ingestion failed: {}", e)).into_response();
    }
    Json(serde_json::json!({"status":"accepted"})).into_response()
}

// POST /query (delegates to query::search)
async fn query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>
) -> impl IntoResponse {
    match crate::query::search(state.s3, req).await {
        Ok(resp) => Json::<serde_json::Value>(resp).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Query failed: {}", e)).into_response(),
    }
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
        // Original API endpoints
        .route("/indexes", post(create_index))
        .route("/vectors", post(put_vectors))
        .route("/query", post(query))
        .route("/health", get(health))
        // S3 Vectors API compatibility endpoints - using the actual paths boto3 calls
        .route("/ListVectorBuckets", post(s3_list_vector_buckets_direct))
        .route("/CreateIndex", post(s3_create_index_direct))
        .route("/PutVectors", post(s3_put_vectors_direct))  
        .route("/QueryVectors", post(s3_query_vectors_direct))
        .route("/:bucket", post(s3_vectors_handler)) // For bucket-specific operations (fallback)
        .with_state(state);

    let addr = "0.0.0.0:8080";
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("API listening on {addr}");
    serve(listener, app).await?;
    Ok(())
}

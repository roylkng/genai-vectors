use axum::{Router, routing::{post, get}, extract::{State, Path, Query}, Json, serve, response::{IntoResponse, Response}, http::StatusCode};

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

use crate::{model::*, ingest::Ingestor, minio::S3Client};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::net::TcpListener;
use serde::{Deserialize};
use anyhow::Context;
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

async fn s3_create_vector_bucket(bucket: String, state: AppState) -> impl IntoResponse {
    // Create an actual bucket in S3
    match state.s3.client.create_bucket().bucket(&bucket).send().await {
        Ok(_) => {
            tracing::info!("Successfully created vector bucket: {}", bucket);
            Json(serde_json::json!({
                "BucketName": bucket,
                "VectorBucket": bucket
            })).into_response()
        }
        Err(e) => {
            // Bucket might already exist, which is OK
            if e.to_string().contains("BucketAlreadyExists") || e.to_string().contains("BucketAlreadyOwnedByYou") {
                tracing::info!("Vector bucket already exists: {}", bucket);
                Json(serde_json::json!({
                    "BucketName": bucket,
                    "VectorBucket": bucket
                })).into_response()
            } else {
                tracing::error!("Failed to create vector bucket {}: {}", bucket, e);
                let error_response = json!({
                    "error": format!("Failed to create bucket: {}", e)
                });
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
            }
        }
    }
}

async fn s3_get_vector_bucket(bucket: String, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 get-vector-bucket request for bucket: {}", bucket);
    
    // Check if bucket exists by trying to head it
    match state.s3.client.head_bucket().bucket(&bucket).send().await {
        Ok(_) => {
            // Bucket exists, get some metadata
            let vector_count = match state.s3.list_objects("vectors/").await {
                Ok(objects) => objects.len(),
                Err(_) => 0,
            };
            
            let indexes = match state.s3.list_objects("indexes/").await {
                Ok(objects) => {
                    objects.into_iter()
                        .filter_map(|key| {
                            if key.ends_with("/config.json") {
                                let index_name = key.strip_prefix("indexes/")?.strip_suffix("/config.json")?;
                                Some(index_name.to_string())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                }
                Err(_) => vec![],
            };
            
            let response = json!({
                "bucket": bucket,
                "exists": true,
                "created_at": "2024-01-01T00:00:00Z",
                "vector_count": vector_count,
                "indexes": indexes
            });
            
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            tracing::error!("Bucket {} does not exist or is not accessible: {}", bucket, e);
            let error_response = json!({
                "error": format!("Bucket not found: {}", e)
            });
            (StatusCode::NOT_FOUND, Json(error_response))
        }
    }
}

async fn s3_delete_vector_bucket(bucket: String, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-vector-bucket request for bucket: {}", bucket);
    
    // First, try to delete all objects in the bucket
    match state.s3.list_objects("").await {
        Ok(objects) => {
            for object_key in objects {
                if let Err(e) = state.s3.delete_object(&object_key).await {
                    tracing::warn!("Failed to delete object {}: {}", object_key, e);
                }
            }
        }
        Err(e) => {
            tracing::warn!("Failed to list objects for deletion: {}", e);
        }
    }
    
    // Now delete the bucket itself
    match state.s3.client.delete_bucket().bucket(&bucket).send().await {
        Ok(_) => {
            tracing::info!("Successfully deleted vector bucket: {}", bucket);
            let response = json!({
                "bucket": bucket,
                "deleted": true,
                "status": "success"
            });
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            tracing::error!("Failed to delete vector bucket {}: {}", bucket, e);
            let error_response = json!({
                "error": format!("Failed to delete bucket: {}", e)
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
        }
    }
}

async fn s3_list_indexes(bucket: String, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 list-indexes request for bucket: {}", bucket);
    
    // List all index configurations from S3
    match state.s3.list_objects("indexes/").await {
        Ok(objects) => {
            let mut indexes = Vec::new();
            
            for object_key in objects {
                if object_key.ends_with("/config.json") {
                    if let Some(index_name) = object_key.strip_prefix("indexes/").and_then(|s| s.strip_suffix("/config.json")) {
                        // Load the index configuration to get details
                        match state.s3.get_object(&object_key).await {
                            Ok(data) => {
                                if let Ok(config) = serde_json::from_slice::<CreateIndex>(&data) {
                                    // Count vectors for this index
                                    let vector_prefix = format!("{}/vectors/", index_name);
                                    let vector_count = state.s3.list_objects(&vector_prefix).await
                                        .map(|objects| objects.len())
                                        .unwrap_or(0);
                                    
                                    indexes.push(json!({
                                        "name": index_name,
                                        "dimension": config.dim,
                                        "metric": config.metric,
                                        "vector_count": vector_count
                                    }));
                                }
                            }
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
        }
        Err(e) => {
            tracing::error!("Failed to list indexes for bucket {}: {}", bucket, e);
            let error_response = json!({
                "error": format!("Failed to list indexes: {}", e)
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
        }
    }
}

async fn s3_get_index(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 get-index request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    // Try to load the actual index configuration from S3
    let config_key = format!("indexes/{}/config.json", index_name);
    match state.s3.get_object(&config_key).await {
        Ok(data) => {
            match serde_json::from_slice::<CreateIndex>(&data) {
                Ok(config) => {
                    // Count vectors for this index
                    let vector_prefix = format!("{}/vectors/", index_name);
                    let vector_count = state.s3.list_objects(&vector_prefix).await
                        .map(|objects| objects.len())
                        .unwrap_or(0);
                    
                    let response = json!({
                        "index": {
                            "vectorBucketName": bucket,
                            "indexName": index_name,
                            "indexArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}/index/{}", bucket, index_name),
                            "creationTime": "2024-01-01T00:00:00Z",
                            "dataType": "FLOAT32",
                            "dimension": config.dim,
                            "distanceMetric": config.metric.to_uppercase(),
                            "vectorCount": vector_count
                        }
                    });
                    
                    (StatusCode::OK, Json(response))
                }
                Err(e) => {
                    tracing::error!("Failed to parse index config for {}: {}", index_name, e);
                    let error_response = json!({
                        "error": format!("Failed to parse index config: {}", e)
                    });
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
                }
            }
        }
        Err(e) => {
            tracing::error!("Index {} not found in bucket {}: {}", index_name, bucket, e);
            let error_response = json!({
                "error": format!("Index not found: {}", e)
            });
            (StatusCode::NOT_FOUND, Json(error_response))
        }
    }
}

async fn s3_delete_index(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-index request for bucket: {}, body: {:?}", bucket, body);
    
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    // Delete all objects associated with this index
    let index_prefix = format!("{}/", index_name);
    match state.s3.list_objects(&index_prefix).await {
        Ok(objects) => {
            for object_key in objects {
                if let Err(e) = state.s3.delete_object(&object_key).await {
                    tracing::warn!("Failed to delete object {}: {}", object_key, e);
                }
            }
        }
        Err(e) => {
            tracing::warn!("Failed to list objects for index {}: {}", index_name, e);
        }
    }
    
    // Delete the index configuration
    let config_key = format!("indexes/{}/config.json", index_name);
    match state.s3.delete_object(&config_key).await {
        Ok(_) => {
            tracing::info!("Successfully deleted index: {}", index_name);
            let response = json!({
                "bucket": bucket,
                "index": index_name,
                "deleted": true,
                "status": "success"
            });
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            tracing::error!("Failed to delete index config {}: {}", index_name, e);
            let error_response = json!({
                "error": format!("Failed to delete index: {}", e)
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
        }
    }
}

async fn s3_list_vectors(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 list-vectors request for bucket: {}, body: {:?}", bucket, body);
    
    let list_request: S3ListVectorsRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => {
            tracing::error!("Failed to parse list vectors request: {}", e);
            return (StatusCode::BAD_REQUEST, "Invalid request format").into_response();
        }
    };
    
    let bucket_name = &list_request.vector_bucket_name;
    let prefix = format!("{}/vectors/", list_request.index_name);
    let max_keys = list_request.max_results.unwrap_or(1000);
    
    match state.s3.client
        .list_objects_v2()
        .bucket(bucket_name.clone())
        .prefix(prefix)
        .max_keys(max_keys as i32)
        .continuation_token(list_request.next_token.unwrap_or_default())
        .send()
        .await {
        Ok(output) => {
            let mut vectors = Vec::new();
            for object in output.contents.as_deref().unwrap_or_default() {
                if let Some(key) = object.key() {
                    if let Some(vector_id) = key.strip_prefix(&format!("{}/vectors/", list_request.index_name))
                        .and_then(|s| s.strip_suffix(".json")) {
                        
                        if let Ok(get_output) = state.s3.client.get_object().bucket(bucket_name.clone()).key(key).send().await {
                            if let Ok(data) = get_output.body.collect().await {
                                if let Ok(vector_data) = serde_json::from_slice::<serde_json::Value>(&data.into_bytes()) {
                                    vectors.push(json!({
                                        "id": vector_id,
                                        "metadata": vector_data.get("metadata").unwrap_or(&json!({}))
                                    }));
                                }
                            }
                        }
                    }
                }
            }
            
            let response = json!({
                "NextToken": output.next_continuation_token,
                "Vectors": vectors
            });
            
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            tracing::error!("Failed to list vectors for index: {}", list_request.index_name);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list vectors: {}", e)).into_response()
        }
    }
}

async fn s3_get_vectors(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 get-vectors request for bucket: {}, body: {:?}", bucket, body);

    let req: S3GetVectorsRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => {
            tracing::error!("Failed to parse get vectors request: {}", e);
            return (StatusCode::BAD_REQUEST, "Invalid request format").into_response();
        }
    };

    let mut vectors = Vec::new();
    let mut not_found_ids = Vec::new();

    for vector_id in &req.keys {
        let vector_key = format!("{}/vectors/{}.json", req.index_name, vector_id);

        match state.s3.client
            .get_object()
            .bucket(req.vector_bucket_name.clone())
            .key(vector_key)
            .send()
            .await
        {
            Ok(get_output) => {
                match get_output.body.collect().await {
                    Ok(data) => {
                        match serde_json::from_slice::<serde_json::Value>(&data.into_bytes()) {
                            Ok(vector_data) => {
                                let vector_entry = json!({
                                    "Key": vector_id,
                                    "Data": vector_data.get("vector").unwrap_or(&json!({})),
                                    "Metadata": vector_data.get("metadata").unwrap_or(&json!({}))
                                });
                                vectors.push(vector_entry);
                            }
                            Err(e) => {
                                tracing::error!("Failed to parse vector data for {}: {}", vector_id, e);
                                not_found_ids.push(vector_id.clone());
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read vector body for {}: {}", vector_id, e);
                        not_found_ids.push(vector_id.clone());
                    }
                }
            }
            Err(_) => {
                not_found_ids.push(vector_id.clone());
            }
        }
    }

    let response = json!({
        "Vectors": vectors,
        "NotFoundIds": not_found_ids
    });

    (StatusCode::OK, Json(response)).into_response()
}

async fn s3_delete_vectors(bucket: String, body: serde_json::Value, state: AppState) -> impl IntoResponse {
    tracing::info!("S3 delete-vectors request for bucket: {}, body: {:?}", bucket, body);
    
    let delete_request: S3DeleteVectorsRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => {
            tracing::error!("Failed to parse delete vectors request: {}", e);
            return (StatusCode::BAD_REQUEST, "Invalid request format").into_response();
        }
    };
    
    let mut deleted_ids = Vec::new();
    let mut failed_ids = Vec::new();
    
    for vector_id in &delete_request.keys {
        let vector_key = format!("{}/vectors/{}.json", delete_request.index_name, vector_id);
        
        match state.s3.client
            .delete_object()
            .bucket(delete_request.vector_bucket_name.clone())
            .key(vector_key)
            .send()
            .await
        {
            Ok(_) => {
                tracing::info!("Successfully deleted vector: {}", vector_id);
                deleted_ids.push(vector_id.clone());
            }
            Err(e) => {
                tracing::error!("Failed to delete vector {}: {}", vector_id, e);
                failed_ids.push(json!({
                    "id": vector_id,
                    "error": format!("Deletion failed: {}", e)
                }));
            }
        }
    }
    
    let response = json!({
        "DeletedIds": deleted_ids,
        "Errors": failed_ids
    });
    
    (StatusCode::OK, Json(response)).into_response()
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
    // List all buckets from S3
    match state.s3.client.list_buckets().send().await {
        Ok(output) => {
            let mut buckets = Vec::new();
            for bucket in output.buckets() {
                if let Some(name) = bucket.name() {
                    buckets.push(serde_json::json!({
                        "Name": name,
                        "CreationDate": bucket.creation_date()
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "2024-01-01T00:00:00Z".to_string())
                    }));
                }
            }
            
            Json(serde_json::json!({
                "VectorBuckets": buckets
            })).into_response()
        }
        Err(e) => {
            tracing::error!("Failed to list vector buckets: {}", e);
            let error_response = json!({
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
    
    // Extract non-filterable metadata keys from the request
    let non_filterable_keys = req.metadata_configuration
        .as_ref()
        .map(|config| config.non_filterable_metadata_keys.clone())
        .unwrap_or_default();
    
    // Convert S3 format to our internal format
    let create_index_req = CreateIndex {
        name: req.index_name.clone(),
        dim: req.dimension,
        metric: req.distance_metric.to_lowercase(),
        nlist: 16, // Default value
        m: 8,      // Default value  
        nbits: 8,  // Default value
        default_nprobe: Some(8), // Default value
        non_filterable_metadata_keys: non_filterable_keys,
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
    
    // Extract vector IDs/keys before moving req.vectors
    let vector_ids: Vec<String> = req.vectors.iter().map(|v| v.key.clone()).collect();
    
    // Load index configuration to validate metadata sizes
    let index_config = match load_index_configuration(&state.s3, &req.index_name).await {
        Ok(config) => config,
        Err(e) => {
            tracing::error!("Failed to load index configuration for '{}': {}", req.index_name, e);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load index configuration: {}", e)).into_response();
        }
    };
    
    // Validate metadata for each vector
    for (i, vector) in req.vectors.iter().enumerate() {
        if let Err(e) = validate_vector_metadata(&vector.metadata, &index_config) {
            tracing::error!("Metadata validation failed for vector {}: {}", i, e);
            return (StatusCode::BAD_REQUEST, format!("Metadata validation failed for vector {}: {}", i, e)).into_response();
        }
    }
    
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

    tracing::info!("Successfully ingested {} vectors to index '{}'", vector_count, req.index_name);
    
    Json(serde_json::json!({
        "VectorIds": vector_ids
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

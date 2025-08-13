use axum::{response::{IntoResponse, Response}, Json, http::StatusCode};
use serde_json::{json, Value};
use super::{AppState, S3CreateIndexRequest};
use crate::model::*;
use anyhow::Context;

/// CreateIndex - Create a new vector index
pub async fn create(bucket: String, body: Value, state: AppState) -> Response {
    let req: S3CreateIndexRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    let non_filterable_keys = req.metadata_configuration
        .as_ref()
        .map(|config| config.non_filterable_metadata_keys.clone())
        .unwrap_or_default();
    
    let create_index_req = CreateIndex {
        name: req.index_name.clone(),
        dim: req.dimension,
        metric: req.distance_metric.to_lowercase(),
        nlist: 16,
        m: 8,
        nbits: 8,
        default_nprobe: Some(8),
        non_filterable_metadata_keys: non_filterable_keys.clone(),
    };
    
    let config_key = format!("indexes/{}/config.json", create_index_req.name);
    let config_data = match serde_json::to_vec(&create_index_req) {
        Ok(data) => data,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization error: {}", e)).into_response(),
    };
    
    if let Err(e) = state.s3.put_object(&config_key, config_data.into()).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create index: {}", e)).into_response();
    }
    
    // AWS S3 Vectors CreateIndex returns index object per OpenAPI spec
    let body = json!({
        "index": {
            "vectorBucketName": req.vector_bucket_name,
            "indexName": req.index_name,
            "indexArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}/index/{}", req.vector_bucket_name, req.index_name),
            "creationTime": "2025-07-01T13:00:00Z",
            "dataType": "float32",
            "dimension": req.dimension,
            "distanceMetric": req.distance_metric.to_lowercase(),
            "metadataConfiguration": {
                "nonFilterableMetadataKeys": non_filterable_keys
            }
        }
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// ListIndexes - List all indexes in a bucket
pub async fn list(bucket: String, state: AppState) -> Response {
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
                                    let index_summary = json!({
                                        "vectorBucketName": bucket,
                                        "indexName": index_name,
                                        "indexArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}/index/{}", bucket, index_name),
                                        "creationTime": "2025-07-01T13:00:00Z",
                                        "dataType": "float32",
                                        "dimension": config.dim,
                                        "distanceMetric": config.metric.to_lowercase(),
                                        "metadataConfiguration": {
                                            "nonFilterableMetadataKeys": config.non_filterable_metadata_keys
                                        }
                                    });
                                    indexes.push(index_summary);
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to load config for index {}: {}", index_name, e);
                            }
                        }
                    }
                }
            }
            
            // AWS S3 Vectors ListIndexes format per OpenAPI spec
            let body = json!({"indexes": indexes});
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let body = json!({"error": format!("Failed to list indexes: {}", e)});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

/// GetIndex - Get information about a specific index
pub async fn get(bucket: String, body: Value, state: AppState) -> Response {
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    let config_key = format!("indexes/{}/config.json", index_name);
    match state.s3.get_object(&config_key).await {
        Ok(data) => {
            match serde_json::from_slice::<CreateIndex>(&data) {
                Ok(config) => {
                    let vector_prefix = format!("{}/vectors/", index_name);
                    let vector_count = state.s3.list_objects(&vector_prefix).await
                        .map(|objects| objects.len())
                        .unwrap_or(0);
                    
                    // AWS S3 Vectors GetIndex format per OpenAPI spec
                    let body = json!({
                        "index": {
                            "vectorBucketName": bucket,
                            "indexName": index_name,
                            "indexArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}/index/{}", bucket, index_name),
                            "creationTime": "2025-07-01T13:00:00Z",
                            "dataType": "float32",
                            "dimension": config.dim,
                            "distanceMetric": config.metric.to_lowercase(),
                            "metadataConfiguration": {
                                "nonFilterableMetadataKeys": config.non_filterable_metadata_keys
                            }
                        }
                    });
                    (StatusCode::OK, Json(body)).into_response()
                }
                Err(e) => {
                    let body = json!({"error": format!("Failed to parse index config: {}", e)});
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
                }
            }
        }
        Err(e) => {
            let body = json!({"error": format!("Index not found: {}", e)});
            (StatusCode::NOT_FOUND, Json(body)).into_response()
        }
    }
}

/// DeleteIndex - Delete an index and all its vectors
pub async fn delete(bucket: String, body: Value, state: AppState) -> Response {
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    // Delete all objects associated with this index
    let index_prefix = format!("{}/", index_name);
    match state.s3.list_objects(&index_prefix).await {
        Ok(objects) => {
            for object_key in objects {
                let _ = state.s3.delete_object(&object_key).await;
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
            let body = json!({"bucket": bucket, "index": index_name, "deleted": true, "status": "success"});
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let body = json!({"error": format!("Failed to delete index: {}", e)});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

// Direct handlers for S3 API routes
use axum::extract::State;

pub async fn list_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    list(bucket, state).await
}

pub async fn get_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    get(bucket, payload, state).await
}

pub async fn delete_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    delete(bucket, payload, state).await
}

pub async fn create_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    create(bucket, payload, state).await
}

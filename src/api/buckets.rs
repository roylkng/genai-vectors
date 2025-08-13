use axum::{response::{IntoResponse, Response}, Json, http::StatusCode};
use serde_json::json;
use crate::api::AppState;

/// Create a new vector bucket
pub async fn create(bucket: String, state: AppState) -> Response {
    match state.s3.client.create_bucket().bucket(&bucket).send().await {
        Ok(_) => {
            // AWS S3 Vectors CreateVectorBucket returns vectorBucket object
            let body = json!({
                "vectorBucket": {
                    "vectorBucketName": bucket,
                    "vectorBucketArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}", bucket),
                    "creationTime": "2025-07-01T12:34:56Z"
                }
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("BucketAlreadyExists") || msg.contains("BucketAlreadyOwnedByYou") {
                // Return success with vectorBucket object even if already exists
                let body = json!({
                    "vectorBucket": {
                        "vectorBucketName": bucket,
                        "vectorBucketArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}", bucket),
                        "creationTime": "2025-07-01T12:34:56Z"
                    }
                });
                (StatusCode::OK, Json(body)).into_response()
            } else {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Error creating bucket: {}", e)).into_response()
            }
        }
    }
}

/// List all vector buckets
pub async fn list(state: AppState) -> Response {
    match state.s3.client.list_buckets().send().await {
        Ok(output) => {
            let buckets: Vec<serde_json::Value> = output.buckets()
                .iter()
                .map(|bucket| {
                    json!({
                        "vectorBucketName": bucket.name().unwrap_or("unknown"),
                        "vectorBucketArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}", bucket.name().unwrap_or("unknown")),
                        "creationTime": bucket.creation_date()
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "2025-07-01T12:34:56Z".to_string())
                    })
                }).collect();
            
            // AWS S3 Vectors ListVectorBuckets returns camelCase format
            let body = json!({
                "vectorBuckets": buckets
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let body = json!({ "error": format!("Failed to list buckets: {}", e) });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

/// Get a specific vector bucket
pub async fn get(bucket: String, state: AppState) -> Response {
    match state.s3.client.head_bucket().bucket(&bucket).send().await {
        Ok(_) => {
            // AWS S3 Vectors GetVectorBucket response format per OpenAPI spec
            let body = json!({
                "vectorBucket": {
                    "vectorBucketName": bucket,
                    "vectorBucketArn": format!("arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/{}", bucket),
                    "creationTime": "2025-07-01T12:34:56Z"
                }
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let body = json!({ "error": format!("Bucket not found: {}", e) });
            (StatusCode::NOT_FOUND, Json(body)).into_response()
        }
    }
}

/// Delete a vector bucket and all its content
pub async fn delete(bucket: String, state: AppState) -> Response {
    // Delete all objects in bucket
    let _ = state.s3.list_objects("").await.unwrap_or_default().into_iter().map(|key| state.s3.client.delete_object().bucket(&bucket).key(&key).send()).collect::<Vec<_>>();
    match state.s3.client.delete_bucket().bucket(&bucket).send().await {
        Ok(_) => {
            let body = json!({ "bucket": bucket, "deleted": true });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let body = json!({ "error": format!("Failed to delete bucket: {}", e) });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

// Direct handlers for S3 API routes (extract bucket from JSON body)
use axum::extract::State;

pub async fn create_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .or_else(|| payload.get("BucketName"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    create(bucket, state).await
}

pub async fn get_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .or_else(|| payload.get("BucketName"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    get(bucket, state).await
}

pub async fn delete_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .or_else(|| payload.get("BucketName"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    delete(bucket, state).await
}

pub async fn list_direct(
    State(state): State<AppState>,
    Json(_payload): Json<serde_json::Value>
) -> impl IntoResponse {
    list(state).await
}

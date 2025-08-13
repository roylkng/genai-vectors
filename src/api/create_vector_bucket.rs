use axum::{response::{IntoResponse, Response}, Json, http::StatusCode};
use crate::api::AppState;
use serde_json::json;

/// Handler for CreateVectorBucket operation
pub async fn handler(
    bucket: String,
    state: AppState,
) -> Response {
    // Create bucket
    match state.s3.client.create_bucket().bucket(&bucket).send().await {
        Ok(_) => {
            let body = json!({
                "bucketName": bucket,
                "vectorBucket": bucket
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            // Existing bucket is OK
            let msg = e.to_string();
            if msg.contains("BucketAlreadyExists") || msg.contains("BucketAlreadyOwnedByYou") {
                let body = json!({
                    "bucketName": bucket,
                    "vectorBucket": bucket
                });
                (StatusCode::OK, Json(body)).into_response()
            } else {
                let error_body = json!({"error": format!("Failed to create bucket: {}", e)});
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error_body)).into_response()
            }
        }
    }
}

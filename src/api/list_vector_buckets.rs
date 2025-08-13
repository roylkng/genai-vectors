use axum::{response::{IntoResponse, Response}, Json, http::StatusCode};
use crate::api::AppState;
use serde_json::json;

/// Handler for ListVectorBuckets operation
pub async fn handler(
    state: AppState,
) -> Response {
    match state.s3.client.list_buckets().send().await {
        Ok(output) => {
            let buckets: Vec<_> = output.buckets().iter().filter_map(|b| b.name().map(|n| json!({"name": n, "creationDate": b.creation_date().map(|d| d.to_string()).unwrap_or_default()}))).collect();
            let body = json!({"vectorBuckets": buckets});
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let error_body = json!({"error": format!("Failed to list buckets: {}", e)});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_body)).into_response()
        }
    }
}

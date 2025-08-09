use axum::{Router, routing::{post, get}, extract::State, Json, serve, response::IntoResponse, http::StatusCode};
use crate::{model::*, ingest::Ingestor, minio::S3Client};
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Clone)]
struct AppState {
    s3: S3Client,
    ingest: Arc<Ingestor>,
}

// POST /indexes - Create a new index
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
        .route("/indexes", post(create_index))
        .route("/vectors", post(put_vectors))
        .route("/query", post(query))
        .route("/health", get(health))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("API listening on {addr}");
    serve(listener, app).await?;
    Ok(())
}

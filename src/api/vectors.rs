use axum::{response::{IntoResponse, Response}, Json, http::StatusCode};
use serde_json::{json, Value};
use super::{AppState, S3PutVectorsRequest, S3ListVectorsRequest, S3GetVectorsRequest, S3DeleteVectorsRequest, S3QueryVectorsRequest};
use crate::model::*;

// Helper function to extract bucket and index names from request
fn extract_bucket_and_index(
    bucket_name: Option<String>, 
    index_name: Option<String>, 
    index_arn: Option<String>
) -> (String, String) {
    if let Some(arn) = index_arn {
        // Parse ARN: arn:aws:s3vectors:us-east-1:123456789012:vector-bucket/bucket-name/index/index-name
        let parts: Vec<&str> = arn.split('/').collect();
        if parts.len() >= 4 {
            let bucket = parts[parts.len() - 3].to_string();
            let index = parts[parts.len() - 1].to_string();
            return (bucket, index);
        }
    }
    
    let bucket = bucket_name.unwrap_or_else(|| "default-bucket".to_string());
    let index = index_name.unwrap_or_else(|| "default-index".to_string());
    (bucket, index)
}

/// PutVectors - Add vectors to an index
pub async fn put(_bucket: String, body: Value, state: AppState) -> Response {
    // Parse S3 put-vectors request
    let req: S3PutVectorsRequest = match serde_json::from_value(body.clone()) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response(),
    };
    
    let (_bucket_name, index_name) = extract_bucket_and_index(
        req.vector_bucket_name.clone(),
        req.index_name.clone(),
        req.index_arn.clone()
    );
    
    // Convert to internal format and ingest
    let vectors: Vec<VectorRecord> = req.vectors.into_iter().filter_map(|v| {
        let id = v.get("key").and_then(|k| k.as_str())?;
        let data = v.get("data").and_then(|d| d.get("float32")).and_then(|f| f.as_array())?;
        let embedding: Vec<f32> = data.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
        let metadata = v.get("metadata").cloned().unwrap_or(json!({}));
        
        Some(VectorRecord {
            id: id.to_string(),
            embedding,
            meta: metadata,
            created_at: chrono::Utc::now(),
        })
    }).collect();

    let bucket_for_ingest = req.vector_bucket_name.as_deref().unwrap_or("default-bucket");
    if let Err(e) = state.ingest.append(vectors, &index_name).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Ingestion failed: {}", e)).into_response();
    }

    // Store individual vector JSON objects for listing/getting
    if let Some(vectors_json) = body.get("vectors").and_then(|v| v.as_array()) {
        for vec_json in vectors_json {
            if let Some(key) = vec_json.get("key").and_then(|v| v.as_str()) {
                let object_key = format!("{}/vectors/{}.json", index_name, key);
                if let Ok(data_bytes) = serde_json::to_vec(vec_json) {
                    let _ = state.s3.client
                        .put_object()
                        .bucket(bucket_for_ingest)
                        .key(&object_key)
                        .body(aws_sdk_s3::primitives::ByteStream::from(data_bytes))
                        .send()
                        .await;
                }
            }
        }
    }
    
    // Trigger indexing
    let _ = crate::indexer::run_once().await;
    
    // AWS S3 Vectors PutVectors returns empty response per OpenAPI spec
    let body = json!({});
    (StatusCode::OK, Json(body)).into_response()
}

/// ListVectors - List vectors in an index
pub async fn list(_bucket: String, body: Value, state: AppState) -> Response {
    // Get parameters from body directly
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default-index");
    
    let bucket_name = body.get("vectorBucketName")
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket");
    
    let return_data = body.get("returnData").and_then(|v| v.as_bool()).unwrap_or(false);
    let return_metadata = body.get("returnMetadata").and_then(|v| v.as_bool()).unwrap_or(false);
    
    let mut vectors = Vec::new();
    let prefix = format!("{}/vectors/", index_name);
    
    // List vectors from S3 storage  
    match state.s3.client
        .list_objects_v2()
        .bucket(bucket_name)
        .prefix(&prefix)
        .send()
        .await 
    {
        Ok(output) => {
            for object in output.contents() {
                if let Some(key) = object.key() {
                    if let Some(vector_id) = key.strip_prefix(&prefix).and_then(|s| s.strip_suffix(".json")) {
                        // Try to load the vector data
                        if let Ok(obj_output) = state.s3.client.get_object().bucket(bucket_name).key(key).send().await {
                            if let Ok(data) = obj_output.body.collect().await {
                                if let Ok(json_val) = serde_json::from_slice::<Value>(&data.into_bytes()) {
                                    let mut vector_entry = json!({
                                        "key": vector_id
                                    });
                                    
                                    if return_data {
                                        vector_entry["data"] = json_val.get("data").unwrap_or(&json!({})).clone();
                                    }
                                    
                                    if return_metadata {
                                        vector_entry["metadata"] = json_val.get("metadata").unwrap_or(&json!({})).clone();
                                    }
                                    
                                    vectors.push(vector_entry);
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            tracing::error!("Failed to list vectors: {}", e);
        }
    }
    
    // AWS S3 Vectors ListVectors format per OpenAPI spec
    let body = json!({
        "vectors": vectors
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// GetVectors - Retrieve specific vectors by ID
pub async fn get(_bucket: String, body: Value, state: AppState) -> Response {
    let req: S3GetVectorsRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(_e) => return (StatusCode::BAD_REQUEST, "Invalid request format").into_response(),
    };
    
    let (bucket_name, index_name) = extract_bucket_and_index(
        req.vector_bucket_name,
        req.index_name,
        req.index_arn
    );
    
    let mut vectors = Vec::new();
    
    for vector_id in &req.keys {
        let key = format!("{}/vectors/{}.json", index_name, vector_id);
        match state.s3.client.get_object().bucket(&bucket_name).key(&key).send().await {
            Ok(output) => match output.body.collect().await {
                Ok(data) => if let Ok(json_val) = serde_json::from_slice::<Value>(&data.into_bytes()) {
                    let mut entry = json!({
                        "key": vector_id
                    });
                    
                    if req.return_data {
                        entry["data"] = json_val.get("data").unwrap_or(&json!({})).clone();
                    }
                    
                    if req.return_metadata {
                        entry["metadata"] = json_val.get("metadata").unwrap_or(&json!({})).clone();
                    }
                    
                    vectors.push(entry);
                } else {
                    // Vector exists but couldn't parse - still include key
                    vectors.push(json!({"key": vector_id}));
                },
                Err(_) => {
                    // Vector exists but couldn't read - still include key
                    vectors.push(json!({"key": vector_id}));
                },
            },
            Err(_) => {
                // Vector doesn't exist - skip it (don't add to results)
            }
        }
    }
    
    // AWS S3 Vectors GetVectors format per OpenAPI spec
    let body = json!({"vectors": vectors});
    (StatusCode::OK, Json(body)).into_response()
}

/// DeleteVectors - Delete specific vectors by ID
pub async fn delete(_bucket: String, body: Value, state: AppState) -> Response {
    let delete_request: S3DeleteVectorsRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(_e) => return (StatusCode::BAD_REQUEST, "Invalid request format").into_response(),
    };
    
    let (bucket_name, index_name) = extract_bucket_and_index(
        delete_request.vector_bucket_name,
        delete_request.index_name,
        delete_request.index_arn
    );
    
    for vector_id in &delete_request.keys {
        let vector_key = format!("{}/vectors/{}.json", index_name, vector_id);
        let _ = state.s3.client
            .delete_object()
            .bucket(&bucket_name)
            .key(&vector_key)
            .send()
            .await;
    }
    
    // AWS S3 Vectors DeleteVectors format per OpenAPI spec (empty response)
    let body = json!({});
    (StatusCode::OK, Json(body)).into_response()
}

/// QueryVectors - Search for similar vectors
/// QueryVectors - Search for similar vectors  
pub async fn query(_bucket: String, body: Value, state: AppState) -> Response {
    // Get parameters from body directly  
    let index_name = body.get("indexName")
        .and_then(|v| v.as_str())
        .unwrap_or("default-index");
    
    let query_vector = body.get("queryVector")
        .or_else(|| body.get("vector"))
        .and_then(|v| {
            // Handle both direct array and nested format
            if let Some(arr) = v.as_array() {
                // Direct array format: [1.0, 2.0, 3.0]
                arr.iter()
                    .map(|v| v.as_f64())
                    .collect::<Option<Vec<f64>>>()
            } else if let Some(data_obj) = v.as_object() {
                // Nested format: {"float32": [1.0, 2.0, 3.0]} or {"data": {"float32": [...]}}
                if let Some(float32_arr) = data_obj.get("float32").and_then(|v| v.as_array()) {
                    float32_arr.iter()
                        .map(|v| v.as_f64())
                        .collect::<Option<Vec<f64>>>()
                } else if let Some(data_nested) = data_obj.get("data").and_then(|v| v.as_object()) {
                    if let Some(float32_arr) = data_nested.get("float32").and_then(|v| v.as_array()) {
                        float32_arr.iter()
                            .map(|v| v.as_f64())
                            .collect::<Option<Vec<f64>>>()
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        });
    
    let top_k = body.get("topK")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;
    
    let return_data = body.get("returnData").and_then(|v| v.as_bool()).unwrap_or(false);
    let return_metadata = body.get("returnMetadata").and_then(|v| v.as_bool()).unwrap_or(false);
    let metadata_filter = body.get("metadataFilter");
    
    if query_vector.is_none() {
        return (StatusCode::BAD_REQUEST, "Query vector is required").into_response();
    }
    
    let query_vector = query_vector.unwrap();
    
    let query_req = QueryRequest {
        index: index_name.to_string(),
        embedding: query_vector.into_iter().map(|f| f as f32).collect(),
        topk: top_k,
        nprobe: None,
        filter: metadata_filter.cloned(),
    };
    
    match crate::query::search(state.s3, query_req).await {
        Ok(resp) => {
            let empty_vec = vec![];
            let results = resp.get("results").and_then(|r| r.as_array()).unwrap_or(&empty_vec);
            let s3_results: Vec<Value> = results.iter().map(|result| {
                let mut entry = json!({
                    "key": result.get("id").unwrap_or(&json!("unknown"))
                });
                
                // Always include distance/score in query results
                entry["distance"] = result.get("score").unwrap_or(&json!(0.0)).clone();
                
                if return_metadata {
                    entry["metadata"] = result.get("metadata").unwrap_or(&json!({})).clone();
                }
                
                if return_data {
                    entry["data"] = result.get("embedding").unwrap_or(&json!({})).clone();
                }
                
                entry
            }).collect();
            
            // AWS S3 Vectors QueryVectors format per OpenAPI spec
            let body = json!({"vectors": s3_results});
            (StatusCode::OK, Json(body)).into_response()
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Query failed: {}", e)).into_response(),
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
    
    list(bucket, payload, state).await
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

pub async fn put_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    put(bucket, payload, state).await
}

pub async fn query_direct(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let bucket = payload.get("vectorBucketName")
        .or_else(|| payload.get("Bucket"))
        .or_else(|| payload.get("bucket"))
        .and_then(|v| v.as_str())
        .unwrap_or("default-bucket")
        .to_string();
    
    query(bucket, payload, state).await
}

use crate::{minio::S3Client, model::*, indexer};
use anyhow::Result;
use arrow::array::{ListArray, RecordBatch, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Float32Type, Schema, TimeUnit};
use bytes::Bytes;
use chrono::Utc;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::{Arc, Mutex};
use tokio::{fs, io::AsyncWriteExt, time::Instant};

pub const SLICE_ROW_LIMIT: usize = 5000;
pub const SLICE_AGE_LIMIT_S: u64 = 30;

pub struct Buffer {
    rows: Vec<VectorRecord>,
    first_seen: Instant,
    format: SliceFormat,
}

#[derive(Clone, Debug)]
pub enum SliceFormat {
    JsonLines,
    Parquet,
}

impl Buffer {
    fn new(format: SliceFormat) -> Self {
        Self {
            rows: Vec::new(),
            first_seen: Instant::now(),
            format,
        }
    }
}

pub struct Ingestor {
    buf: Arc<Mutex<Buffer>>,
    s3: S3Client,
    bucket: String,
    slice_format: SliceFormat,
}

impl Ingestor {
    pub fn new(s3: S3Client, bucket: String) -> Self {
        let slice_format = match std::env::var("SLICE_FORMAT").as_deref() {
            Ok("parquet") => SliceFormat::Parquet,
            Ok("jsonl") | Ok("ndjson") | _ => SliceFormat::JsonLines, // Default to JSONL
        };
        tracing::info!("Ingestor configured with slice format: {:?}", slice_format);
        Self {
            buf: Arc::new(Mutex::new(Buffer::new(slice_format.clone()))),
            s3,
            bucket,
            slice_format,
        }
    }

    pub async fn append(&self, vecs: Vec<VectorRecord>, index: &str) -> anyhow::Result<()> {
        let mut wal_bytes = Vec::new();
        for rec in &vecs {
            wal_bytes.extend(serde_json::to_vec(rec)?);
            wal_bytes.push(b'\n');
        }
        self.s3
            .append_object(&self.bucket, "wal/current.ndjson", Bytes::from(wal_bytes))
            .await?;

        let slice_rows = {
            let mut guard = self.buf.lock().unwrap();
            if guard.rows.is_empty() {
                guard.first_seen = Instant::now();
            }
            guard.rows.extend(vecs);

            if guard.rows.len() >= SLICE_ROW_LIMIT
                || guard.first_seen.elapsed().as_secs() >= SLICE_AGE_LIMIT_S
            {
                Some(std::mem::take(&mut guard.rows))
            } else {
                None
            }
        };

        if let Some(rows) = slice_rows {
            self.write_slice(rows, index).await?;
        }
        Ok(())
    }

    async fn write_slice(&self, rows: Vec<VectorRecord>, index: &str) -> Result<()> {
        let ts = Utc::now().format("%Y%m%dT%H%M%S%3f");
        
        let (key, local_path) = match self.slice_format {
            SliceFormat::JsonLines => {
                let key = format!("staged/{}/slice-{}.jsonl", index, ts);
                let local_path = "/tmp/slice.jsonl";
                let mut tmp = fs::File::create(local_path).await?;
                for r in &rows {
                    tmp.write_all(serde_json::to_string(r)?.as_bytes()).await?;
                    tmp.write_u8(b'\n').await?;
                }
                tmp.sync_all().await?;
                (key, local_path.to_string())
            }
            SliceFormat::Parquet => {
                let key = format!("staged/{}/slice-{}.parquet", index, ts);
                let local_path = self.write_parquet_slice(&rows).await?;
                (key, local_path)
            }
        };

        self.s3.put_file(&self.bucket, &key, &local_path).await?;
        tokio::fs::remove_file(&local_path).await?;

        tracing::debug!("Wrote {} vectors to slice: {}", rows.len(), key);

        // Enhanced callback indexing - trigger immediately after slice upload
        let s3_clone = self.s3.clone();
        let key_clone = key.clone();
        tokio::spawn(async move {
            tracing::info!("Triggering immediate indexing for slice: {}", key_clone);
            // Add a small delay to ensure object is fully written
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            if let Err(e) = indexer::trigger_indexing_for_slice(s3_clone, key_clone).await {
                tracing::error!("Failed to trigger indexing for slice: {}", e);
            } else {
                tracing::info!("Successfully triggered indexing callback");
            }
        });
        
        Ok(())
    }

    async fn write_parquet_slice(&self, rows: &[VectorRecord]) -> Result<String> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new("meta", DataType::Utf8, true),
            Field::new(
                "created_at",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
        ]));

        let ids: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
        let embeddings_iter = rows.iter().map(|r| Some(r.embedding.iter().map(|&f| Some(f)).collect::<Vec<_>>()));
        let metas: Vec<String> = rows.iter().map(|r| r.meta.to_string()).collect();
        let created_ats: Vec<i64> = rows
            .iter()
            .map(|r| r.created_at.timestamp_nanos_opt().unwrap_or(0))
            .collect();

        let id_array = Arc::new(StringArray::from(ids));
        let embedding_array = Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings_iter,
        ));
        let meta_array = Arc::new(StringArray::from(metas));
        let created_at_array = Arc::new(TimestampNanosecondArray::from(created_ats));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array, embedding_array, meta_array, created_at_array],
        )?;

        let local_path = format!("/tmp/slice-{}.parquet", Utc::now().timestamp_nanos_opt().unwrap_or(0));
        let file = File::create(&local_path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        
        Ok(local_path)
    }
}

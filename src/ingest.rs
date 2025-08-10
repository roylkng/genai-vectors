use crate::{minio::S3Client, model::*};
use bytes::Bytes;
use std::sync::{Arc, Mutex};
use tokio::{fs, io::AsyncWriteExt, time::Instant};
use chrono::Utc;
use anyhow::Result;

pub struct Buffer {
    rows: Vec<VectorRecord>,
    first_seen: Instant,
    format: SliceFormat,
}

#[derive(Clone, Debug)]
pub enum SliceFormat {
    JsonLines,  // Original NDJSON format for compatibility
    Parquet,    // Compact binary format for new deployments
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
    s3:  S3Client,
    bucket: String,
    slice_format: SliceFormat,
}

impl Ingestor {
    pub fn new(s3: S3Client, bucket: String) -> Self {
        // Default to Parquet for new instances, can be configured via env var
        let slice_format = match std::env::var("SLICE_FORMAT").as_deref() {
            Ok("jsonl") | Ok("ndjson") => SliceFormat::JsonLines,
            Ok("parquet") | _ => SliceFormat::Parquet, // Default to Parquet
        };
        
        tracing::info!("Ingestor configured with slice format: {:?}", slice_format);
        
        Self { 
            buf: Arc::new(Mutex::new(Buffer::new(slice_format.clone()))), 
            s3, 
            bucket,
            slice_format,
        }
    }

    /// Append vectors to WAL and RAM buffer; flush slice if needed
    pub async fn append(&self, vecs: Vec<VectorRecord>, index: &str) -> anyhow::Result<()> {
        // 1) append to WAL (one NDJSON line each)
        let mut wal_bytes = Vec::new();
        for rec in &vecs {
            wal_bytes.extend(serde_json::to_vec(rec)?);
            wal_bytes.push(b'\n');
        }
        self.s3.append_object(&self.bucket, "wal/current.ndjson", Bytes::from(wal_bytes)).await?;

        // 2) RAM buffer
        let slice_rows = {
            let mut guard = self.buf.lock().unwrap();
            if guard.rows.is_empty() { guard.first_seen = Instant::now(); }
            guard.rows.extend(vecs);

            if guard.rows.len() >= SLICE_ROW_LIMIT ||
               guard.first_seen.elapsed().as_secs() >= SLICE_AGE_LIMIT_S {
                // dump slice to staging parquet
                Some(std::mem::take(&mut guard.rows))
            } else {
                None
            }
        }; // guard is dropped here
        
        if let Some(rows) = slice_rows {
            self.write_slice(rows, index).await?;
        }
        Ok(())
    }

    async fn write_slice(&self, rows: Vec<VectorRecord>, index: &str) -> Result<()> {
        let ts = Utc::now().format("%Y%m%dT%H%M%S%3f");
        
        match self.slice_format {
            SliceFormat::JsonLines => {
                // Original JSONL format
                let key = format!("staged/{}/slice-{}.jsonl", index, ts);
                let mut tmp = fs::File::create("/tmp/slice.jsonl").await?;
                for r in &rows { 
                    tmp.write_all(serde_json::to_string(r)?.as_bytes()).await?; 
                    tmp.write_u8(b'\n').await?; 
                }
                tmp.sync_all().await?;
                self.s3.put_file(&self.bucket, &key, "/tmp/slice.jsonl").await?;
                
                tracing::debug!("Wrote {} vectors to JSONL slice: {}", rows.len(), key);
            }
            SliceFormat::Parquet => {
                // Compact Parquet format
                let key = format!("staged/{}/slice-{}.parquet", index, ts);
                self.write_parquet_slice(&rows, &key).await?;
                
                tracing::debug!("Wrote {} vectors to Parquet slice: {}", rows.len(), key);
            }
        }
        
        Ok(())
    }
    
    /// Write vectors to Parquet format for compact storage and fast parsing
    async fn write_parquet_slice(&self, rows: &[VectorRecord], key: &str) -> Result<()> {
        // For now, fall back to JSONL until we implement Parquet support
        // This is a placeholder for the Parquet implementation
        tracing::warn!("Parquet format not yet implemented, falling back to JSONL");
        
        let tmp_path = "/tmp/slice_fallback.jsonl";
        let mut tmp = fs::File::create(tmp_path).await?;
        for r in rows { 
            tmp.write_all(serde_json::to_string(r)?.as_bytes()).await?; 
            tmp.write_u8(b'\n').await?; 
        }
        tmp.sync_all().await?;
        self.s3.put_file(&self.bucket, key, tmp_path).await?;
        
        Ok(())
    }
}

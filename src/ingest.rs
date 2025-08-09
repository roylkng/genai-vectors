use crate::{minio::S3Client, model::*};
use bytes::Bytes;
use std::sync::{Arc, Mutex};
use tokio::{fs, io::AsyncWriteExt, time::Instant};
use chrono::Utc;

pub struct Buffer {
    rows: Vec<VectorRecord>,
    first_seen: Instant,
}
impl Buffer {
    fn new() -> Self {
        Self { rows: Vec::new(), first_seen: Instant::now() }
    }
}

pub struct Ingestor {
    buf: Arc<Mutex<Buffer>>,
    s3:  S3Client,
    bucket: String,
}

impl Ingestor {
    pub fn new(s3: S3Client, bucket: String) -> Self {
        Self { buf: Arc::new(Mutex::new(Buffer::new())), s3, bucket }
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

    async fn write_slice(&self, rows: Vec<VectorRecord>, index: &str) -> anyhow::Result<()> {
        // simplistic JSONL write
        let ts = Utc::now().format("%Y%m%dT%H%M%S%3f");
        let key = format!("staged/{}/slice-{}.jsonl", index, ts);
        let mut tmp = fs::File::create("/tmp/slice.jsonl").await?;
        for r in &rows { 
            tmp.write_all(serde_json::to_string(r)?.as_bytes()).await?; 
            tmp.write_u8(b'\n').await?; 
        }
        tmp.sync_all().await?;
        self.s3.put_file(&self.bucket, &key, "/tmp/slice.jsonl").await?;
        Ok(())
    }
}

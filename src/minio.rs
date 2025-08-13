use anyhow::{Context, Result};
use aws_config::Region;
use aws_sdk_s3::{config::Builder, Client, primitives::ByteStream};
use bytes::Bytes;
use tokio::fs;

#[derive(Clone)]
pub struct S3Client {
    pub client: Client,
    bucket: String,
}

impl S3Client {
    pub async fn from_env() -> Result<Self> {
        let endpoint = std::env::var("AWS_ENDPOINT_URL")
            .unwrap_or_else(|_| "http://minio:9000".to_string());
        let access_key = std::env::var("AWS_ACCESS_KEY_ID")
            .unwrap_or_else(|_| "minioadmin".to_string());
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .unwrap_or_else(|_| "minioadmin".to_string());
        let bucket_name = std::env::var("VEC_BUCKET")
            .unwrap_or_else(|_| "vectors".to_string());

        let creds = aws_sdk_s3::config::Credentials::new(
            access_key,
            secret_key,
            None,
            None,
            "static",
        );

        let config = Builder::new()
            .endpoint_url(endpoint)
            .region(Region::new("us-east-1"))
            .credentials_provider(creds)
            .force_path_style(true)
            .build();

        let client = Client::from_conf(config);

        // Ensure bucket exists
        let _response = client
            .create_bucket()
            .bucket(&bucket_name)
            .send()
            .await;

        Ok(Self {
            client,
            bucket: bucket_name,
        })
    }

    pub async fn put_object(&self, key: &str, data: Bytes) -> Result<()> {
        tracing::info!("🔍 MinIO put_object attempt - bucket: {}, key: {}, data_size: {}", &self.bucket, key, data.len());
        
        match self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(data))
            .send()
            .await
        {
            Ok(_) => {
                tracing::info!("✅ MinIO put_object success - key: {}", key);
                Ok(())
            },
            Err(e) => {
                tracing::error!("❌ MinIO put_object failed - key: {}, detailed_error: {:?}", key, e);
                Err(anyhow::anyhow!("Failed to put object {}: {:?}", key, e))
            }
        }
    }

    pub async fn get_object(&self, key: &str) -> Result<Bytes> {
        let response = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .context("Failed to get object")?;

        let data = response
            .body
            .collect()
            .await
            .context("Failed to read object body")?;

        Ok(data.into_bytes())
    }

    pub async fn put_file(&self, _bucket: &str, key: &str, file_path: &str) -> Result<()> {
        let data = fs::read(file_path).await
            .context("Failed to read file")?;
        self.put_object(key, Bytes::from(data)).await
    }

    pub async fn append_object(&self, _bucket: &str, key: &str, data: Bytes) -> Result<()> {
        // For simplicity, we'll just put the object (overwrite)
        // In production, you'd want proper append logic
        self.put_object(key, data).await
    }

    pub async fn list_buckets(&self) -> Result<Vec<String>> {
        let response = self.client
            .list_buckets()
            .send()
            .await
            .context("Failed to list buckets")?;
        
        let mut bucket_names = Vec::new();
        if let Some(buckets) = response.buckets {
            for bucket in buckets {
                if let Some(name) = bucket.name {
                    bucket_names.push(name);
                }
            }
        }
        Ok(bucket_names)
    }

    pub async fn list_objects(&self, prefix: &str) -> Result<Vec<String>> {
        let response = self.client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(prefix)
            .send()
            .await
            .context("Failed to list objects")?;
        
        let mut keys = Vec::new();
        if let Some(contents) = response.contents {
            for object in contents {
                if let Some(key) = object.key {
                    keys.push(key);
                }
            }
        }
        Ok(keys)
    }

    pub async fn delete_object(&self, key: &str) -> Result<()> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .context("Failed to delete object")?;
        Ok(())
    }
}

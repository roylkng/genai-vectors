mod api;
mod faiss_utils;
mod ingest;
mod indexer;
mod metadata_filter;
mod metrics;
mod query;
mod model;
mod minio;

use clap::{Parser, Subcommand};
use tracing::Level;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run REST gateway (CreateIndex / PutVectors / QueryVectors)
    Api,
    /// Run indexer loop once (train/merge) – scheduled via CronJob
    Indexer,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    
    // Show which backend is being used
    tracing::info!("🚀 Vector Database Starting");
    tracing::info!("Backend: Real Faiss (IVF-PQ)");
    
    // Initialize metrics collection
    metrics::get_metrics_collector().start_monitoring();
    tracing::info!("Metrics collection started");
    
    match Cli::parse().cmd {
        Cmd::Api => api::run().await?,
        Cmd::Indexer => indexer::run_once().await?,
    }
    Ok(())
}

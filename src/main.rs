mod api;
mod ingest;
mod indexer;
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
    match Cli::parse().cmd {
        Cmd::Api => api::run().await?,
        Cmd::Indexer => indexer::run_once().await?,
    }
    Ok(())
}

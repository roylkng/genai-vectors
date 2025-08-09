use std::time::Instant;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Comprehensive performance metrics collection for vector database operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub operation_type: OperationType,
    pub index_name: String,
    pub vector_count: usize,
    pub dimension: usize,
    
    // Timing metrics
    pub duration_ms: f64,
    pub cpu_time_ms: f64,
    
    // Index configuration
    pub index_config: IndexConfig,
    
    // Memory metrics
    pub memory_usage_mb: f64,
    pub peak_memory_mb: f64,
    
    // Query-specific metrics
    pub query_metrics: Option<QueryMetrics>,
    
    // Indexing-specific metrics
    pub indexing_metrics: Option<IndexingMetrics>,
    
    // Error metrics
    pub error_count: u32,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    IndexCreation,
    VectorInsertion,
    IndexTraining,
    VectorQuery,
    ShardCreation,
    IndexOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub nlist: u32,
    pub nprobe: Option<u32>,
    pub m: u32,
    pub nbits: u32,
    pub metric: String,
    pub shard_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub topk: usize,
    pub nprobe_used: u32,
    pub shards_searched: usize,
    pub vectors_scanned: usize,
    pub result_count: usize,
    pub latency_breakdown: LatencyBreakdown,
    pub recall_estimate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub index_load_ms: f64,
    pub search_ms: f64,
    pub result_merge_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingMetrics {
    pub vectors_processed: usize,
    pub training_time_ms: f64,
    pub insertion_time_ms: f64,
    pub compression_ratio: f64,
    pub index_size_mb: f64,
    pub throughput_vectors_per_sec: f64,
}

/// Thread-safe performance metrics collector
pub struct MetricsCollector {
    metrics: std::sync::Arc<std::sync::Mutex<Vec<PerformanceMetrics>>>,
    current_operation: std::sync::Arc<std::sync::Mutex<Option<OperationTracker>>>,
}

struct OperationTracker {
    operation_type: OperationType,
    index_name: String,
    start_time: Instant,
    start_memory: u64,
    config: IndexConfig,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            current_operation: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
    
    /// Start tracking a new operation
    pub fn start_operation(&self, 
                          operation_type: OperationType, 
                          index_name: String, 
                          config: IndexConfig) {
        let tracker = OperationTracker {
            operation_type,
            index_name,
            start_time: Instant::now(),
            start_memory: Self::get_memory_usage_mb() as u64,
            config,
        };
        
        let mut current = self.current_operation.lock().unwrap();
        *current = Some(tracker);
    }
    
    /// Finish tracking and record metrics
    pub fn finish_operation(&self, additional_data: HashMap<String, f64>) {
        let mut current = self.current_operation.lock().unwrap();
        if let Some(tracker) = current.take() {
            let duration = tracker.start_time.elapsed();
            let current_memory = Self::get_memory_usage_mb();
            
            let metrics = PerformanceMetrics {
                timestamp: Utc::now(),
                operation_type: tracker.operation_type,
                index_name: tracker.index_name,
                vector_count: *additional_data.get("vector_count").unwrap_or(&0.0) as usize,
                dimension: *additional_data.get("dimension").unwrap_or(&0.0) as usize,
                duration_ms: duration.as_secs_f64() * 1000.0,
                cpu_time_ms: Self::get_cpu_time_ms(),
                index_config: tracker.config,
                memory_usage_mb: current_memory,
                peak_memory_mb: current_memory.max(tracker.start_memory as f64),
                query_metrics: None,
                indexing_metrics: None,
                error_count: *additional_data.get("error_count").unwrap_or(&0.0) as u32,
                error_rate: *additional_data.get("error_rate").unwrap_or(&0.0),
            };
            
            let mut metrics_vec = self.metrics.lock().unwrap();
            metrics_vec.push(metrics);
        }
    }
    
    /// Record query-specific metrics
    pub fn record_query_metrics(&self, query_metrics: QueryMetrics) {
        let mut metrics_vec = self.metrics.lock().unwrap();
        if let Some(last_metric) = metrics_vec.last_mut() {
            last_metric.query_metrics = Some(query_metrics);
        }
    }
    
    /// Record indexing-specific metrics
    pub fn record_indexing_metrics(&self, indexing_metrics: IndexingMetrics) {
        let mut metrics_vec = self.metrics.lock().unwrap();
        if let Some(last_metric) = metrics_vec.last_mut() {
            last_metric.indexing_metrics = Some(indexing_metrics);
        }
    }
    
    /// Get all collected metrics
    pub fn get_metrics(&self) -> Vec<PerformanceMetrics> {
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }
    
    /// Track a simple metric value
    pub fn track_metric(&self, name: &str, value: f64) {
        // For now, just log the metric. In a production system, you'd want to
        // store these in a time-series database or metrics collection system
        tracing::debug!("Metric {}: {}", name, value);
        
        // You could also store these in a separate hashmap for simple metrics
        // that don't need the full PerformanceMetrics structure
    }
    
    /// Start monitoring background processes
    pub fn start_monitoring(&self) {
        tracing::info!("Performance monitoring started");
        // In a production system, you might start background threads here
        // for metrics aggregation, alerts, etc.
    }
    
    /// Get metrics summary for a specific operation type
    pub fn get_summary(&self, operation_type: OperationType) -> MetricsSummary {
        let metrics = self.metrics.lock().unwrap();
        let filtered: Vec<&PerformanceMetrics> = metrics.iter()
            .filter(|m| std::mem::discriminant(&m.operation_type) == std::mem::discriminant(&operation_type))
            .collect();
        
        if filtered.is_empty() {
            return MetricsSummary::default();
        }
        
        let durations: Vec<f64> = filtered.iter().map(|m| m.duration_ms).collect();
        let memory_usage: Vec<f64> = filtered.iter().map(|m| m.memory_usage_mb).collect();
        
        MetricsSummary {
            operation_count: filtered.len(),
            avg_duration_ms: durations.iter().sum::<f64>() / durations.len() as f64,
            min_duration_ms: durations.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_duration_ms: durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            p50_duration_ms: Self::percentile(&durations, 50.0),
            p95_duration_ms: Self::percentile(&durations, 95.0),
            p99_duration_ms: Self::percentile(&durations, 99.0),
            avg_memory_mb: memory_usage.iter().sum::<f64>() / memory_usage.len() as f64,
            peak_memory_mb: memory_usage.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            total_errors: filtered.iter().map(|m| m.error_count).sum(),
            avg_error_rate: filtered.iter().map(|m| m.error_rate).sum::<f64>() / filtered.len() as f64,
        }
    }
    
    /// Export metrics to JSON file
    pub fn export_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let metrics = self.get_metrics();
        let json = serde_json::to_string_pretty(&metrics)?;
        std::fs::write(filename, json)?;
        tracing::info!("Exported {} metrics to {}", metrics.len(), filename);
        Ok(())
    }
    
    /// Clear all collected metrics
    pub fn clear(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.clear();
    }
    
    // Helper methods
    fn get_memory_usage_mb() -> f64 {
        // This is a simplified implementation
        // In production, you'd use proper memory monitoring
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb) = line.split_whitespace().nth(1) {
                            if let Ok(kb_val) = kb.parse::<f64>() {
                                return kb_val / 1024.0; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback estimation
        0.0
    }
    
    fn get_cpu_time_ms() -> f64 {
        // Simplified CPU time measurement
        // In production, you'd use proper CPU monitoring
        0.0
    }
    
    fn percentile(data: &[f64], percentile: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsSummary {
    pub operation_count: usize,
    pub avg_duration_ms: f64,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub p50_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p99_duration_ms: f64,
    pub avg_memory_mb: f64,
    pub peak_memory_mb: f64,
    pub total_errors: u32,
    pub avg_error_rate: f64,
}

/// Global metrics collector instance
pub static METRICS_COLLECTOR: std::sync::OnceLock<MetricsCollector> = std::sync::OnceLock::new();

/// Get the global metrics collector
pub fn get_metrics_collector() -> &'static MetricsCollector {
    METRICS_COLLECTOR.get_or_init(|| MetricsCollector::new())
}

/// Simple macro for measuring operation duration
#[macro_export]
macro_rules! measure_operation {
    ($operation_name:expr) => {
        {
            let start = std::time::Instant::now();
            struct OperationTimer {
                start: std::time::Instant,
                name: String,
            }
            
            impl Drop for OperationTimer {
                fn drop(&mut self) {
                    let duration = self.start.elapsed().as_millis() as f64;
                    crate::metrics::get_metrics_collector().track_metric(
                        &format!("{}_duration_ms", self.name), 
                        duration
                    );
                }
            }
            
            OperationTimer { start, name: $operation_name.to_string() }
        }
    };
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_detailed_logging: bool,
    pub export_interval_seconds: u64,
    pub max_metrics_retention: usize,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub max_query_latency_ms: f64,
    pub max_indexing_time_ms: f64,
    pub max_memory_usage_mb: f64,
    pub max_error_rate: f64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_detailed_logging: true,
            export_interval_seconds: 60,
            max_metrics_retention: 10000,
            alert_thresholds: AlertThresholds {
                max_query_latency_ms: 1000.0,
                max_indexing_time_ms: 300000.0, // 5 minutes
                max_memory_usage_mb: 8192.0,    // 8GB
                max_error_rate: 0.05,           // 5%
            },
        }
    }
}

/// Automated performance monitoring and alerting
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    last_export: std::sync::Arc<std::sync::Mutex<Instant>>,
}

impl PerformanceMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            last_export: std::sync::Arc::new(std::sync::Mutex::new(Instant::now())),
        }
    }
    
    /// Check metrics against thresholds and generate alerts
    pub fn check_alerts(&self) -> Vec<Alert> {
        let collector = get_metrics_collector();
        let mut alerts = Vec::new();
        
        // Check query latency
        let query_summary = collector.get_summary(OperationType::VectorQuery);
        if query_summary.p95_duration_ms > self.config.alert_thresholds.max_query_latency_ms {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                message: format!("High query latency: P95 = {:.1}ms", query_summary.p95_duration_ms),
                metric_type: "query_latency".to_string(),
                current_value: query_summary.p95_duration_ms,
                threshold: self.config.alert_thresholds.max_query_latency_ms,
            });
        }
        
        // Check indexing performance
        let indexing_summary = collector.get_summary(OperationType::IndexTraining);
        if indexing_summary.avg_duration_ms > self.config.alert_thresholds.max_indexing_time_ms {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                message: format!("Slow indexing: Avg = {:.1}ms", indexing_summary.avg_duration_ms),
                metric_type: "indexing_time".to_string(),
                current_value: indexing_summary.avg_duration_ms,
                threshold: self.config.alert_thresholds.max_indexing_time_ms,
            });
        }
        
        // Check memory usage
        if query_summary.peak_memory_mb > self.config.alert_thresholds.max_memory_usage_mb {
            alerts.push(Alert {
                severity: AlertSeverity::Critical,
                message: format!("High memory usage: {:.1}MB", query_summary.peak_memory_mb),
                metric_type: "memory_usage".to_string(),
                current_value: query_summary.peak_memory_mb,
                threshold: self.config.alert_thresholds.max_memory_usage_mb,
            });
        }
        
        // Check error rates
        if query_summary.avg_error_rate > self.config.alert_thresholds.max_error_rate {
            alerts.push(Alert {
                severity: AlertSeverity::Critical,
                message: format!("High error rate: {:.2}%", query_summary.avg_error_rate * 100.0),
                metric_type: "error_rate".to_string(),
                current_value: query_summary.avg_error_rate,
                threshold: self.config.alert_thresholds.max_error_rate,
            });
        }
        
        alerts
    }
    
    /// Periodic export of metrics
    pub fn maybe_export_metrics(&self) {
        let mut last_export = self.last_export.lock().unwrap();
        let now = Instant::now();
        
        if now.duration_since(*last_export).as_secs() >= self.config.export_interval_seconds {
            let collector = get_metrics_collector();
            let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
            let filename = format!("metrics_export_{}.json", timestamp);
            
            if let Err(e) = collector.export_to_file(&filename) {
                tracing::error!("Failed to export metrics: {}", e);
            }
            
            *last_export = now;
        }
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let collector = get_metrics_collector();
        
        PerformanceReport {
            timestamp: Utc::now(),
            query_summary: collector.get_summary(OperationType::VectorQuery),
            indexing_summary: collector.get_summary(OperationType::IndexTraining),
            insertion_summary: collector.get_summary(OperationType::VectorInsertion),
            alerts: self.check_alerts(),
            recommendations: self.generate_recommendations(),
        }
    }
    
    fn generate_recommendations(&self) -> Vec<String> {
        let collector = get_metrics_collector();
        let mut recommendations = Vec::new();
        
        let query_summary = collector.get_summary(OperationType::VectorQuery);
        
        // Latency recommendations
        if query_summary.p95_duration_ms > 100.0 {
            recommendations.push("Consider reducing nprobe value to improve query latency".to_string());
        }
        
        if query_summary.p95_duration_ms > 500.0 {
            recommendations.push("Consider increasing shard size or reducing nlist for better performance".to_string());
        }
        
        // Memory recommendations
        if query_summary.peak_memory_mb > 4096.0 {
            recommendations.push("Consider using higher PQ compression (lower nbits) to reduce memory usage".to_string());
        }
        
        // Throughput recommendations
        if query_summary.operation_count > 0 {
            let avg_qps = query_summary.operation_count as f64 / (query_summary.avg_duration_ms / 1000.0);
            if avg_qps < 10.0 {
                recommendations.push("Consider optimizing index parameters or adding more shards for better throughput".to_string());
            }
        }
        
        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub message: String,
    pub metric_type: String,
    pub current_value: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: DateTime<Utc>,
    pub query_summary: MetricsSummary,
    pub indexing_summary: MetricsSummary,
    pub insertion_summary: MetricsSummary,
    pub alerts: Vec<Alert>,
    pub recommendations: Vec<String>,
}

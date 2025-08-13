# Scripts

Utility scripts for development, deployment, and maintenance.

## Development Scripts

### `setup.sh`
- Initial development environment setup
- Installs dependencies and configures local environment

### Performance Scripts

#### `analyze_performance.py`
- Analyzes vector database performance metrics
- Generates performance reports and recommendations

#### `monitor_performance.py` 
- Real-time performance monitoring
- Tracks resource usage during operations

#### `performance_tuning.py`
- Automated performance optimization
- Adjusts FAISS parameters based on workload

## Production Scripts

### `setup_production_faiss.sh`
- Production FAISS setup and optimization
- Configures optimal parameters for production workloads

## Usage

```bash
# Development setup
./scripts/setup.sh

# Performance monitoring
python scripts/monitor_performance.py --duration 300

# Performance analysis
python scripts/analyze_performance.py --log-file data/logs/performance.log

# Production setup
./scripts/setup_production_faiss.sh
```

# Archive Directory

This directory contains historical documentation and test files that were consolidated during the cleanup process.

## Archive Contents

### `docs/` - Historical Documentation
- `BACKEND_ARCHITECTURE.md` - Original backend architecture design
- `FAISS_IMPROVEMENTS_SUMMARY.md` - FAISS optimization notes
- `FAISS_INTEGRATION.md` - FAISS integration documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation progress notes
- `OPTIMIZATION_SUMMARY.md` - Performance optimization notes
- `PERFORMANCE_OPTIMIZATION_COMPLETE.md` - Detailed performance work
- `REAL_FAISS_IMPLEMENTATION.md` - FAISS implementation details
- `REAL_FAISS_STATUS.md` - FAISS status tracking
- `SCALABILITY_ANALYSIS.md` - Scalability analysis (empty)

### `tests/` - Legacy Test Files
- `test_all_s3_commands.py` - Original S3 commands test (replaced by `test_s3_compatibility.py`)
- `test_boto3.py` - Early boto3 integration tests
- `test_e2e_cached.py` - End-to-end cached tests
- `test_e2e.sh` - Shell-based end-to-end tests
- `test_vector_db.py` - Basic vector database tests

## Current Active Files

After cleanup, the project now uses:
- **`README.md`** - Concise project overview with S3 API compatibility status
- **`PRODUCTION_GUIDE.md`** - Comprehensive production deployment guide
- **`test_s3_compatibility.py`** - Essential S3 API compatibility test suite
- **`tests/`** - Core integration test suite

## Rationale for Archival

These files were moved to archive because:
1. **Documentation**: Consolidated into `PRODUCTION_GUIDE.md` and updated `README.md`
2. **Tests**: Consolidated into the essential `test_s3_compatibility.py` 
3. **Redundancy**: Multiple files covered similar topics
4. **Maintenance**: Reduced cognitive load for new contributors

## Recovery

If any archived content is needed:
```bash
# Copy back specific files
cp archive/docs/FAISS_INTEGRATION.md .
cp archive/tests/test_boto3.py .
```

The archive preserves the development history while keeping the main project clean and focused.

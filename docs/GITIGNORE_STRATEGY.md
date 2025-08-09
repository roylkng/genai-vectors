# Git Ignore Strategy for GenAI Vectors

## Data Directory Management

Our `.gitignore` is configured to balance development efficiency with repository size management.

### ✅ **Files INCLUDED in Git** (Tracked)
- `data/embedding_cache_test.pkl` (**69KB**) - Small test dataset for development
- `data/embeddings_cache_5k_s3vectors.pkl` (**33MB**) - Medium 5K dataset for testing

### 🚫 **Files EXCLUDED from Git** (Ignored)

#### Large Cache Files
```gitignore
data/embedding_cache_500k.pkl    # 3.3GB - Too large for repository
data/*_500k*.pkl                 # Any 500k+ vector files
data/*_1m*.pkl                   # Any 1M+ vector files  
data/*_large*.pkl                # Any files marked as "large"
```

#### Working Directories
```gitignore
data/temp/                       # Temporary processing files
data/logs/                       # Application logs
data/indexes/                    # Generated index files
```

#### Log Files (All Locations)
```gitignore
*.log                           # All log files
logs/                           # Root logs directory
data/logs/                      # Data logs directory
*.log.*                         # Rotated logs
log_*                           # Log files with prefix
```

## Strategy Rationale

### **Size Management**
- **Small files (< 100KB)**: Always track - essential for development
- **Medium files (1-50MB)**: Selective tracking - useful datasets
- **Large files (> 100MB)**: Always ignore - use external storage

### **Development Workflow**
1. **Quick Setup**: Developers get working test data immediately
2. **Performance Testing**: 5K dataset provides realistic performance testing
3. **Large Scale**: 500K dataset downloaded separately when needed

### **Repository Health**
- Keeps repository size manageable (< 50MB for data)
- Fast clone/pull operations
- Reduces Git LFS dependency

## Usage Examples

### Adding New Cache Files
```bash
# Small test datasets - these will be tracked
data/embedding_cache_100.pkl
data/test_vectors_small.pkl

# Large datasets - these will be ignored automatically  
data/embedding_cache_1m.pkl      # Ignored by *_1m*.pkl pattern
data/vectors_large_batch.pkl     # Ignored by *_large*.pkl pattern
```

### Working with Logs
```bash
# All of these are automatically ignored
data/logs/indexer.log
data/logs/api.2024-08-09.log
application.log
debug_output.log.1
```

### Temporary Files
```bash
# Automatically ignored
data/temp/processing_batch_1.json
data/temp/intermediate_vectors.pkl
data/indexes/hnsw_index_001/
```

## Maintenance Commands

### Check What Would Be Added
```bash
git add data/ --dry-run
```

### See Ignored Files
```bash
git status --ignored
```

### Force Add Large File (if needed)
```bash
git add -f data/special_large_file.pkl
```

## Best Practices

1. **Name Large Files Appropriately**: Use `_500k`, `_1m`, `_large` suffixes
2. **Use Temp Directory**: Put all processing files in `data/temp/`
3. **Regular Cleanup**: Remove old files from ignored directories
4. **Document External Storage**: Large datasets should be documented in README

This strategy ensures developers have working datasets while keeping the repository lean and fast.

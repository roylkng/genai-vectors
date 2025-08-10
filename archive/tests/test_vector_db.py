#!/usr/bin/env python3
"""
Comprehensive test of the GenAI Vector Database using cached vectors.
Tests the full pipeline: API, indexing, querying, and MinIO storage.
"""

import boto3
import json
import pickle
import random
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

# Configuration
MINIO_ENDPOINT = "http://localhost:9001"
API_ENDPOINT = "http://localhost:8080"
BUCKET_NAME = "vectors"
AWS_ACCESS_KEY = "minioadmin"
AWS_SECRET_KEY = "minioadmin"

def create_s3_client():
    """Create S3 client for MinIO."""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

def load_cached_vectors():
    """Load cached vectors from pickle files."""
    cache_dir = Path("data")
    cached_vectors = {}
    
    cache_files = {
        'test': cache_dir / 'embedding_cache_test.pkl',
        '5k': cache_dir / 'embeddings_cache_5k_s3vectors.pkl'
    }
    
    for name, file_path in cache_files.items():
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    cached_vectors[name] = data
                    print(f"✅ Loaded {name} cache: {len(data)} vectors")
            except Exception as e:
                print(f"❌ Failed to load {name} cache: {e}")
    
    return cached_vectors

def test_api_health():
    """Test API health endpoint."""
    try:
        response = requests.get(f"{API_ENDPOINT}/health")
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
            return True
        else:
            print(f"❌ API Health Check: FAILED (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Health Check: FAILED ({e})")
        return False

def test_minio_connection():
    """Test MinIO S3 connection."""
    try:
        s3 = create_s3_client()
        buckets = s3.list_buckets()
        bucket_names = [b['Name'] for b in buckets['Buckets']]
        
        if BUCKET_NAME in bucket_names:
            print(f"✅ MinIO Connection: PASSED (bucket '{BUCKET_NAME}' exists)")
            return True
        else:
            print(f"❌ MinIO Connection: FAILED (bucket '{BUCKET_NAME}' not found)")
            print(f"   Available buckets: {bucket_names}")
            return False
    except Exception as e:
        print(f"❌ MinIO Connection: FAILED ({e})")
        return False

def test_create_index():
    """Test creating a vector index."""
    index_config = {
        "name": "test-cached-vectors",
        "dim": 384,
        "metric": "cosine", 
        "nlist": 4,  # Small for testing
        "m": 4,
        "nbits": 8,
        "default_nprobe": 2
    }
    
    try:
        response = requests.post(
            f"{API_ENDPOINT}/indexes",
            headers={"Content-Type": "application/json"},
            json=index_config
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Index Creation: PASSED (index: {result.get('index', 'unknown')})")
            return True
        else:
            print(f"❌ Index Creation: FAILED (status: {response.status_code}, response: {response.text})")
            return False
    except Exception as e:
        print(f"❌ Index Creation: FAILED ({e})")
        return False

def convert_cached_vectors_to_api_format(cached_data, limit: int = 50):
    """Convert cached vectors to API format."""
    vectors = []
    
    if isinstance(cached_data, dict):
        # Format: {text: embedding_vector}
        for i, (text, embedding) in enumerate(cached_data.items()):
            if i >= limit:
                break
            vectors.append({
                'id': f'cached_vec_{i}',
                'embedding': embedding,
                'meta': {'text': text[:100], 'source': 'cached'}  # Truncate long text
            })
    elif isinstance(cached_data, list):
        # List format
        for i, item in enumerate(cached_data[:limit]):
            if isinstance(item, dict):
                # Already in good format
                if 'embedding' in item and 'id' in item:
                    vectors.append({
                        'id': item['id'],
                        'embedding': item['embedding'],
                        'meta': item.get('meta', item.get('metadata', {'source': 'cached'}))
                    })
                elif 'vector' in item:
                    vectors.append({
                        'id': item.get('id', f'cached_vec_{i}'),
                        'embedding': item['vector'],
                        'meta': item.get('metadata', {'source': 'cached'})
                    })
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # Assume [vector, metadata] or similar format
                vectors.append({
                    'id': f'cached_vec_{i}',
                    'embedding': item[0],
                    'meta': item[1] if len(item) > 1 else {'source': 'cached'}
                })
            elif isinstance(item, list):
                # Just a vector
                vectors.append({
                    'id': f'cached_vec_{i}',
                    'embedding': item,
                    'meta': {'source': 'cached'}
                })
    
    return vectors

def test_add_cached_vectors():
    """Test adding cached vectors to the index."""
    cached_vectors = load_cached_vectors()
    
    if not cached_vectors:
        print("❌ Vector Addition: FAILED (no cached vectors found)")
        return False
    
    # Use test cache first (smaller)
    cache_name = 'test' if 'test' in cached_vectors else list(cached_vectors.keys())[0]
    raw_vectors = cached_vectors[cache_name]
    
    print(f"Using {cache_name} cache with {len(raw_vectors)} vectors")
    
    # Convert to API format
    api_vectors = convert_cached_vectors_to_api_format(raw_vectors, limit=20)
    
    if not api_vectors:
        print("❌ Vector Addition: FAILED (could not convert cached vectors)")
        return False
    
    print(f"Converted {len(api_vectors)} vectors to API format")
    print(f"Sample vector: id={api_vectors[0]['id']}, embedding_dim={len(api_vectors[0]['embedding'])}")
    
    payload = {
        "index": "test-cached-vectors",
        "vectors": api_vectors
    }
    
    try:
        response = requests.post(
            f"{API_ENDPOINT}/vectors",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Vector Addition: PASSED (status: {result.get('status', 'unknown')})")
            return True
        else:
            print(f"❌ Vector Addition: FAILED (status: {response.status_code}, response: {response.text})")
            return False
    except Exception as e:
        print(f"❌ Vector Addition: FAILED ({e})")
        return False

def test_minio_storage():
    """Test that vectors are stored in MinIO."""
    try:
        s3 = create_s3_client()
        
        # Check for staged vectors
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="staged/")
        staged_objects = response.get('Contents', [])
        
        # Check for WAL
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="wal/")
        wal_objects = response.get('Contents', [])
        
        if staged_objects or wal_objects:
            print(f"✅ MinIO Storage: PASSED ({len(staged_objects)} staged objects, {len(wal_objects)} WAL objects)")
            return True
        else:
            print("❌ MinIO Storage: FAILED (no objects found in staged/ or wal/)")
            return False
    except Exception as e:
        print(f"❌ MinIO Storage: FAILED ({e})")
        return False

def test_query_vectors():
    """Test querying vectors."""
    cached_vectors = load_cached_vectors()
    
    if not cached_vectors:
        print("❌ Vector Query: FAILED (no cached vectors for query)")
        return False
    
    # Use test cache to get a real vector for querying
    cache_name = 'test' if 'test' in cached_vectors else list(cached_vectors.keys())[0]
    raw_vectors = cached_vectors[cache_name]
    api_vectors = convert_cached_vectors_to_api_format(raw_vectors, limit=1)
    
    if not api_vectors:
        print("❌ Vector Query: FAILED (could not get query vector)")
        return False
    
    query_vector = api_vectors[0]['embedding']
    
    query_payload = {
        "index": "test-cached-vectors",
        "embedding": query_vector,
        "topk": 5,
        "nprobe": 2
    }
    
    try:
        response = requests.post(
            f"{API_ENDPOINT}/query",
            headers={"Content-Type": "application/json"},
            json=query_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            results = result.get('results', [])
            took_ms = result.get('took_ms', 0)
            
            print(f"✅ Vector Query: PASSED ({len(results)} results in {took_ms}ms)")
            
            if results:
                print("   Top results:")
                for i, res in enumerate(results[:3]):
                    print(f"     {i+1}. Score: {res.get('score', 'N/A')}, ID: {res.get('id', 'N/A')}")
            
            return True
        else:
            print(f"❌ Vector Query: FAILED (status: {response.status_code}, response: {response.text})")
            return False
    except Exception as e:
        print(f"❌ Vector Query: FAILED ({e})")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🚀 Starting Comprehensive Vector Database Test")
    print("=" * 60)
    
    tests = [
        ("API Health", test_api_health),
        ("MinIO Connection", test_minio_connection),
        ("Index Creation", test_create_index),
        ("Vector Addition", test_add_cached_vectors),
        ("MinIO Storage", test_minio_storage),
        ("Vector Query", test_query_vectors)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}: FAILED (exception: {e})")
            results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Vector database is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)

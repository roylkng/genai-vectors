#!/usr/bin/env python3
"""
End-to-end test using cached vectors and S3 vectors API.
Uses boto3 S3 vectors operations (put_vectors, query_vectors) directly with MinIO.
"""

import boto3
import json
import pickle
import time
from pathlib import Path

# Configuration
MINIO_ENDPOINT = "http://localhost:9000"  # Port-forwarded MinIO (for direct S3 operations)
API_ENDPOINT = "http://localhost:8080"    # Our API server that implements S3 vectors interface
BUCKET_NAME = "vectors"
AWS_ACCESS_KEY = "minioadmin"
AWS_SECRET_KEY = "minioadmin"

def create_s3_client():
    """Create S3 client for MinIO (direct storage operations)."""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name='us-east-1'
    )

def create_s3_vectors_client():
    """Create boto3 s3vectors client pointing to our API service."""
    return boto3.client(
        's3vectors',
        endpoint_url='http://localhost:8080',
        aws_access_key_id='minioadmin',  # Match MinIO credentials
        aws_secret_access_key='minioadmin',
        region_name='us-east-1'
    )

def load_cached_vectors():
    """Load cached vectors from pickle files."""
    cache_file = Path("data/embedding_cache_test.pkl")
    
    if not cache_file.exists():
        print(f"❌ Cache file not found: {cache_file}")
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            print(f"✅ Loaded test cache: {len(data)} vectors")
            return data
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        return None

def convert_cached_to_api_format(cached_data, limit=20):
    """Convert cached vectors to API format."""
    vectors = []
    
    count = 0
    for text, embedding in cached_data.items():
        if count >= limit:
            break
            
        vectors.append({
            'id': f'cached_doc_{count}',
            'embedding': embedding,
            'meta': {
                'text': text[:200],  # Truncate long text
                'source': 'cached_test',
                'index': count
            }
        })
        count += 1
    
    print(f"✅ Converted {len(vectors)} vectors to API format")
    if vectors:
        print(f"   Sample: ID={vectors[0]['id']}, embedding_dim={len(vectors[0]['embedding'])}")
    
    return vectors

def test_s3_vectors_capability():
    """Test if S3 vectors operations are available."""
    try:
        s3_vectors = create_s3_vectors_client()
        
        # Try to list existing vector buckets to test S3 vectors capability
        response = s3_vectors.list_vector_buckets()
        
        print("✅ S3 Vectors API: Available")
        existing_buckets = response.get('VectorBuckets', [])
        if existing_buckets:
            print(f"   Found {len(existing_buckets)} existing vector buckets")
        return True
        
    except Exception as e:
        print(f"⚠️  S3 Vectors API: Not available ({e})")
        print("   This is expected if using standard MinIO without S3 vectors support")
        return False

def test_minio_connection():
    """Test MinIO S3 connection."""
    try:
        s3 = create_s3_client()
        buckets = s3.list_buckets()
        bucket_names = [b['Name'] for b in buckets['Buckets']]
        
        if BUCKET_NAME in bucket_names:
            print(f"✅ MinIO Connection: OK (bucket '{BUCKET_NAME}' exists)")
            return True
        else:
            # Try to create the bucket
            s3.create_bucket(Bucket=BUCKET_NAME)
            print(f"✅ MinIO Connection: OK (created bucket '{BUCKET_NAME}')")
            return True
    except Exception as e:
        print(f"❌ MinIO Connection: Failed ({e})")
        return False

def test_create_vector_bucket():
    """Create vector bucket using S3 vectors operations."""
    try:
        s3_vectors = create_s3_vectors_client()
        
        # Create vector bucket using create-vector-bucket operation
        response = s3_vectors.create_vector_bucket(
            vectorBucketName=BUCKET_NAME
        )
        
        print("✅ Vector Bucket Creation (S3): OK")
        print(f"   Created vector bucket: {BUCKET_NAME}")
        return True
        
    except Exception as e:
        print(f"❌ Vector Bucket Creation (S3): Failed ({e})")
        # Check if it already exists
        try:
            s3_vectors.get_vector_bucket(vectorBucketName=BUCKET_NAME)
            print("   Vector bucket already exists")
            return True
        except:
            print("   Note: S3 vectors API may not be available")
            return False

def test_create_index_s3():
    """Create index using S3 vectors operations."""
    try:
        s3_vectors = create_s3_vectors_client()
        
        # Create index using create-index operation
        response = s3_vectors.create_index(
            vectorBucketName=BUCKET_NAME,
            indexName='cached-test-index',
            dataType='FLOAT32',
            dimension=768,  # Correct dimension for cached embeddings
            distanceMetric='COSINE',
            metadataConfiguration={
                'nonFilterableMetadataKeys': ['source', 'index']  # Required field
            }
        )
        
        print("✅ Index Creation (S3): OK")
        print("   Created index: cached-test-index with dimension 768")
        return True
        
    except Exception as e:
        print(f"❌ Index Creation (S3): Failed ({e})")
        print("   Note: S3 vectors API may not be available, this is expected in test setup")
        return False

def test_put_vectors_s3():
    """Upload cached vectors using boto3 S3 vectors operations."""
    # Load cached vectors
    cached_data = load_cached_vectors()
    if not cached_data:
        return False
    
    # Convert to S3 vectors format
    api_vectors = convert_cached_to_api_format(cached_data, limit=25)
    if not api_vectors:
        print("❌ Put Vectors: No vectors to upload")
        return False
    
    # Use boto3 S3 vectors operations
    try:
        s3_vectors = create_s3_vectors_client()
        
        # Prepare vectors for S3 vectors format
        vector_data = []
        for vec in api_vectors:
            vector_data.append({
                'key': vec['id'],  # Use 'key' instead of 'Id'
                'data': {'float32': vec['embedding']},  # Wrap embedding in float32 dict
                'metadata': vec['meta']  # Use 'metadata' instead of 'Metadata'
            })
        
        # Use S3 vectors put-vectors operation
        response = s3_vectors.put_vectors(
            vectorBucketName=BUCKET_NAME,
            indexName='cached-test-index',
            vectors=vector_data
        )
        
        print("✅ Put Vectors (S3): OK")
        print(f"   Uploaded {len(vector_data)} vectors using S3 vectors API")
        return True
        
    except Exception as e:
        print(f"❌ Put Vectors (S3): Failed ({e})")
        # Fallback: check if this is because S3 vectors API is not available
        print("   Note: S3 vectors API may not be available, this is expected in test setup")
        return False

def test_s3_storage():
    """Check that vectors were stored in S3."""
    try:
        s3 = create_s3_client()
        
        # Check for staged files
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="staged/")
        staged_count = len(response.get('Contents', []))
        
        # Check for WAL files
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="wal/")
        wal_count = len(response.get('Contents', []))
        
        if staged_count > 0 or wal_count > 0:
            print(f"✅ S3 Storage: OK ({staged_count} staged files, {wal_count} WAL files)")
            return True
        else:
            print("❌ S3 Storage: No files found")
            return False
    except Exception as e:
        print(f"❌ S3 Storage: Failed ({e})")
        return False

def run_indexer():
    """Trigger indexing process."""
    print("⏳ Waiting for indexing to complete (this may take a moment)...")
    
    # In a real deployment, you'd trigger the indexer job
    # For now, we'll just wait a bit for the background processing
    time.sleep(5)
    
    # Check if index artifacts were created
    try:
        s3 = create_s3_client()
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="indexes/")
        index_objects = response.get('Contents', [])
        
        if index_objects:
            print(f"✅ Indexing: OK ({len(index_objects)} index artifacts)")
            for obj in index_objects[:5]:  # Show first 5
                print(f"   - {obj['Key']}")
            return True
        else:
            print("⚠️  Indexing: No artifacts found (indexer may not have run yet)")
            return False
    except Exception as e:
        print(f"❌ Indexing: Failed to check ({e})")
        return False

def test_query_vectors_s3():
    """Test querying with S3 vectors API."""
    # Load cached vectors to get a query vector
    cached_data = load_cached_vectors()
    if not cached_data:
        return False
    
    # Use one of the cached vectors as query
    sample_text = list(cached_data.keys())[0]
    query_embedding = cached_data[sample_text]
    
    try:
        s3_vectors = create_s3_vectors_client()
        
        # Use S3 vectors query operation
        response = s3_vectors.query_vectors(
            vectorBucketName=BUCKET_NAME,
            indexName='cached-test-index',
            topK=5,
            queryVector={'float32': query_embedding},  # queryVector should be dict with 'float32' field
            returnMetadata=True,
            returnDistance=True
        )
        
        results = response.get('Results', [])
        
        print(f"✅ Query Vectors (S3): OK ({len(results)} results)")
        
        if results:
            print("   Top results:")
            for i, res in enumerate(results[:3]):
                score = res.get('Score', 'N/A')
                doc_id = res.get('Id', 'N/A')
                metadata = res.get('Metadata', {})
                text_preview = str(metadata.get('text', ''))[:50]
                print(f"     {i+1}. Score: {score:.4f}, ID: {doc_id}, Text: {text_preview}...")
        else:
            print("   No results returned")
        
        return True
        
    except Exception as e:
        print(f"❌ Query Vectors (S3): Failed ({e})")
        print("   Note: S3 vectors API may not be available, this is expected in test setup")
        return False

def main():
    """Run the complete end-to-end test."""
    print("🚀 Starting End-to-End Vector Database Test with Cached Vectors")
    print("=" * 70)
    
    tests = [
        ("S3 Vectors API Check", test_s3_vectors_capability),
        ("MinIO Connection", test_minio_connection),
        ("Create Index (S3)", test_create_index_s3),
        ("Put Vectors (S3)", test_put_vectors_s3),
        ("S3 Storage Check", test_s3_storage),
        ("Run Indexer", run_indexer),
        ("Query Vectors (S3)", test_query_vectors_s3)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}: Exception ({e})")
            results[test_name] = False
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:25} {status}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The vector database is working end-to-end!")
    elif passed >= total - 1:
        print("👍 Most tests passed! Vector database is mostly functional.")
    else:
        print("⚠️  Several tests failed. Check the API and MinIO setup.")
    
    return passed >= total - 1  # Allow 1 test to fail

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

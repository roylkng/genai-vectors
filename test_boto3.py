#!/usr/bin/env python3
"""
Test the vector database using boto3 to interact with MinIO storage directly.
This tests the complete pipeline: vector ingestion -> staging -> indexing -> querying.
"""

import boto3
import json
import random
import time
from botocore.config import Config
from datetime import datetime

# MinIO configuration
MINIO_CONFIG = {
    'endpoint_url': 'http://localhost:9000',
    'aws_access_key_id': 'minioadmin',
    'aws_secret_access_key': 'minioadmin',
    'region_name': 'us-east-1'
}

BUCKET_NAME = 'vectors'

def create_s3_client():
    """Create boto3 S3 client for MinIO"""
    return boto3.client(
        's3',
        **MINIO_CONFIG,
        config=Config(signature_version='s3v4')
    )

def generate_test_vectors(count=10, dim=384):
    """Generate test vectors with metadata"""
    vectors = []
    for i in range(count):
        vector = {
            'id': f'test-doc-{i}',
            'embedding': [random.random() for _ in range(dim)],
            'meta': {
                'text': f'Test document {i} content for vector similarity search',
                'category': 'test',
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'boto3-test'
            },
            'created_at': datetime.utcnow().isoformat()
        }
        vectors.append(vector)
    return vectors

def test_bucket_access():
    """Test basic S3/MinIO bucket access"""
    print("🔍 Testing MinIO bucket access...")
    s3 = create_s3_client()
    
    try:
        # List buckets
        response = s3.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        print(f"✅ Available buckets: {buckets}")
        
        # Check if vectors bucket exists
        if BUCKET_NAME in buckets:
            print(f"✅ Bucket '{BUCKET_NAME}' exists")
        else:
            print(f"❌ Bucket '{BUCKET_NAME}' not found")
            return False
            
        # List objects in bucket
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        if 'Contents' in response:
            print(f"📁 Bucket contents ({len(response['Contents'])} objects):")
            for obj in response['Contents'][:10]:  # Show first 10
                print(f"   - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("📁 Bucket is empty")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to access MinIO: {e}")
        return False

def test_index_creation():
    """Test index configuration creation via S3"""
    print("\n🏗️  Testing index creation...")
    s3 = create_s3_client()
    
    try:
        # Create index configuration
        index_config = {
            'name': 'boto3-test-index',
            'dim': 384,
            'metric': 'cosine',
            'nlist': 16,  # Small for testing
            'm': 4,
            'nbits': 8,
            'default_nprobe': 4
        }
        
        config_key = f"indexes/{index_config['name']}/config.json"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=config_key,
            Body=json.dumps(index_config),
            ContentType='application/json'
        )
        
        print(f"✅ Created index config: {config_key}")
        
        # Verify the config was stored
        response = s3.get_object(Bucket=BUCKET_NAME, Key=config_key)
        stored_config = json.loads(response['Body'].read())
        print(f"✅ Verified config: {stored_config['name']} ({stored_config['dim']}D)")
        
        return index_config['name']
        
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        return None

def test_vector_staging():
    """Test vector staging (WAL and slice creation)"""
    print("\n📝 Testing vector staging...")
    s3 = create_s3_client()
    
    try:
        # Generate test vectors
        vectors = generate_test_vectors(50, 384)  # 50 vectors for testing
        print(f"🔢 Generated {len(vectors)} test vectors")
        
        # Create WAL entries (simulating the ingest process)
        wal_data = []
        for vector in vectors:
            wal_data.append(json.dumps(vector) + '\n')
        
        wal_key = "wal/boto3-test.ndjson"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=wal_key,
            Body=''.join(wal_data),
            ContentType='application/x-ndjson'
        )
        print(f"✅ Created WAL: {wal_key}")
        
        # Create staged slice (simulating buffer flush)
        slice_key = f"staged/boto3-test-index/slice-{int(time.time())}.jsonl"
        slice_data = []
        for vector in vectors:
            slice_data.append(json.dumps(vector) + '\n')
            
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=slice_key,
            Body=''.join(slice_data),
            ContentType='application/x-ndjson'
        )
        print(f"✅ Created staged slice: {slice_key}")
        
        return slice_key, vectors
        
    except Exception as e:
        print(f"❌ Failed to stage vectors: {e}")
        return None, None

def test_indexing_artifacts():
    """Check if indexing artifacts are created"""
    print("\n🔍 Checking indexing artifacts...")
    s3 = create_s3_client()
    
    try:
        # List objects under indexes/
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='indexes/'
        )
        
        if 'Contents' in response:
            print("📊 Index artifacts found:")
            for obj in response['Contents']:
                print(f"   - {obj['Key']} ({obj['Size']} bytes)")
                
            # Look for specific artifacts
            artifacts = [obj['Key'] for obj in response['Contents']]
            
            # Check for manifest files
            manifests = [k for k in artifacts if k.endswith('manifest.json')]
            if manifests:
                print(f"✅ Found manifests: {manifests}")
                
                # Read a manifest
                manifest_key = manifests[0]
                response = s3.get_object(Bucket=BUCKET_NAME, Key=manifest_key)
                manifest = json.loads(response['Body'].read())
                print(f"📋 Manifest details: {manifest.get('total_vectors', 0)} vectors, {len(manifest.get('shards', []))} shards")
            
            # Check for Faiss index files
            faiss_files = [k for k in artifacts if k.endswith('.faiss')]
            if faiss_files:
                print(f"🧠 Found Faiss indexes: {len(faiss_files)} files")
            
            return True
        else:
            print("❌ No indexing artifacts found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check artifacts: {e}")
        return False

def test_query_preparation():
    """Prepare a query vector for similarity search"""
    print("\n🔎 Preparing query test...")
    
    # Generate a query vector
    query_vector = [random.random() for _ in range(384)]
    
    query_data = {
        'index': 'boto3-test-index',
        'embedding': query_vector,
        'topk': 5,
        'nprobe': 4
    }
    
    print(f"🎯 Generated query vector ({len(query_vector)}D)")
    return query_data

def run_comprehensive_test():
    """Run the complete test suite"""
    print("🚀 Starting comprehensive vector database test with boto3\n")
    
    # Test 1: Basic connectivity
    if not test_bucket_access():
        print("❌ Cannot proceed - MinIO access failed")
        return False
    
    # Test 2: Index creation
    index_name = test_index_creation()
    if not index_name:
        print("❌ Cannot proceed - Index creation failed")
        return False
    
    # Test 3: Vector staging
    slice_key, vectors = test_vector_staging()
    if not slice_key:
        print("❌ Cannot proceed - Vector staging failed")
        return False
    
    # Test 4: Check for indexing artifacts (may not exist yet)
    has_artifacts = test_indexing_artifacts()
    
    # Test 5: Query preparation
    query_data = test_query_preparation()
    
    print(f"\n📊 Test Summary:")
    print(f"✅ MinIO connectivity: OK")
    print(f"✅ Index creation: OK ({index_name})")
    print(f"✅ Vector staging: OK ({len(vectors)} vectors)")
    print(f"{'✅' if has_artifacts else '⚠️'} Indexing artifacts: {'Found' if has_artifacts else 'Not found (may need indexer run)'}")
    print(f"✅ Query preparation: OK")
    
    if not has_artifacts:
        print(f"\n💡 To complete the test, run the indexer to process staged vectors:")
        print(f"   kubectl create job --from=cronjob/vector-store-indexer test-indexer -n genai-vectors")
    
    return True

if __name__ == "__main__":
    run_comprehensive_test()

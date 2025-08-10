#!/usr/bin/env python3
"""
Comprehensive test of all S3 vectors API commands.
Tests every operation to ensure production readiness.
"""

import boto3
import json
import time
from datetime import datetime

def create_s3vectors_client():
    """Create boto3 s3vectors client."""
    return boto3.client(
        's3vectors',
        endpoint_url='http://localhost:8080',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        region_name='us-east-1'
    )

def test_command(command_name, test_func):
    """Run a test command and report results."""
    print(f"\n🧪 Testing {command_name}...")
    try:
        result = test_func()
        print(f"✅ {command_name}: SUCCESS")
        if result:
            print(f"   Result: {result}")
        return True
    except Exception as e:
        print(f"❌ {command_name}: FAILED")
        print(f"   Error: {str(e)}")
        return False

def main():
    print("🚀 Comprehensive S3 Vectors API Command Test")
    print("=" * 70)
    
    client = create_s3vectors_client()
    test_results = {}
    
    # Test data
    test_bucket = "test-command-bucket"
    test_index = "test-command-index"
    test_vectors = [
        {
            'key': 'test-vec-1',
            'data': {'float32': [0.1, 0.2, 0.3, 0.4]},
            'metadata': {'category': 'test', 'id': 1}
        },
        {
            'key': 'test-vec-2', 
            'data': {'float32': [0.5, 0.6, 0.7, 0.8]},
            'metadata': {'category': 'test', 'id': 2}
        }
    ]
    
    # 1. CREATE-VECTOR-BUCKET
    def test_create_vector_bucket():
        response = client.create_vector_bucket(vectorBucketName=test_bucket)
        return f"Bucket {test_bucket} created"
    
    test_results['create-vector-bucket'] = test_command('create-vector-bucket', test_create_vector_bucket)
    
    # 2. LIST-VECTOR-BUCKETS
    def test_list_vector_buckets():
        response = client.list_vector_buckets()
        buckets = response.get('VectorBuckets', [])
        return f"Found {len(buckets)} buckets"
    
    test_results['list-vector-buckets'] = test_command('list-vector-buckets', test_list_vector_buckets)
    
    # 3. GET-VECTOR-BUCKET
    def test_get_vector_bucket():
        response = client.get_vector_bucket(vectorBucketName=test_bucket)
        return f"Bucket info retrieved"
    
    test_results['get-vector-bucket'] = test_command('get-vector-bucket', test_get_vector_bucket)
    
    # 4. CREATE-INDEX
    def test_create_index():
        response = client.create_index(
            indexName=test_index,
            vectorBucketName=test_bucket,
            dataType='float32',
            dimension=4,
            distanceMetric='cosine'
        )
        return f"Index {test_index} created"
    
    test_results['create-index'] = test_command('create-index', test_create_index)
    
    # 5. LIST-INDEXES
    def test_list_indexes():
        response = client.list_indexes(vectorBucketName=test_bucket)
        indexes = response.get('Indexes', [])
        return f"Found {len(indexes)} indexes"
    
    test_results['list-indexes'] = test_command('list-indexes', test_list_indexes)
    
    # 6. GET-INDEX
    def test_get_index():
        response = client.get_index(
            indexName=test_index,
            vectorBucketName=test_bucket
        )
        return f"Index info retrieved"
    
    test_results['get-index'] = test_command('get-index', test_get_index)
    
    # 7. PUT-VECTORS
    def test_put_vectors():
        response = client.put_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket,
            vectors=test_vectors
        )
        return f"Put {len(test_vectors)} vectors"
    
    test_results['put-vectors'] = test_command('put-vectors', test_put_vectors)
    
    # Wait a moment for indexing
    print("\n⏳ Waiting for indexing to complete...")
    time.sleep(3)
    
    # 8. LIST-VECTORS
    def test_list_vectors():
        response = client.list_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket
        )
        vectors = response.get('Vectors', [])
        return f"Found {len(vectors)} vectors"
    
    test_results['list-vectors'] = test_command('list-vectors', test_list_vectors)
    
    # 9. GET-VECTORS
    def test_get_vectors():
        response = client.get_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket,
            keys=['test-vec-1']
        )
        vectors = response.get('Vectors', [])
        return f"Retrieved {len(vectors)} vectors"
    
    test_results['get-vectors'] = test_command('get-vectors', test_get_vectors)
    
    # 10. QUERY-VECTORS
    def test_query_vectors():
        response = client.query_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket,
            queryVector={'float32': [0.2, 0.3, 0.4, 0.5]},
            topK=2
        )
        results = response.get('Results', [])
        return f"Query returned {len(results)} results"
    
    test_results['query-vectors'] = test_command('query-vectors', test_query_vectors)
    
    # 11. DELETE-VECTORS
    def test_delete_vectors():
        response = client.delete_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket,
            keys=['test-vec-1']
        )
        return f"Deleted vectors"
    
    test_results['delete-vectors'] = test_command('delete-vectors', test_delete_vectors)
    
    # 12. DELETE-INDEX
    def test_delete_index():
        response = client.delete_index(
            indexName=test_index,
            vectorBucketName=test_bucket
        )
        return f"Index {test_index} deleted"
    
    test_results['delete-index'] = test_command('delete-index', test_delete_index)
    
    # 13. DELETE-VECTOR-BUCKET
    def test_delete_vector_bucket():
        response = client.delete_vector_bucket(vectorBucketName=test_bucket)
        return f"Bucket {test_bucket} deleted"
    
    test_results['delete-vector-bucket'] = test_command('delete-vector-bucket', test_delete_vector_bucket)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for command, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {command:<25} {status}")
    
    print(f"\n🏆 Overall: {passed}/{total} commands passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All S3 vectors API commands working perfectly!")
    else:
        print("⚠️ Some commands need attention for production readiness.")

if __name__ == "__main__":
    main()

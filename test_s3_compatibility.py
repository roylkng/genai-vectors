#!/usr/bin/env python3
"""
S3 Vectors API Compatibility Test Suite

This is the essential test file for validating complete S3 vectors API compatibility.
Tests all 13 AWS S3 Vectors operations to ensure production readiness.

Run with: python test_s3_compatibility.py

Expected Result: 13/13 commands passed (100.0%)
"""

import boto3
import json
import time
from datetime import datetime

def create_s3vectors_client():
    """Create boto3 s3vectors client pointing to our service."""
    return boto3.client(
        's3vectors',
        endpoint_url='http://localhost:8080',
        aws_access_key_id='test',  # Service accepts any credentials
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

def test_command(command_name, test_func):
    """Execute a test command and report results."""
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
    print("🚀 S3 Vectors API Compatibility Test")
    print("Testing all 13 AWS S3 Vectors operations")
    print("=" * 70)
    
    client = create_s3vectors_client()
    test_results = {}
    
    # Test configuration
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

    # 1. Create Vector Bucket
    def test_create_vector_bucket():
        response = client.create_vector_bucket(vectorBucketName=test_bucket)
        return f"Bucket {test_bucket} created"
    
    test_results['create-vector-bucket'] = test_command('create-vector-bucket', test_create_vector_bucket)

    # 2. List Vector Buckets
    def test_list_vector_buckets():
        response = client.list_vector_buckets()
        bucket_count = len(response.get('vectorBuckets', []))
        return f"Found {bucket_count} buckets"
    
    test_results['list-vector-buckets'] = test_command('list-vector-buckets', test_list_vector_buckets)

    # 3. Get Vector Bucket
    def test_get_vector_bucket():
        response = client.get_vector_bucket(vectorBucketName=test_bucket)
        return "Bucket info retrieved"
    
    test_results['get-vector-bucket'] = test_command('get-vector-bucket', test_get_vector_bucket)

    # 4. Create Index
    def test_create_index():
        response = client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dimension=4,  # Match test vector dimension
            distanceMetric='COSINE',
            dataType='FLOAT32'  # Required parameter
        )
        return f"Index {test_index} created"
    
    test_results['create-index'] = test_command('create-index', test_create_index)

    # 5. List Indexes
    def test_list_indexes():
        response = client.list_indexes(vectorBucketName=test_bucket)
        index_count = len(response.get('indexes', []))
        return f"Found {index_count} indexes"
    
    test_results['list-indexes'] = test_command('list-indexes', test_list_indexes)

    # 6. Get Index
    def test_get_index():
        response = client.get_index(
            indexName=test_index,
            vectorBucketName=test_bucket
        )
        return "Index info retrieved"
    
    test_results['get-index'] = test_command('get-index', test_get_index)

    # 7. Put Vectors
    def test_put_vectors():
        response = client.put_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            vectors=test_vectors
        )
        return f"Put {len(test_vectors)} vectors"
    
    test_results['put-vectors'] = test_command('put-vectors', test_put_vectors)

    # Wait for indexing
    print("\n⏳ Waiting for indexing to complete...")
    time.sleep(3)

    # 8. List Vectors
    def test_list_vectors():
        response = client.list_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index
        )
        vector_count = len(response.get('vectors', []))
        return f"Found {vector_count} vectors"
    
    test_results['list-vectors'] = test_command('list-vectors', test_list_vectors)

    # 9. Get Vectors
    def test_get_vectors():
        response = client.get_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=['test-vec-1']
        )
        vector_count = len(response.get('vectors', []))
        return f"Retrieved {vector_count} vectors"
    
    test_results['get-vectors'] = test_command('get-vectors', test_get_vectors)

    # 10. Query Vectors
    def test_query_vectors():
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={'float32': [0.1, 0.2, 0.3, 0.4]},
            topK=5
        )
        result_count = len(response.get('matches', []))
        return f"Query returned {result_count} results"
    
    test_results['query-vectors'] = test_command('query-vectors', test_query_vectors)

    # 11. Delete Vectors
    def test_delete_vectors():
        response = client.delete_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=['test-vec-1']
        )
        return "Deleted vectors"
    
    test_results['delete-vectors'] = test_command('delete-vectors', test_delete_vectors)

    # 12. Delete Index
    def test_delete_index():
        response = client.delete_index(
            vectorBucketName=test_bucket,
            indexName=test_index
        )
        return f"Index {test_index} deleted"
    
    test_results['delete-index'] = test_command('delete-index', test_delete_index)

    # 13. Delete Vector Bucket
    def test_delete_vector_bucket():
        response = client.delete_vector_bucket(vectorBucketName=test_bucket)
        return f"Bucket {test_bucket} deleted"
    
    test_results['delete-vector-bucket'] = test_command('delete-vector-bucket', test_delete_vector_bucket)

    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25} {status}")
        if passed:
            passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🏆 Overall: {passed_tests}/{total_tests} commands passed ({success_rate:.1f}%)")
    
    if success_rate == 100.0:
        print("🎉 All S3 vectors API commands working perfectly!")
    elif success_rate >= 90.0:
        print("⚠️ Most commands working - minor issues need attention.")
    else:
        print("⚠️ Some commands need attention for production readiness.")
    
    return success_rate == 100.0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Enhanced comprehensive test of all S3 vectors API commands.
Tests that validate actual functionality, not just status codes.
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

def cleanup_resources(client, test_bucket, test_index):
    """Clean up any existing test resources."""
    try:
        # Try to delete vectors
        client.delete_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=['test-vec-1', 'test-vec-2', 'test-vec-3']
        )
    except:
        pass
    
    try:
        # Try to delete index
        client.delete_index(
            vectorBucketName=test_bucket,
            indexName=test_index
        )
    except:
        pass
    
    try:
        # Try to delete bucket
        client.delete_vector_bucket(vectorBucketName=test_bucket)
    except:
        pass

def main():
    print("🚀 Enhanced Comprehensive S3 Vectors API Test")
    print("=" * 70)
    
    client = create_s3vectors_client()
    test_results = {}
    
    # Test data
    test_bucket = "test-enhanced-bucket"
    test_index = "test-enhanced-index"
    
    # Known test vectors with predictable similarity
    test_vectors = [
        {
            'key': 'test-vec-1',
            'data': {'float32': [1.0, 0.0, 0.0, 0.0]},  # Orthogonal vector
            'metadata': {'category': 'test', 'id': 1, 'name': 'vector-one'}
        },
        {
            'key': 'test-vec-2',
            'data': {'float32': [0.0, 1.0, 0.0, 0.0]},  # Orthogonal vector
            'metadata': {'category': 'test', 'id': 2, 'name': 'vector-two'}
        },
        {
            'key': 'test-vec-3',
            'data': {'float32': [0.9, 0.1, 0.0, 0.0]},  # Similar to vec-1
            'metadata': {'category': 'similar', 'id': 3, 'name': 'vector-three'}
        }
    ]
    
    # Clean up any existing test resources
    cleanup_resources(client, test_bucket, test_index)
    
    # 1. CREATE-VECTOR-BUCKET
    def test_create_vector_bucket():
        response = client.create_vector_bucket(vectorBucketName=test_bucket)
        # Validate response structure
        if 'BucketName' not in response:
            raise Exception("Response missing BucketName field")
        if response['BucketName'] != test_bucket:
            raise Exception(f"Expected bucket name {test_bucket}, got {response['BucketName']}")
        return f"Created bucket: {response['BucketName']}"
    
    test_results['create-vector-bucket'] = test_command('create-vector-bucket', test_create_vector_bucket)
    
    # 2. LIST-VECTOR-BUCKETS
    def test_list_vector_buckets():
        response = client.list_vector_buckets()
        if 'VectorBuckets' not in response:
            raise Exception("Response missing VectorBuckets field")
        buckets = response['VectorBuckets']
        bucket_names = [b['Name'] for b in buckets]
        if test_bucket not in bucket_names:
            raise Exception(f"Test bucket {test_bucket} not found in list: {bucket_names}")
        return f"Found {len(buckets)} buckets, including our test bucket"
    
    test_results['list-vector-buckets'] = test_command('list-vector-buckets', test_list_vector_buckets)
    
    # 3. CREATE-INDEX
    def test_create_index():
        response = client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dataType='float32',
            dimension=4,
            distanceMetric='euclidean'
        )
        if 'IndexName' not in response:
            raise Exception("Response missing IndexName field")
        if response['IndexName'] != test_index:
            raise Exception(f"Expected index name {test_index}, got {response['IndexName']}")
        return f"Created index: {response['IndexName']}"
    
    test_results['create-index'] = test_command('create-index', test_create_index)
    
    # 4. PUT-VECTORS (add test data)
    def test_put_vectors():
        response = client.put_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            vectors=test_vectors
        )
        if 'VectorIds' not in response:
            raise Exception("Response missing VectorIds field")
        vector_ids = response['VectorIds']
        expected_keys = [v['key'] for v in test_vectors]
        for key in expected_keys:
            if key not in vector_ids:
                raise Exception(f"Expected vector key {key} not found in response")
        return f"Added {len(vector_ids)} vectors: {vector_ids}"
    
    test_results['put-vectors'] = test_command('put-vectors', test_put_vectors)
    
    # Wait for indexing
    print("\n⏳ Waiting for indexing to complete...")
    time.sleep(3)
    
    # 5. LIST-VECTORS
    def test_list_vectors():
        response = client.list_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index
        )
        if 'Vectors' not in response:
            raise Exception("Response missing Vectors field")
        vectors = response['Vectors']
        if len(vectors) != len(test_vectors):
            raise Exception(f"Expected {len(test_vectors)} vectors, got {len(vectors)}")
        return f"Listed {len(vectors)} vectors in index"
    
    test_results['list-vectors'] = test_command('list-vectors', test_list_vectors)
    
    # 6. GET-VECTORS
    def test_get_vectors():
        response = client.get_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=['test-vec-1', 'test-vec-2']
        )
        if 'Vectors' not in response:
            raise Exception("Response missing Vectors field")
        vectors = response['Vectors']
        if len(vectors) != 2:
            raise Exception(f"Expected 2 vectors, got {len(vectors)}")
        
        # Validate vector content
        vec1 = next((v for v in vectors if v['Key'] == 'test-vec-1'), None)
        if not vec1:
            raise Exception("test-vec-1 not found in response")
        if vec1['Data']['float32'] != [1.0, 0.0, 0.0, 0.0]:
            raise Exception(f"test-vec-1 has incorrect data: {vec1['Data']['float32']}")
        
        return f"Retrieved {len(vectors)} vectors with correct data"
    
    test_results['get-vectors'] = test_command('get-vectors', test_get_vectors)
    
    # 7. QUERY-VECTORS (Enhanced with multiple test cases)
    def test_query_vectors():
        # Test 1: Query with exact match to test-vec-1
        response1 = client.query_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket,
            queryVector={'float32': [1.0, 0.0, 0.0, 0.0]},  # Exact match to test-vec-1
            topK=3
        )
        results1 = response1.get('Results', [])
        
        if len(results1) == 0:
            raise Exception("Query returned 0 results - vectors not properly indexed")
        
        # The most similar should be test-vec-1 (exact match)
        top_result = results1[0]
        if top_result['Id'] != 'test-vec-1':
            raise Exception(f"Expected top result to be test-vec-1, got {top_result['Id']}")
        
        # Test 2: Query with vector similar to test-vec-3
        response2 = client.query_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket,
            queryVector={'float32': [0.8, 0.2, 0.0, 0.0]},  # Similar to test-vec-3
            topK=2
        )
        results2 = response2.get('Results', [])
        
        if len(results2) == 0:
            raise Exception("Second query returned 0 results")
        
        return f"Query 1: {len(results1)} results (top: {results1[0]['Id']}), Query 2: {len(results2)} results (top: {results2[0]['Id']})"
    
    test_results['query-vectors'] = test_command('query-vectors', test_query_vectors)
    
    # 8. DELETE-VECTORS
    def test_delete_vectors():
        response = client.delete_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=['test-vec-1', 'test-vec-2']
        )
        if 'DeletedIds' not in response:
            raise Exception("Response missing DeletedIds field")
        deleted_ids = response['DeletedIds']
        if len(deleted_ids) != 2:
            raise Exception(f"Expected 2 deleted IDs, got {len(deleted_ids)}")
        if 'test-vec-1' not in deleted_ids or 'test-vec-2' not in deleted_ids:
            raise Exception(f"Expected specific IDs to be deleted, got: {deleted_ids}")
        return f"Deleted {len(deleted_ids)} vectors: {deleted_ids}"
    
    test_results['delete-vectors'] = test_command('delete-vectors', test_delete_vectors)
    
    # 9. Verify deletion with another query
    def test_verify_deletion():
        response = client.query_vectors(
            indexName=test_index,
            vectorBucketName=test_bucket,
            queryVector={'float32': [1.0, 0.0, 0.0, 0.0]},
            topK=3
        )
        results = response.get('Results', [])
        # Should only find test-vec-3 now (test-vec-1 and test-vec-2 deleted)
        remaining_ids = [r['Id'] for r in results]
        if 'test-vec-1' in remaining_ids or 'test-vec-2' in remaining_ids:
            raise Exception(f"Deleted vectors still found in query: {remaining_ids}")
        if 'test-vec-3' not in remaining_ids:
            raise Exception(f"Expected test-vec-3 to remain, but got: {remaining_ids}")
        return f"Verified deletion: only {len(results)} vectors remain"
    
    test_results['verify-deletion'] = test_command('verify-deletion', test_verify_deletion)
    
    # 10. Clean up
    def test_cleanup():
        # Delete remaining vector
        client.delete_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=['test-vec-3']
        )
        
        # Delete index
        client.delete_index(
            vectorBucketName=test_bucket,
            indexName=test_index
        )
        
        # Delete bucket
        client.delete_vector_bucket(vectorBucketName=test_bucket)
        
        return "Cleaned up all test resources"
    
    test_results['cleanup'] = test_command('cleanup', test_cleanup)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Enhanced Test Results Summary:")
    print("=" * 70)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! S3 API is working correctly with real data validation.")
    else:
        print("⚠️ Some tests failed. Check the S3 API implementation.")

if __name__ == "__main__":
    main()

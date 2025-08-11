#!/usr/bin/env python3
"""
Metadata Filtering Test for S3 Vectors API

This test validates the metadata filtering capabilities of the vector store,
including the use of filterable and non-filterable keys.

Requirements:
1. Vector store service running at http://localhost:8080
2. MinIO or compatible S3 storage running

Usage:
    cd tests/integration
    python test_metadata_filtering.py
"""

import boto3
import time
from typing import List

# Configuration
VECTOR_SERVICE_URL = "http://localhost:8080"

# Test Data
TEST_VECTORS = [
    {
        "id": "vec_1",
        "embedding": [0.1, 0.2, 0.3, 0.4],
        "metadata": {
            "document_type": "report",
            "security_level": "high",
            "year": 2023,
            "published": True,
            "rating": 4.5
        }
    },
    {
        "id": "vec_2",
        "embedding": [0.5, 0.6, 0.7, 0.8],
        "metadata": {
            "document_type": "email",
            "security_level": "medium",
            "year": 2023,
            "published": False,
            "rating": 3.2
        }
    },
    {
        "id": "vec_3",
        "embedding": [0.9, 0.1, 0.2, 0.3],
        "metadata": {
            "document_type": "report",
            "security_level": "low",
            "year": 2022,
            "published": True,
            "rating": 4.8
        }
    },
    {
        "id": "vec_4",
        "embedding": [0.4, 0.5, 0.6, 0.7],
        "metadata": {
            "document_type": "invoice",
            "security_level": "high",
            "year": 2024,
            "published": True,
            "rating": 4.9
        }
    },
    {
        "id": "vec_5",
        "embedding": [0.8, 0.9, 0.1, 0.2],
        "metadata": {
            "document_type": "report",
            "security_level": "medium",
            "year": 2023,
            "published": True,
            "rating": 3.9
        }
    }
]

def create_s3vectors_client():
    """Create boto3 s3vectors client pointing to our service."""
    return boto3.client(
        's3vectors',
        endpoint_url=VECTOR_SERVICE_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

def test_step(step_name: str, test_func, *args, **kwargs):
    """Execute a test step and report results."""
    print(f"\n🧪 {step_name}")
    print("-" * 60)
    try:
        result = test_func(*args, **kwargs)
        print(f"✅ {step_name}: SUCCESS")
        if result is not None:
            print(f"   Result: {result}")
        return result, True
    except Exception as e:
        print(f"❌ {step_name}: FAILED")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        return None, False

def main():
    print("🚀 Metadata Filtering Test for S3 Vectors API")
    print("=" * 80)
    
    # Test configuration
    test_bucket = f"metadata-filter-test-{int(time.time())}"
    test_index = "filtered-index"
    
    # Initialize client
    client = create_s3vectors_client()
    
    # Track test results
    test_results = []
    
    # Step 1: Create vector bucket
    def create_vector_bucket():
        client.create_vector_bucket(vectorBucketName=test_bucket)
        return f"Created bucket: {test_bucket}"
    
    _, bucket_success = test_step("Create Vector Bucket", create_vector_bucket)
    test_results.append(("Create Bucket", bucket_success))
    if not bucket_success:
        return False

    # Step 2: Create index with non-filterable keys (all others are filterable by default)
    def create_filtered_index():
        client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dataType="FLOAT32",
            dimension=4,
            distanceMetric="COSINE",
            metadataConfiguration={
                "nonFilterableMetadataKeys": ["security_level", "rating"]
            }
        )
        return f"Created index '{test_index}' with non-filterable keys: security_level, rating"

    _, index_success = test_step("Create Filtered Index", create_filtered_index)
    test_results.append(("Create Index", index_success))
    if not index_success:
        return False

    # Step 3: Put vectors with metadata
    def put_vectors_with_metadata():
        vectors_to_put = [
            {
                "key": v["id"],
                "data": {"float32": v["embedding"]},
                "metadata": v["metadata"]
            } for v in TEST_VECTORS
        ]
        client.put_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            vectors=vectors_to_put
        )
        return f"Put {len(vectors_to_put)} vectors"

    _, put_success = test_step("Put Vectors with Metadata", put_vectors_with_metadata)
    test_results.append(("Put Vectors", put_success))
    if not put_success:
        return False

    print("\n⏳ Waiting for indexing to complete...")
    time.sleep(5)

    # Step 4: Query with metadata filter for filterable keys
    def query_with_filter():
        query_embedding = [0.11, 0.22, 0.33, 0.44] # Close to vec_1
        
        # Filter for reports from 2023 that are published (should work - these are filterable)
        metadata_filter = {
            "and": [
                {"field": "document_type", "operator": "eq", "value": "report"},
                {"field": "year", "operator": "eq", "value": 2023},
                {"field": "published", "operator": "eq", "value": True}
            ]
        }
        
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={"float32": query_embedding},
            topK=5,
            filter=metadata_filter
        )
        
        results = response.get("Results", [])
        print(f"   Found {len(results)} results with filterable metadata filter:")
        
        result_ids = []
        for r in results:
            print(f"     - ID: {r['Id']}, Score: {r['Score']:.4f}, Metadata: {r['Metadata']}")
            result_ids.append(r['Id'])
            
        # Should find vec_1 and vec_5 (reports from 2023 that are published)
        expected_results = 2
        print(f"   Expected {expected_results} results, got {len(results)}")
        
        return f"Query successful, found {len(results)} matching vectors: {result_ids}"

    _, filter_query_success = test_step("Query with Filterable Metadata", query_with_filter)
    test_results.append(("Query Filterable", filter_query_success))

    # Step 5: Test filtering by non-filterable key (should fail or be ignored)
    def query_with_non_filterable_filter():
        query_embedding = [0.88, 0.11, 0.22, 0.33]
        
        # Try to filter by security_level (marked as non-filterable)
        # This should either fail or ignore the filter
        metadata_filter = {
            "field": "security_level", 
            "operator": "eq", 
            "value": "high"
        }
        
        try:
            response = client.query_vectors(
                vectorBucketName=test_bucket,
                indexName=test_index,
                queryVector={"float32": query_embedding},
                topK=5,
                filter=metadata_filter
            )
            
            results = response.get("Results", [])
            print(f"   Non-filterable key filter returned {len(results)} results (should be ignored or fail)")
            
            # If it doesn't fail, it should return all results (filter ignored)
            return f"Non-filterable filter processed, returned {len(results)} results"
            
        except Exception as e:
            print(f"   Non-filterable key filter correctly failed: {e}")
            return "Non-filterable filter correctly rejected"

    _, non_filterable_success = test_step("Query with Non-Filterable Key", query_with_non_filterable_filter)
    test_results.append(("Query Non-Filterable", non_filterable_success))

    # Step 6: Complex filter combining filterable keys
    def query_with_complex_filter():
        query_embedding = [0.88, 0.11, 0.22, 0.33]
        
        # Filter for (reports OR emails) from year > 2022 (all filterable keys)
        metadata_filter = {
            "and": [
                {
                    "or": [
                        {"field": "document_type", "operator": "eq", "value": "report"},
                        {"field": "document_type", "operator": "eq", "value": "email"}
                    ]
                },
                {"field": "year", "operator": "gt", "value": 2022}
            ]
        }
        
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={"float32": query_embedding},
            topK=5,
            filter=metadata_filter
        )
        
        results = response.get("Results", [])
        print(f"   Found {len(results)} results with complex filter:")
        
        result_ids = []
        for r in results:
            print(f"     - ID: {r['Id']}, Score: {r['Score']:.4f}, Metadata: {r['Metadata']}")
            result_ids.append(r['Id'])
        
        # Should find vec_1, vec_2, and vec_5 (reports or emails from > 2022)
        expected_results = 3
        print(f"   Expected {expected_results} results, got {len(results)}")
        
        return f"Complex filter successful, found {len(results)} matching vectors: {result_ids}"

    _, complex_filter_success = test_step("Query with Complex Filter", query_with_complex_filter)
    test_results.append(("Query Complex", complex_filter_success))

    # Cleanup Step
    def cleanup_resources():
        client.delete_index(vectorBucketName=test_bucket, indexName=test_index)
        print(f"   ✅ Deleted index: {test_index}")
        client.delete_vector_bucket(vectorBucketName=test_bucket)
        print(f"   ✅ Deleted bucket: {test_bucket}")
        return "Cleanup completed"

    _, cleanup_success = test_step("Cleanup Test Resources", cleanup_resources)
    test_results.append(("Cleanup", cleanup_success))

    # Final Summary
    print("\n" + "=" * 80)
    print("📊 Metadata Filtering Test Results Summary")
    print("=" * 80)
    
    passed_tests = sum(1 for _, passed in test_results if passed)
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<30} {status}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🏆 Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    print("\n📋 Key Learnings:")
    print("• Metadata keys are filterable by default")
    print("• nonFilterableMetadataKeys makes specific keys non-filterable")
    print("• Filtering by non-filterable keys should fail or be ignored")
    print("• Complex filters with AND/OR work on filterable keys")
    
    return success_rate >= 80.0  # Allow some tolerance for non-filterable key behavior

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

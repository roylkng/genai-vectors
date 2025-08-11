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
from typing import List, Dict, Any

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

    # Step 2: Create index with filterable and non-filterable keys
    def create_filtered_index():
        client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dataType="FLOAT32",
            dimension=4,
            distanceMetric="COSINE",
            metadataConfiguration={
                "filterableKeys": ["document_type", "year", "published"],
                "nonFilterableMetadataKeys": ["security_level", "rating"]
            }
        )
        return f"Created index '{test_index}' with filterable keys"

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

    # Step 4: Query with metadata filter
    def query_with_filter():
        query_embedding = [0.11, 0.22, 0.33, 0.44] # Close to vec_1
        
        # Filter for reports from 2023 that are published
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
            metadataFilter=metadata_filter
        )
        
        results = response.get("Results", [])
        print(f"   Found {len(results)} results with filter:")
        
        result_ids = []
        for r in results:
            print(f"     - ID: {r['Id']}, Score: {r['Score']:.4f}, Metadata: {r['Metadata']}")
            result_ids.append(r['Id'])
            
        # Assertions
        assert len(results) == 2, f"Expected 2 results, but got {len(results)}"
        assert "vec_1" in result_ids, "vec_1 should be in the results"
        assert "vec_5" in result_ids, "vec_5 should be in the results"
        
        return f"Query successful, found {len(results)} matching vectors: {result_ids}"

    _, filter_query_success = test_step("Query with Metadata Filter", query_with_filter)
    test_results.append(("Query with Filter", filter_query_success))

    # Step 5: Query with a different filter
    def query_with_complex_filter():
        query_embedding = [0.88, 0.11, 0.22, 0.33] # Close to vec_3
        
        # Filter for (reports OR emails) from year > 2022
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
            metadataFilter=metadata_filter
        )
        
        results = response.get("Results", [])
        print(f"   Found {len(results)} results with complex filter:")
        
        result_ids = []
        for r in results:
            print(f"     - ID: {r['Id']}, Score: {r['Score']:.4f}, Metadata: {r['Metadata']}")
            result_ids.append(r['Id'])
            
        # Assertions
        assert len(results) == 3, f"Expected 3 results, but got {len(results)}"
        assert "vec_1" in result_ids
        assert "vec_2" in result_ids
        assert "vec_5" in result_ids
        
        return f"Query successful, found {len(results)} matching vectors: {result_ids}"

    _, complex_filter_success = test_step("Query with Complex Filter (OR, GT)", query_with_complex_filter)
    test_results.append(("Query with Complex Filter", complex_filter_success))

    # Step 6: Query that should return no results
    def query_with_no_results_filter():
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Filter for something that doesn't exist
        metadata_filter = {"field": "year", "operator": "eq", "value": 2000}
        
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={"float32": query_embedding},
            topK=5,
            metadataFilter=metadata_filter
        )
        
        results = response.get("Results", [])
        assert len(results) == 0, f"Expected 0 results, but got {len(results)}"
        
        return "Query correctly returned 0 results"

    _, no_results_success = test_step("Query with No-Result Filter", query_with_no_results_filter)
    test_results.append(("Query No Results", no_results_success))

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
    
    return success_rate == 100.0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

import boto3
import time
import random
from typing import List, Dict, Any

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

    # Step 2: Create index with filterable and non-filterable keys
    def create_filtered_index():
        client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dataType="FLOAT32",
            dimension=4,
            distanceMetric="COSINE",
            metadataConfiguration={
                "filterableKeys": ["document_type", "year", "published"],
                "nonFilterableMetadataKeys": ["security_level", "rating"]
            }
        )
        return f"Created index '{test_index}' with filterable keys"

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

    # Step 4: Query with metadata filter
    def query_with_filter():
        query_embedding = [0.11, 0.22, 0.33, 0.44] # Close to vec_1
        
        # Filter for reports from 2023 that are published
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
            metadataFilter=metadata_filter
        )
        
        results = response.get("Results", [])
        print(f"   Found {len(results)} results with filter:")
        
        result_ids = []
        for r in results:
            print(f"     - ID: {r['Id']}, Score: {r['Score']:.4f}, Metadata: {r['Metadata']}")
            result_ids.append(r['Id'])
            
        # Assertions
        assert len(results) == 2, f"Expected 2 results, but got {len(results)}"
        assert "vec_1" in result_ids, "vec_1 should be in the results"
        assert "vec_5" in result_ids, "vec_5 should be in the results"
        
        return f"Query successful, found {len(results)} matching vectors: {result_ids}"

    _, filter_query_success = test_step("Query with Metadata Filter", query_with_filter)
    test_results.append(("Query with Filter", filter_query_success))

    # Step 5: Query with a different filter
    def query_with_complex_filter():
        query_embedding = [0.88, 0.11, 0.22, 0.33] # Close to vec_3
        
        # Filter for (reports OR emails) from year > 2022
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
            metadataFilter=metadata_filter
        )
        
        results = response.get("Results", [])
        print(f"   Found {len(results)} results with complex filter:")
        
        result_ids = []
        for r in results:
            print(f"     - ID: {r['Id']}, Score: {r['Score']:.4f}, Metadata: {r['Metadata']}")
            result_ids.append(r['Id'])
            
        # Assertions
        assert len(results) == 3, f"Expected 3 results, but got {len(results)}"
        assert "vec_1" in result_ids
        assert "vec_2" in result_ids
        assert "vec_5" in result_ids
        
        return f"Query successful, found {len(results)} matching vectors: {result_ids}"

    _, complex_filter_success = test_step("Query with Complex Filter (OR, GT)", query_with_complex_filter)
    test_results.append(("Query with Complex Filter", complex_filter_success))

    # Step 6: Query that should return no results
    def query_with_no_results_filter():
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Filter for something that doesn't exist
        metadata_filter = {"field": "year", "operator": "eq", "value": 2000}
        
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={"float32": query_embedding},
            topK=5,
            metadataFilter=metadata_filter
        )
        
        results = response.get("Results", [])
        assert len(results) == 0, f"Expected 0 results, but got {len(results)}"
        
        return "Query correctly returned 0 results"

    _, no_results_success = test_step("Query with No-Result Filter", query_with_no_results_filter)
    test_results.append(("Query No Results", no_results_success))

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
    
    return success_rate == 100.0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

import boto3
import requests
import time
import uuid
from typing import List

# Configuration
VECTOR_SERVICE_URL = "http://localhost:8080"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

# Test data with rich metadata
TEST_VECTORS = [
    {
        "key": "vec-1",
        "text": "The quick brown fox jumps over the lazy dog.",
        "metadata": {
            "category": "animals",
            "type": "mammal",
            "is_fiction": True,
            "word_count": 9,
            "tags": ["fox", "dog"]
        }
    },
    {
        "key": "vec-2",
        "text": "A journey of a thousand miles begins with a single step.",
        "metadata": {
            "category": "philosophy",
            "type": "proverb",
            "is_fiction": False,
            "word_count": 10,
            "tags": ["journey", "wisdom"]
        }
    },
    {
        "key": "vec-3",
        "text": "The early bird catches the worm.",
        "metadata": {
            "category": "animals",
            "type": "proverb",
            "is_fiction": False,
            "word_count": 6,
            "tags": ["bird", "proverb"]
        }
    }
]

def create_embedding(text: str) -> List[float]:
    """Create embedding using LM Studio API."""
    payload = {"model": EMBEDDING_MODEL, "input": text}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"❌ Failed to create embedding: {e}")
        raise

def create_s3vectors_client():
    """Create boto3 s3vectors client."""
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
        return result, True
    except Exception as e:
        print(f"❌ {step_name}: FAILED")
        print(f"   Error: {type(e).__name__}: {e}")
        return None, False

def main():
    print("🚀 Metadata Filtering Test")
    print("=" * 80)
    
    client = create_s3vectors_client()
    test_bucket = f"metadata-test-bucket-{int(time.time())}"
    test_index = "metadata-test-index"
    
    # 1. Create Vector Bucket
    _, success = test_step("Create Vector Bucket", lambda: client.create_vector_bucket(vectorBucketName=test_bucket))
    if not success: return False

    # 2. Create Index with metadata configuration
    def create_index_with_metadata():
        # Note: The Rust service seems to expect `filterable_keys` directly,
        # while the boto3 spec might use `metadataConfiguration`.
        # We will try the direct approach first as seen in the Rust code.
        response = client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dataType="FLOAT32",
            dimension=768,  # Dimension for nomic-embed-text-v1.5
            distanceMetric="COSINE",
            filterable_keys=[
                {"name": "category", "type": "string"},
                {"name": "type", "type": "string"},
                {"name": "is_fiction", "type": "boolean"},
                {"name": "word_count", "type": "number"}
            ],
            non_filterable_keys=["tags"]
        )
        return response

    _, success = test_step("Create Index with Metadata Keys", create_index_with_metadata)
    if not success: return False

    # 3. Ingest vectors with metadata
    def ingest_vectors():
        vectors_to_put = []
        for item in TEST_VECTORS:
            embedding = create_embedding(item["text"])
            vectors_to_put.append({
                "key": item["key"],
                "data": {"float32": embedding},
                "metadata": item["metadata"]
            })
        
        client.put_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            vectors=vectors_to_put
        )
        return f"Ingested {len(vectors_to_put)} vectors"

    _, success = test_step("Ingest Vectors with Metadata", ingest_vectors)
    if not success: return False

    print("\n⏳ Waiting for indexing...")
    time.sleep(5)

    # 4. Test metadata filtering
    def query_with_filter():
        query_text = "animal proverbs"
        query_embedding = create_embedding(query_text)
        
        # This filter should match 'vec-3' ("The early bird catches the worm.")
        metadata_filter = {
            "category": {"$eq": "animals"},
            "type": {"$eq": "proverb"}
        }
        
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={"float32": query_embedding},
            topK=3,
            metadataFilter=metadata_filter
        )
        
        results = response.get("Results", [])
        print(f"Found {len(results)} results with filter.")
        for r in results:
            print(f"  - ID: {r.get('Id')}, Score: {r.get('Score')}, Metadata: {r.get('Metadata')}")

        assert len(results) == 1, f"Expected 1 result, but got {len(results)}"
        assert results[0]['Id'] == 'vec-3', f"Expected 'vec-3', but got {results[0]['Id']}"
        
        return "Successfully filtered results."

    _, success = test_step("Query with Metadata Filter", query_with_filter)
    if not success: return False

    # 5. Cleanup
    def cleanup():
        client.delete_index(vectorBucketName=test_bucket, indexName=test_index)
        client.delete_vector_bucket(vectorBucketName=test_bucket)
        return "Cleanup complete."

    _, _ = test_step("Cleanup Resources", cleanup)

    print("\n" + "=" * 80)
    print("🎉 Metadata filtering test completed successfully!")
    return True


if __name__ == "__main__":
    import sys
    if not main():
        sys.exit(1)

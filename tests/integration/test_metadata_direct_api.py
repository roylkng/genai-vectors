#!/usr/bin/env python3
"""
Direct API Test for Metadata Filtering

Tests our API directly without boto3 client validation to understand what parameters work.
"""

import requests
import json
import time
from typing import List, Dict, Any

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
VECTOR_SERVICE_URL = "http://localhost:8080"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

def create_embedding(text: str) -> List[float]:
    """Create embedding using LM Studio API."""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    response = requests.post(LM_STUDIO_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    embedding = data["data"][0]["embedding"]
    
    print(f"✅ Created embedding with {len(embedding)} dimensions")
    return embedding

def test_direct_api():
    """Test our API directly to understand parameter formats."""
    
    print("🚀 Direct API Metadata Test")
    print("=" * 50)
    
    test_bucket = f"direct-test-{int(time.time())}"
    test_index = "direct-metadata-index"
    
    # Step 1: Create bucket
    print("\n1. Creating bucket...")
    response = requests.post(f"{VECTOR_SERVICE_URL}/vector-bucket/{test_bucket}", json={})
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
        return
    
    # Step 2: Test different index creation parameter formats
    print("\n2. Testing index creation formats...")
    
    embedding = create_embedding("test embedding")
    
    # Format 1: Our current implementation format
    index_payload_1 = {
        "indexName": test_index,
        "vectorBucketName": test_bucket,
        "dataType": "FLOAT32",
        "dimension": len(embedding),
        "distanceMetric": "COSINE",
        "filterable_keys": [
            {"name": "category", "type": "string"},
            {"name": "rating", "type": "number"}
        ],
        "non_filterable_keys": ["description"]
    }
    
    print("   Trying format 1 (our implementation):")
    response = requests.post(f"{VECTOR_SERVICE_URL}/index", json=index_payload_1)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Format 1 works!")
        index_created = True
    else:
        print(f"   ❌ Format 1 failed: {response.text}")
        index_created = False
        
        # Format 2: AWS standard format
        print("   Trying format 2 (AWS standard):")
        index_payload_2 = {
            "indexName": test_index,
            "vectorBucketName": test_bucket,
            "dataType": "FLOAT32", 
            "dimension": len(embedding),
            "distanceMetric": "COSINE",
            "metadataConfiguration": {
                "filterableMetadataKeys": [
                    {"name": "category", "type": "string"},
                    {"name": "rating", "type": "number"}
                ],
                "nonFilterableMetadataKeys": ["description"]
            }
        }
        
        response = requests.post(f"{VECTOR_SERVICE_URL}/index", json=index_payload_2)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Format 2 works!")
            index_created = True
        else:
            print(f"   ❌ Format 2 failed: {response.text}")
            index_created = False
    
    if not index_created:
        print("❌ Could not create index with any format")
        return
    
    # Step 3: Put test vector
    print("\n3. Adding test vector...")
    vector_payload = {
        "indexName": test_index,
        "vectorBucketName": test_bucket,
        "vectors": [
            {
                "key": "test_doc_1",
                "data": {"float32": embedding},
                "metadata": {
                    "category": "technology",
                    "rating": 4.5,
                    "description": "A test document about technology"
                }
            }
        ]
    }
    
    response = requests.post(f"{VECTOR_SERVICE_URL}/vectors", json=vector_payload)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
        return
    
    # Wait for indexing
    print("\n4. Waiting for indexing...")
    time.sleep(3)
    
    # Step 4: Test query formats
    print("\n5. Testing query formats...")
    
    query_embedding = create_embedding("technology search")
    
    # Format 1: metadataFilter
    print("   Trying metadataFilter parameter:")
    query_payload_1 = {
        "indexName": test_index,
        "vectorBucketName": test_bucket,
        "queryVector": {"float32": query_embedding},
        "topK": 5,
        "metadataFilter": {"category": {"$eq": "technology"}}
    }
    
    response = requests.post(f"{VECTOR_SERVICE_URL}/query", json=query_payload_1)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"   ✅ metadataFilter works! Found {len(results.get('results', []))} results")
    else:
        print(f"   ❌ metadataFilter failed: {response.text}")
        
        # Format 2: filter
        print("   Trying filter parameter:")
        query_payload_2 = {
            "indexName": test_index,
            "vectorBucketName": test_bucket,
            "queryVector": {"float32": query_embedding},
            "topK": 5,
            "filter": {"category": {"$eq": "technology"}}
        }
        
        response = requests.post(f"{VECTOR_SERVICE_URL}/query", json=query_payload_2)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ filter works! Found {len(results.get('results', []))} results")
        else:
            print(f"   ❌ filter failed: {response.text}")
    
    # Cleanup
    print("\n6. Cleaning up...")
    requests.delete(f"{VECTOR_SERVICE_URL}/index/{test_bucket}/{test_index}")
    requests.delete(f"{VECTOR_SERVICE_URL}/vector-bucket/{test_bucket}")
    print("   ✅ Cleanup completed")

if __name__ == "__main__":
    test_direct_api()

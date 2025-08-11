#!/usr/bin/env python3
"""
Simple metadata test using direct boto3 client calls with our current implementation.
This bypasses the metadataConfiguration validation by not including it and focusing on testing filtering.
"""

import boto3
import requests
import time
import json

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
VECTOR_SERVICE_URL = "http://localhost:8080"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

def create_embedding(text: str):
    """Create embedding using LM Studio API."""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    response = requests.post(LM_STUDIO_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    embedding = data["data"][0]["embedding"]
    
    print(f"✅ Created {len(embedding)}-dim embedding")
    return embedding

def test_current_metadata_implementation():
    """Test our current metadata implementation without the metadataConfiguration."""
    
    print("🔬 Testing Current Metadata Implementation")
    print("=" * 60)
    
    # Create client
    client = boto3.client(
        's3vectors',
        endpoint_url=VECTOR_SERVICE_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )
    
    test_bucket = f"meta-test-{int(time.time())}"
    test_index = "simple-meta-index"
    
    try:
        # Step 1: Create bucket
        print("\n1. Creating vector bucket...")
        client.create_vector_bucket(vectorBucketName=test_bucket)
        print("   ✅ Bucket created")
        
        # Step 2: Create index without metadata configuration (let it use defaults)
        print("\n2. Creating index (no metadata config)...")
        embedding = create_embedding("test")
        
        client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dataType="FLOAT32",
            dimension=len(embedding),
            distanceMetric="COSINE"
        )
        print("   ✅ Index created")
        
        # Step 3: Put vectors with metadata
        print("\n3. Adding vectors with metadata...")
        
        test_docs = [
            {
                "key": "doc1",
                "text": "Machine learning is a powerful technology",
                "metadata": {
                    "category": "technology",
                    "rating": 4.5,
                    "published": True,
                    "tags": ["ml", "ai"]
                }
            },
            {
                "key": "doc2", 
                "text": "Climate change requires urgent action",
                "metadata": {
                    "category": "environment",
                    "rating": 4.2,
                    "published": True,
                    "tags": ["climate", "environment"]
                }
            }
        ]
        
        vectors = []
        for doc in test_docs:
            embedding = create_embedding(doc["text"])
            vectors.append({
                "key": doc["key"],
                "data": {"float32": embedding},
                "metadata": doc["metadata"]
            })
        
        client.put_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            vectors=vectors
        )
        print(f"   ✅ Added {len(vectors)} vectors")
        
        # Step 4: Wait and test queries
        print("\n4. Waiting for indexing...")
        time.sleep(5)
        
        # Step 5: Try basic query (no filter)
        print("\n5. Testing basic query (no filter)...")
        query_embedding = create_embedding("artificial intelligence")
        
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={"float32": query_embedding},
            topK=5
        )
        
        results = response.get("Results", [])
        print(f"   ✅ Found {len(results)} results without filter")
        
        for result in results:
            print(f"     - {result.get('Id')}: score {result.get('Score', 0):.3f}")
        
        # Step 6: Try query with filter (using 'filter' parameter)
        print("\n6. Testing filtered query (with filter)...")
        
        try:
            response = client.query_vectors(
                vectorBucketName=test_bucket,
                indexName=test_index,
                queryVector={"float32": query_embedding},
                topK=5,
                filter={"category": {"$eq": "technology"}}
            )
            
            results = response.get("Results", [])
            print(f"   ✅ Filtered query worked! Found {len(results)} results")
            
            for result in results:
                print(f"     - {result.get('Id')}: score {result.get('Score', 0):.3f}")
                
        except Exception as e:
            print(f"   ❌ Filtered query failed: {e}")
        
        # Step 7: Test metadata retrieval
        print("\n7. Testing metadata retrieval...")
        
        response = client.get_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=["doc1", "doc2"],
            returnMetadata=True
        )
        
        vectors = response.get("vectors", [])
        print(f"   ✅ Retrieved {len(vectors)} vectors with metadata")
        
        for vector in vectors:
            key = vector.get("key")
            metadata = vector.get("metadata", {})
            print(f"     - {key}: {metadata}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n8. Cleanup...")
        try:
            client.delete_index(vectorBucketName=test_bucket, indexName=test_index)
            client.delete_vector_bucket(vectorBucketName=test_bucket)
            print("   ✅ Cleanup completed")
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    test_current_metadata_implementation()

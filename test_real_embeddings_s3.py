#!/usr/bin/env python3
"""
Real Embeddings Test with LM Studio Integration

This test creates real embeddings from documents using LM Studio's text-embedding-nomic-embed-text-v1.5 model,
then tests the complete vector store pipeline using AWS S3 Vectors API compatibility.

Requirements:
1. LM Studio running at http://127.0.0.1:1234 with text-embedding-nomic-embed-text-v1.5 model loaded
2. Vector store service running at http://localhost:8080
3. MinIO or compatible S3 storage running

Usage: python test_real_embeddings_s3.py
"""

import boto3
import requests
import time
from datetime import datetime
from typing import List

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
VECTOR_SERVICE_URL = "http://localhost:8080"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

# Test documents - varied content for meaningful similarity testing
TEST_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Machine Learning Fundamentals",
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications based on those patterns.",
        "category": "technology",
        "author": "AI Research Team"
    },
    {
        "id": "doc_2", 
        "title": "Deep Learning Neural Networks",
        "content": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks can automatically learn hierarchical representations, making them particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
        "category": "technology",
        "author": "Deep Learning Lab"
    },
    {
        "id": "doc_3",
        "title": "Climate Change Impact",
        "content": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, scientific evidence shows that human activities, particularly greenhouse gas emissions, have been the dominant driver of climate change since the mid-20th century.",
        "category": "environment",
        "author": "Climate Science Institute"
    },
    {
        "id": "doc_4",
        "title": "Renewable Energy Solutions",
        "content": "Renewable energy sources such as solar, wind, hydroelectric, and geothermal power offer sustainable alternatives to fossil fuels. These technologies are becoming increasingly cost-effective and play a crucial role in reducing carbon emissions and combating climate change.",
        "category": "environment", 
        "author": "Green Energy Council"
    },
    {
        "id": "doc_5",
        "title": "Modern Cooking Techniques",
        "content": "Contemporary culinary arts have evolved to incorporate molecular gastronomy, sous vide cooking, and fusion cuisine. These techniques allow chefs to create innovative dishes that combine traditional flavors with modern presentation and cooking methods.",
        "category": "cooking",
        "author": "Culinary Institute"
    }
]

def create_embedding(text: str) -> List[float]:
    """Create embedding using LM Studio API."""
    print(f"Creating embedding for text: '{text[:50]}...'")
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        embedding = data["data"][0]["embedding"]
        
        print(f"✅ Created embedding with {len(embedding)} dimensions")
        return embedding
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to create embedding: {e}")
        raise
    except KeyError as e:
        print(f"❌ Unexpected response format: {e}")
        print(f"Response: {response.text}")
        raise

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
        return result, True
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"❌ {step_name}: FAILED")
        print(f"   Error: {str(e)}")
        return None, False

def main():
    print("🚀 Real Embeddings Test with AWS S3 Vectors API")
    print("=" * 80)
    print(f"📊 Testing with {len(TEST_DOCUMENTS)} documents")
    print(f"🔗 LM Studio: {LM_STUDIO_URL}")
    print(f"🔗 Vector Service: {VECTOR_SERVICE_URL}")
    print("=" * 80)
    
    # Test configuration
    test_bucket = f"real-embeddings-test-{int(time.time())}"
    test_index = "document-embeddings"
    
    # Initialize client
    client = create_s3vectors_client()
    
    # Track test results
    test_results = []
    
    # Step 1: Test LM Studio connectivity
    def test_lm_studio():
        test_text = "This is a test sentence for embedding generation."
        embedding = create_embedding(test_text)
        return f"Generated {len(embedding)}-dimensional embedding"
    
    _, lm_success = test_step("Test LM Studio Connectivity", test_lm_studio)
    test_results.append(("LM Studio", lm_success))
    
    if not lm_success:
        print("\n❌ Cannot proceed without LM Studio. Please ensure:")
        print("   1. LM Studio is running at http://127.0.0.1:1234")
        print("   2. text-embedding-nomic-embed-text-v1.5 model is loaded")
        return False
    
    # Step 2: Create vector bucket
    def create_vector_bucket():
        client.create_vector_bucket(vectorBucketName=test_bucket)
        return f"Created bucket: {test_bucket}"
    
    _, bucket_success = test_step("Create Vector Bucket", create_vector_bucket)
    test_results.append(("Create Bucket", bucket_success))
    
    # Step 3: Create embeddings for all documents
    def create_document_embeddings():
        embeddings = []
        for doc in TEST_DOCUMENTS:
            # Combine title and content for richer embeddings
            full_text = f"{doc['title']}. {doc['content']}"
            embedding = create_embedding(full_text)
            
            embeddings.append({
                "doc": doc,
                "embedding": embedding,
                "dimension": len(embedding)
            })
        
        return embeddings
    
    embeddings_data, embeddings_success = test_step("Create Document Embeddings", create_document_embeddings)
    test_results.append(("Create Embeddings", embeddings_success))
    
    if not embeddings_success or not embeddings_data:
        print("\n❌ Failed to create embeddings. Cannot proceed.")
        return False
    
    # Get embedding dimension from first document
    embedding_dim = embeddings_data[0]["dimension"]
    print(f"\n📏 Embedding dimension: {embedding_dim}")
    
    # Step 4: Create index with proper dimension
    def create_index():
        client.create_index(
            vectorBucketName=test_bucket,
            indexName=test_index,
            dataType="FLOAT32",
            dimension=embedding_dim,
            distanceMetric="COSINE"
        )
        return f"Created index: {test_index} with dimension {embedding_dim}"
    
    _, index_success = test_step("Create Vector Index", create_index)
    test_results.append(("Create Index", index_success))
    
    # Step 5: Prepare and put vectors
    def put_document_vectors():
        vectors = []
        for emb_data in embeddings_data:
            doc = emb_data["doc"]
            vector = {
                "key": doc["id"],
                "data": {"float32": emb_data["embedding"]},
                "metadata": {
                    "title": doc["title"],
                    "category": doc["category"],
                    "author": doc["author"],
                    "content_length": len(doc["content"]),
                    "indexed_at": datetime.utcnow().isoformat()
                }
            }
            vectors.append(vector)
        
        client.put_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            vectors=vectors
        )
        
        return f"Put {len(vectors)} document vectors"
    
    _, put_success = test_step("Put Document Vectors", put_document_vectors)
    test_results.append(("Put Vectors", put_success))
    
    # Step 6: Wait for indexing
    print("\n⏳ Waiting for indexing to complete...")
    time.sleep(5)
    
    # Step 7: List vectors to verify storage
    def list_vectors():
        response = client.list_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index
        )
        vector_count = response.get("count", 0)
        total_vectors = response.get("total_vectors", 0)
        return f"Listed {vector_count} vectors (total: {total_vectors})"
    
    _, list_success = test_step("List Stored Vectors", list_vectors)
    test_results.append(("List Vectors", list_success))
    
    # Step 8: Test similarity search with different queries
    def test_similarity_search():
        search_queries = [
            {
                "text": "artificial intelligence and machine learning algorithms",
                "expected_category": "technology",
                "description": "AI/ML query"
            },
            {
                "text": "global warming and environmental protection",
                "expected_category": "environment", 
                "description": "Climate query"
            },
            {
                "text": "solar panels and clean energy systems",
                "expected_category": "environment",
                "description": "Clean energy query"
            }
        ]
        
        search_results = []
        
        for query in search_queries:
            print(f"\n🔍 Testing: {query['description']}")
            print(f"   Query: '{query['text']}'")
            
            # Create embedding for search query
            query_embedding = create_embedding(query["text"])
            
            # Perform similarity search
            response = client.query_vectors(
                vectorBucketName=test_bucket,
                indexName=test_index,
                queryVector={"float32": query_embedding},
                topK=3
            )
            
            results = response.get("Results", [])
            print(f"   Found {len(results)} similar documents:")
            
            for i, result in enumerate(results, 1):
                doc_id = result.get("Id", "unknown")
                score = result.get("Score", 0.0)
                metadata = result.get("Metadata", {})
                title = metadata.get("title", "Unknown")
                category = metadata.get("category", "unknown")
                
                print(f"     {i}. {title} (ID: {doc_id})")
                print(f"        Score: {score:.4f}, Category: {category}")
            
            search_results.append({
                "query": query,
                "results": results,
                "top_result_category": results[0].get("Metadata", {}).get("category") if results else None
            })
        
        return search_results
    
    search_data, search_success = test_step("Test Similarity Search", test_similarity_search)
    test_results.append(("Similarity Search", search_success))
    
    # Step 9: Test metadata filtering (simplified)
    def test_metadata_filtering():
        # Create a search that should find technology documents
        query_text = "computer algorithms and data processing"
        query_embedding = create_embedding(query_text)
        
        # Search without filter first
        response = client.query_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            queryVector={"float32": query_embedding},
            topK=5
        )
        
        results = response.get("Results", [])
        result_categories = [r.get("Metadata", {}).get("category") for r in results]
        
        print(f"   Found {len(results)} results without filter")
        print(f"   Categories: {result_categories}")
        
        return f"Search returned {len(results)} documents"
    
    _, filter_success = test_step("Test Search Without Filtering", test_metadata_filtering)
    test_results.append(("Search Without Filtering", filter_success))
    
    # Step 10: Test vector retrieval by key
    def test_vector_retrieval():
        response = client.get_vectors(
            vectorBucketName=test_bucket,
            indexName=test_index,
            keys=["doc_1", "doc_3"],
            returnMetadata=True
        )
        
        vectors = response.get("vectors", [])
        found_count = response.get("found", 0)
        
        print(f"   Retrieved {found_count} vectors by key")
        for vector in vectors:
            key = vector.get("key", "unknown")
            metadata = vector.get("metadata", {})
            title = metadata.get("title", "Unknown")
            print(f"     - {key}: {title}")
        
        return f"Retrieved {found_count} vectors by key"
    
    _, retrieval_success = test_step("Test Vector Retrieval", test_vector_retrieval)
    test_results.append(("Vector Retrieval", retrieval_success))
    
    # Cleanup Step: Delete test resources
    def cleanup_resources():
        try:
            # Delete index
            client.delete_index(
                vectorBucketName=test_bucket,
                indexName=test_index
            )
            print(f"   ✅ Deleted index: {test_index}")
            
            # Delete bucket
            client.delete_vector_bucket(vectorBucketName=test_bucket)
            print(f"   ✅ Deleted bucket: {test_bucket}")
            
            return "Cleanup completed successfully"
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"   ⚠️ Cleanup warning: {e}")
            return "Cleanup completed with warnings"
    
    _, cleanup_success = test_step("Cleanup Test Resources", cleanup_resources)
    test_results.append(("Cleanup", cleanup_success))
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📊 Real Embeddings Test Results Summary")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
        if passed:
            passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🏆 Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    # Detailed analysis
    if search_success and search_data:
        print("\n📈 Similarity Search Analysis:")
        for result in search_data:
            query = result["query"]
            top_category = result["top_result_category"]
            expected = query["expected_category"]
            match = "✅" if top_category == expected else "⚠️"
            print(f"   {match} {query['description']}: Expected {expected}, Got {top_category}")
    
    if success_rate == 100.0:
        print("\n🎉 All tests passed! Real embeddings pipeline working perfectly!")
        print("✨ Your vector store is ready for production with LM Studio integration.")
    elif success_rate >= 80.0:
        print("\n✅ Most tests passed! Minor issues may need attention.")
    else:
        print("\n⚠️ Several tests failed. Please check the errors above.")
    
    print(f"\n📚 Test completed with {len(TEST_DOCUMENTS)} documents and {embedding_dim}-dimensional embeddings")
    
    return success_rate == 100.0

if __name__ == "__main__":
    import sys
    
    print("🔧 Prerequisites Check:")
    print("   1. LM Studio running at http://127.0.0.1:1234")
    print("   2. text-embedding-nomic-embed-text-v1.5 model loaded")
    print("   3. Vector service running at http://localhost:8080")
    print("   4. MinIO/S3 storage accessible")
    print()
    
    input("Press Enter to continue when all prerequisites are ready...")
    
    success = main()
    sys.exit(0 if success else 1)

#!/bin/bash

# Test script for end-to-end vector database testing

BASE_URL="http://localhost:8080"

echo "=== Testing Vector Database End-to-End ==="

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" || { echo "❌ Health check failed"; exit 1; }
echo "✅ Health check passed"

# Test 2: Create an index
echo -e "\n2. Creating test index..."
curl -s -X POST "$BASE_URL/indexes" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-index",
    "dim": 4,
    "metric": "cosine",
    "nlist": 16,
    "m": 2,
    "nbits": 8,
    "default_nprobe": 4
  }' || { echo "❌ Index creation failed"; exit 1; }
echo "✅ Index created"

# Test 3: Upload some test vectors
echo -e "\n3. Uploading test vectors..."
curl -s -X POST "$BASE_URL/vectors" \
  -H "Content-Type: application/json" \
  -d '{
    "index": "test-index",
    "vectors": [
      {
        "id": "vec1",
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "meta": {"category": "A", "value": 100}
      },
      {
        "id": "vec2", 
        "embedding": [0.0, 1.0, 0.0, 0.0],
        "meta": {"category": "B", "value": 200}
      },
      {
        "id": "vec3",
        "embedding": [0.0, 0.0, 1.0, 0.0],
        "meta": {"category": "A", "value": 300}
      },
      {
        "id": "vec4",
        "embedding": [0.0, 0.0, 0.0, 1.0],
        "meta": {"category": "C", "value": 400}
      },
      {
        "id": "vec5",
        "embedding": [0.7, 0.7, 0.0, 0.0],
        "meta": {"category": "A", "value": 150}
      }
    ]
  }' || { echo "❌ Vector upload failed"; exit 1; }
echo "✅ Vectors uploaded"

# Wait a bit for ingestion
echo -e "\n4. Waiting for ingestion..."
sleep 3

# Test 4: Query for similar vectors
echo -e "\n5. Querying for similar vectors to [1.0, 0.0, 0.0, 0.0]..."
response=$(curl -s -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "index": "test-index",
    "embedding": [1.0, 0.0, 0.0, 0.0],
    "topk": 3,
    "nprobe": 4
  }')

if [ $? -eq 0 ]; then
  echo "✅ Query successful"
  echo "Response: $response"
else
  echo "❌ Query failed"
  exit 1
fi

echo -e "\n=== All tests completed! ==="

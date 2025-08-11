#!/bin/bash

# Real Embeddings Test Setup Script
# This script helps set up and run the real embeddings test

set -e

echo "🚀 Real Embeddings Test Setup"
echo "==============================="

# Check if LM Studio is accessible
echo "🔍 Checking LM Studio connectivity..."
if curl -s "http://127.0.0.1:1234/v1/models" > /dev/null 2>&1; then
    echo "✅ LM Studio is accessible at http://127.0.0.1:1234"
    
    # Try to get models list
    echo "📋 Available models:"
    curl -s "http://127.0.0.1:1234/v1/models" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    for model in models:
        print(f'   - {model.get(\"id\", \"unknown\")}')
    if not models:
        print('   No models found - please load text-embedding-nomic-embed-text-v1.5')
except:
    print('   Unable to parse models response')
" 2>/dev/null || echo "   Unable to get models list"
else
    echo "❌ LM Studio not accessible at http://127.0.0.1:1234"
    echo "   Please start LM Studio and load the text-embedding-nomic-embed-text-v1.5 model"
    exit 1
fi

# Check if vector service is running
echo
echo "🔍 Checking vector service..."
if curl -s "http://localhost:8080/CreateVectorBucket" -X POST -H "Content-Type: application/json" -d '{"vectorBucketName":"test"}' > /dev/null 2>&1; then
    echo "✅ Vector service is accessible at http://localhost:8080"
else
    echo "❌ Vector service not accessible at http://localhost:8080"
    echo "   Please start the vector service with: cargo run --release"
    exit 1
fi

# Check if MinIO is running (optional - service might use other S3)
echo
echo "🔍 Checking MinIO/S3 storage..."
if curl -s "http://localhost:9000/minio/health/live" > /dev/null 2>&1; then
    echo "✅ MinIO is accessible at http://localhost:9000"
elif curl -s "http://localhost:9001" > /dev/null 2>&1; then
    echo "✅ MinIO console is accessible at http://localhost:9001"
else
    echo "⚠️  MinIO not detected - service may be using alternative S3 storage"
fi

# Install Python dependencies if needed
echo
echo "🔍 Checking Python dependencies..."
python3 -c "import boto3, requests" 2>/dev/null && echo "✅ Python dependencies (boto3, requests) are available" || {
    echo "❌ Missing Python dependencies. Installing..."
    pip3 install boto3 requests
}

echo
echo "✅ Environment check complete!"
echo
echo "🚀 Starting real embeddings test..."
echo

# Run the test
python3 test_real_embeddings_s3.py

echo
echo "📊 Test completed!"

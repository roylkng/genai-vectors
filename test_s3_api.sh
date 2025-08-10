#!/bin/bash
# S3 Vectors API Test Script
# Tests all 13 AWS S3 Vectors operations using curl

set -e

# Configuration
BASE_URL="http://localhost:8080"
BUCKET="cli-test-$(date +%s)"
INDEX="test-embeddings"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to test API endpoint
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local payload="$3"
    local expected_status="${4:-200}"
    
    print_status "Testing $name..."
    
    # Use temp files for better compatibility
    temp_response=$(mktemp)
    
    http_code=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/$endpoint" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        -o "$temp_response")
    
    body=$(cat "$temp_response")
    rm -f "$temp_response"
    
    if [[ "$http_code" == "$expected_status" ]]; then
        print_success "$name passed (HTTP $http_code)"
        if command -v jq &> /dev/null && [[ -n "$body" ]]; then
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            echo "$body"
        fi
    else
        print_error "$name failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
    echo
}

# Check if service is running
check_service() {
    print_status "Checking if S3 vectors service is running..."
    if curl -s -f "http://localhost:8080/health" > /dev/null 2>&1; then
        print_success "Service is running"
    else
        print_error "Service is not running on http://localhost:8080"
        echo "Please start the service first:"
        echo "  AWS_ENDPOINT_URL=http://localhost:9000 \\"
        echo "  AWS_ACCESS_KEY_ID=minioadmin \\"
        echo "  AWS_SECRET_ACCESS_KEY=minioadmin \\"
        echo "  ./target/release/genai-vectors api"
        exit 1
    fi
    echo
}

# Main test sequence
main() {
    echo "🚀 S3 Vectors API Comprehensive Test"
    echo "===================================="
    echo "Bucket: $BUCKET"
    echo "Index: $INDEX"
    echo "Base URL: $BASE_URL"
    echo

    check_service

    # Test all 13 operations
    
    # 1. Create Vector Bucket
    test_endpoint "create-vector-bucket" "CreateVectorBucket" \
        "{\"vectorBucketName\": \"$BUCKET\"}"

    # 2. List Vector Buckets
    test_endpoint "list-vector-buckets" "ListVectorBuckets" \
        "{}"

    # 3. Get Vector Bucket
    test_endpoint "get-vector-bucket" "GetVectorBucket" \
        "{\"vectorBucketName\": \"$BUCKET\"}"

    # 4. Create Index
    test_endpoint "create-index" "CreateIndex" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\",
            \"dimension\": 1536,
            \"distanceMetric\": \"COSINE\",
            \"dataType\": \"FLOAT32\"
        }"

    # 5. List Indexes
    test_endpoint "list-indexes" "ListIndexes" \
        "{\"vectorBucketName\": \"$BUCKET\"}"

    # 6. Get Index
    test_endpoint "get-index" "GetIndex" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\"
        }"

    # 7. Put Vectors
    test_endpoint "put-vectors" "PutVectors" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\",
            \"vectors\": [
                {
                    \"key\": \"doc-1\",
                    \"data\": {
                        \"float32\": [0.1, 0.2, 0.3, 0.4, 0.5]
                    },
                    \"metadata\": {
                        \"title\": \"Test Document 1\",
                        \"category\": \"test\",
                        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
                    }
                },
                {
                    \"key\": \"doc-2\",
                    \"data\": {
                        \"float32\": [0.6, 0.7, 0.8, 0.9, 1.0]
                    },
                    \"metadata\": {
                        \"title\": \"Test Document 2\",
                        \"category\": \"test\",
                        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
                    }
                }
            ]
        }"

    # Wait for indexing
    print_status "Waiting 3 seconds for indexing to complete..."
    sleep 3
    echo

    # 8. List Vectors
    test_endpoint "list-vectors" "ListVectors" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\",
            \"maxResults\": 10
        }"

    # 9. Get Vectors
    test_endpoint "get-vectors" "GetVectors" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\",
            \"keys\": [\"doc-1\", \"doc-2\"],
            \"returnData\": true,
            \"returnMetadata\": true
        }"

    # 10. Query Vectors
    test_endpoint "query-vectors" "QueryVectors" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\",
            \"queryVector\": {
                \"float32\": [0.15, 0.25, 0.35, 0.45, 0.55]
            },
            \"topK\": 5,
            \"returnData\": true,
            \"returnMetadata\": true
        }"

    # 11. Delete Vectors
    test_endpoint "delete-vectors" "DeleteVectors" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\",
            \"keys\": [\"doc-1\"]
        }"

    # 12. Delete Index
    test_endpoint "delete-index" "DeleteIndex" \
        "{
            \"vectorBucketName\": \"$BUCKET\",
            \"indexName\": \"$INDEX\"
        }"

    # 13. Delete Vector Bucket
    test_endpoint "delete-vector-bucket" "DeleteVectorBucket" \
        "{\"vectorBucketName\": \"$BUCKET\"}"

    echo "🎉 All tests completed successfully!"
    echo "✅ 13/13 S3 Vectors API operations verified"
}

# Check for required tools
if ! command -v curl &> /dev/null; then
    print_error "curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    print_warning "jq is not installed - JSON output will not be formatted"
fi

# Run tests
main "$@"

#!/bin/bash
# AWS CLI S3 Vectors API Test Script
# Tests all 13 AWS S3 Vectors operations using the actual AWS CLI

set -e

# Configuration
ENDPOINT_URL="http://localhost:8080"
BUCKET="aws-cli-test-$(date +%s)"
INDEX="test-embeddings"
REGION="us-east-1"

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

# Function to test AWS CLI command
test_aws_command() {
    local name="$1"
    local command="$2"
    
    print_status "Testing $name..."
    
    if eval "$command"; then
        print_success "$name passed"
    else
        print_error "$name failed"
        return 1
    fi
    echo
}

# Check if AWS CLI and service are available
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is required but not installed"
        echo "Install with: pip install awscli"
        exit 1
    fi
    
    # Check if service is running
    if curl -s -f "http://localhost:8080/health" > /dev/null 2>&1; then
        print_success "S3 vectors service is running"
    else
        print_error "S3 vectors service is not running on http://localhost:8080"
        echo "Please start the service first:"
        echo "  AWS_ENDPOINT_URL=http://localhost:9000 \\"
        echo "  AWS_ACCESS_KEY_ID=minioadmin \\"
        echo "  AWS_SECRET_ACCESS_KEY=minioadmin \\"
        echo "  ./target/release/genai-vectors api"
        exit 1
    fi
    
    # Configure AWS CLI for local testing
    export AWS_ACCESS_KEY_ID=test
    export AWS_SECRET_ACCESS_KEY=test
    export AWS_DEFAULT_REGION=$REGION
    
    print_success "Prerequisites check completed"
    echo
}

# Create temporary JSON files for complex payloads
create_temp_files() {
    # Put vectors payload
    cat > /tmp/put_vectors.json << EOF
{
    "vectorBucketName": "$BUCKET",
    "indexName": "$INDEX",
    "vectors": [
        {
            "key": "doc-1",
            "data": {
                "float32": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "metadata": {
                "title": "Test Document 1",
                "category": "test",
                "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            }
        },
        {
            "key": "doc-2",
            "data": {
                "float32": [0.6, 0.7, 0.8, 0.9, 1.0]
            },
            "metadata": {
                "title": "Test Document 2",
                "category": "test",
                "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            }
        }
    ]
}
EOF

    # Query vectors payload
    cat > /tmp/query_vectors.json << EOF
{
    "vectorBucketName": "$BUCKET",
    "indexName": "$INDEX",
    "queryVector": {
        "float32": [0.15, 0.25, 0.35, 0.45, 0.55]
    },
    "topK": 5,
    "returnData": true,
    "returnMetadata": true
}
EOF
}

# Cleanup temporary files
cleanup_temp_files() {
    rm -f /tmp/put_vectors.json /tmp/query_vectors.json
}

# Main test sequence
main() {
    echo "🚀 AWS CLI S3 Vectors API Comprehensive Test"
    echo "============================================"
    echo "Bucket: $BUCKET"
    echo "Index: $INDEX"
    echo "Endpoint: $ENDPOINT_URL"
    echo

    check_prerequisites
    create_temp_files

    # Test all 13 operations using AWS CLI
    
    # 1. Create Vector Bucket
    test_aws_command "create-vector-bucket" \
        "aws s3vectors create-vector-bucket --vector-bucket-name '$BUCKET' --endpoint-url '$ENDPOINT_URL'"

    # 2. List Vector Buckets
    test_aws_command "list-vector-buckets" \
        "aws s3vectors list-vector-buckets --endpoint-url '$ENDPOINT_URL'"

    # 3. Get Vector Bucket
    test_aws_command "get-vector-bucket" \
        "aws s3vectors get-vector-bucket --vector-bucket-name '$BUCKET' --endpoint-url '$ENDPOINT_URL'"

    # 4. Create Index
    test_aws_command "create-index" \
        "aws s3vectors create-index --vector-bucket-name '$BUCKET' --index-name '$INDEX' --dimension 1536 --distance-metric COSINE --data-type FLOAT32 --endpoint-url '$ENDPOINT_URL'"

    # 5. List Indexes
    test_aws_command "list-indexes" \
        "aws s3vectors list-indexes --vector-bucket-name '$BUCKET' --endpoint-url '$ENDPOINT_URL'"

    # 6. Get Index
    test_aws_command "get-index" \
        "aws s3vectors get-index --vector-bucket-name '$BUCKET' --index-name '$INDEX' --endpoint-url '$ENDPOINT_URL'"

    # 7. Put Vectors
    test_aws_command "put-vectors" \
        "aws s3vectors put-vectors --cli-input-json file:///tmp/put_vectors.json --endpoint-url '$ENDPOINT_URL'"

    # Wait for indexing
    print_status "Waiting 3 seconds for indexing to complete..."
    sleep 3
    echo

    # 8. List Vectors
    test_aws_command "list-vectors" \
        "aws s3vectors list-vectors --vector-bucket-name '$BUCKET' --index-name '$INDEX' --max-results 10 --endpoint-url '$ENDPOINT_URL'"

    # 9. Get Vectors
    test_aws_command "get-vectors" \
        "aws s3vectors get-vectors --vector-bucket-name '$BUCKET' --index-name '$INDEX' --keys 'doc-1' 'doc-2' --return-data --return-metadata --endpoint-url '$ENDPOINT_URL'"

    # 10. Query Vectors
    test_aws_command "query-vectors" \
        "aws s3vectors query-vectors --cli-input-json file:///tmp/query_vectors.json --endpoint-url '$ENDPOINT_URL'"

    # 11. Delete Vectors
    test_aws_command "delete-vectors" \
        "aws s3vectors delete-vectors --vector-bucket-name '$BUCKET' --index-name '$INDEX' --keys 'doc-1' --endpoint-url '$ENDPOINT_URL'"

    # 12. Delete Index
    test_aws_command "delete-index" \
        "aws s3vectors delete-index --vector-bucket-name '$BUCKET' --index-name '$INDEX' --endpoint-url '$ENDPOINT_URL'"

    # 13. Delete Vector Bucket
    test_aws_command "delete-vector-bucket" \
        "aws s3vectors delete-vector-bucket --vector-bucket-name '$BUCKET' --endpoint-url '$ENDPOINT_URL'"

    cleanup_temp_files

    echo "🎉 All AWS CLI tests completed successfully!"
    echo "✅ 13/13 S3 Vectors API operations verified with AWS CLI"
}

# Trap to ensure cleanup on exit
trap cleanup_temp_files EXIT

# Run tests
main "$@"

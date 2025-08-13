# GenAI Vector Database Makefile

.PHONY: help build test test-integration test-compatibility clean deploy dev delete setup

# Default target
help:
	@echo "GenAI Vector Database - Available targets:"
	@echo ""
	@echo "Build & Development:"
	@echo "  build              Build the release version"
	@echo "  build-dev          Build the development version"
	@echo "  run-api            Start the API server"
	@echo "  run-indexer        Run the indexer once"
	@echo ""
	@echo "Testing:"
	@echo "  test               Run all tests"
	@echo "  test-unit          Run unit tests only"
	@echo "  test-integration   Run integration tests"
	@echo "  test-compatibility Run S3 API compatibility test"
	@echo "  test-real          Run real embeddings test (requires LM Studio)"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy             Deploy MinIO to Kubernetes"
	@echo "  deploy-all         Deploy MinIO and vector store"
	@echo "  dev                Start development environment"
	@echo "  delete             Delete all deployments"
	@echo ""
	@echo "Utilities:"
	@echo "  setup              Set up development environment"
	@echo "  clean              Clean build artifacts and temporary files"
	@echo "  logs               Show service logs"

# Build targets
build:
	cargo build --release

build-dev:
	cargo build

# Run targets
run-api:
	cargo run --release api

run-indexer:
	cargo run --release indexer

# Test targets
test: test-unit test-integration

test-unit:
	cargo test
	pytest tests/test_small_scale.py tests/test_large_scale.py

test-integration:
	@echo "Running S3 compatibility test..."
	@cd tests/integration && python test_s3_compatibility.py

test-compatibility: test-integration

test-real:
	@echo "Running real embeddings test (requires LM Studio)..."
	@cd tests/integration && ./run_real_embeddings_test.sh

# Deployment targets  
deploy:
	skaffold deploy -p minio

deploy-all:
	skaffold deploy -p minio
	skaffold deploy -p vector-store

dev:
	skaffold dev -p vector-store --port-forward

delete:
	skaffold delete -p minio
	skaffold delete -p qdrant
	skaffold delete -p vector-store

<<<<<<< HEAD
deploy:
	skaffold deploy -p minio
	skaffold deploy -p qdrant
=======
# Utility targets
setup:
	@echo "Setting up development environment..."
	@./scripts/setup.sh
>>>>>>> b4defcdf9022a3cdb3403fbeb6bc66d9648e389d

clean:
	cargo clean
	rm -rf target/
	rm -rf data/temp/*
	rm -f data/logs/*.log
	rm -f vector_service.log

logs:
	@echo "Recent vector service logs:"
	@tail -50 vector_service.log 2>/dev/null || echo "No logs found"

# Port forwarding for local development
port-forward-minio:
	kubectl port-forward -n genai-vectors service/minio 9000:9000

# Environment setup for local testing
env-local:
	@echo "export AWS_ACCESS_KEY_ID=minioadmin"
	@echo "export AWS_SECRET_ACCESS_KEY=minioadmin"  
	@echo "export AWS_ENDPOINT_URL=http://localhost:9000"
	@echo "export VEC_BUCKET=vectors"
	@echo ""
	@echo "Run: eval \$$(make env-local)"

#!/bin/bash
# Setup script for GenAI Vector Database development environment

set -e

echo "🚀 Setting up GenAI Vector Database development environment..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Some features may not work."
fi

# Install Python test dependencies
echo "📦 Setting up Python virtual environment..."
if [ -d ".venv" ]; then
    echo "✅ Using existing virtual environment"
    source .venv/bin/activate
else
    echo "⚠️  Creating new virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
fi

echo "📦 Installing Python test dependencies..."
pip install boto3 requests pytest pytest-asyncio || {
    echo "⚠️  Failed to install some Python packages"
    echo "   You may need to install them manually:"
    echo "   pip install boto3 requests pytest pytest-asyncio"
}

# Build Rust project
echo "🔨 Building Rust project..."
cargo build

# Create local config if it doesn't exist
if [ ! -f "config/local.toml" ]; then
    echo "📋 Creating local configuration..."
    cp config/development.toml config/local.toml
    echo "✏️  Please edit config/local.toml with your configuration"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from example..."
    cp .env.example .env
    echo "✏️  Please edit .env with your AWS credentials"
fi

# Create data directory if it doesn't exist
mkdir -p data/temp
mkdir -p data/indexes
mkdir -p data/logs

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Set up MinIO: kubectl port-forward -n genai-vectors service/minio 9000:9000"
echo "2. Set environment variables:"
echo "   export AWS_ACCESS_KEY_ID=minioadmin"
echo "   export AWS_SECRET_ACCESS_KEY=minioadmin"
echo "   export AWS_ENDPOINT_URL=http://localhost:9000"
echo "3. Start API server: cargo run --release api"
echo "4. Run tests: make test-compatibility"
echo "5. For real embeddings test: Start LM Studio and run make test-real"

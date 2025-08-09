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
pip install boto3 pytest pytest-asyncio numpy requests --no-deps || {
    echo "⚠️  Some packages may have failed, continuing..."
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
echo "Next steps:"
echo "1. Edit .env with your AWS credentials, OR"
echo "2. Edit config/local.toml with your configuration, OR" 
echo "3. Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
echo "4. Run tests: source .venv/bin/activate && python -m pytest tests/ -v"
echo "5. Start development server: cargo run"
echo "6. Or use Docker: docker-compose up"

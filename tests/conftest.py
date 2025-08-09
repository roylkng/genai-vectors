"""
Shared test configuration and utilities for vector database tests.
"""

import pytest
import boto3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from botocore.exceptions import ClientError

# Import configuration constants
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from constants import (
    get_aws_credentials, get_s3_config, get_test_config,
    validate_configuration, print_config_summary
)

# Test configuration from constants
TEST_CONFIG = get_test_config()

@pytest.fixture(scope="session")
def aws_credentials():
    """Ensure AWS credentials are configured."""
    if not validate_configuration():
        pytest.skip("AWS credentials not configured properly")

@pytest.fixture(scope="session")
def s3_client():
    """Provide S3 client for tests."""
    s3_config = get_s3_config()
    try:
        client = boto3.client('s3', **s3_config)
        # Test connection
        client.list_buckets()
        return client
    except Exception as e:
        pytest.skip(f"Could not create S3 client: {e}")

@pytest.fixture(scope="session")
def s3_config():
    """Provide S3 configuration for tests."""
    return get_s3_config()

@pytest.fixture(scope="session") 
def cached_vectors():
    """Load cached vectors if available."""
    cache_dir = TEST_CONFIG['cache_dir']
    cached_data = {}
    
    # Load different cached vector sets
    cache_files = {
        'test': cache_dir / 'embedding_cache_test.pkl',
        '5k': cache_dir / 'embeddings_cache_5k_s3vectors.pkl', 
        '500k': cache_dir / 'embedding_cache_500k.pkl'
    }
    
    for name, file_path in cache_files.items():
        if file_path.exists():
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    cached_data[name] = pickle.load(f)
                print(f"Loaded cached vectors: {name} ({len(cached_data[name])} vectors)")
            except Exception as e:
                print(f"Warning: Could not load {name} cache: {e}")
    
    return cached_data

@pytest.fixture
def test_vectors():
    """Generate test vectors for small-scale tests."""
    def _generate(count: int, dimension: int = TEST_CONFIG['test_dimension']):
        vectors = []
        for i in range(count):
            vector = np.random.rand(dimension).astype(np.float32).tolist()
            vectors.append({
                'id': f'test_vec_{i:06d}',
                'embedding': vector,
                'metadata': {
                    'category': np.random.choice(['A', 'B', 'C']),
                    'value': np.random.randint(1, 100),
                    'timestamp': 1691500000 + i,
                    'text': f'Test document {i}'
                }
            })
        return vectors
    return _generate

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "large_scale: marks tests as large scale"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on patterns."""
    for item in items:
        # Mark large scale tests
        if "large_scale" in item.nodeid or "500k" in item.nodeid:
            item.add_marker(pytest.mark.large_scale)
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "test_" in item.name and "integration" not in item.keywords:
            item.add_marker(pytest.mark.integration)

def pytest_sessionstart(session):
    """Print configuration summary at test session start."""
    print("\n" + "="*60)
    print_config_summary()
    print("="*60)

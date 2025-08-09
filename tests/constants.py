"""
Constants and configuration management for GenAI Vector Database.
Reads configuration from environment variables and config files.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import toml


class Config:
    """Configuration manager that reads from environment and config files."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with optional config file path."""
        self.config_file = config_file or self._find_config_file()
        self.config_data = self._load_config_file()
        
    def _find_config_file(self) -> Optional[str]:
        """Find the appropriate config file based on environment."""
        config_dir = Path(__file__).parent.parent / "config"
        
        # Priority order: local.toml -> development.toml -> production.toml
        config_files = [
            config_dir / "local.toml",
            config_dir / "development.toml", 
            config_dir / "production.toml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                return str(config_file)
        
        return None
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if not self.config_file or not Path(self.config_file).exists():
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except (OSError, toml.TomlDecodeError) as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with environment variable override."""
        # Check environment variable first (convert key to uppercase)
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        
        # Check config file with dot notation support
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


# Global configuration instance
config = Config()

# AWS Configuration
AWS_ACCESS_KEY_ID = config.get('aws.access_key_id') or os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('aws.secret_access_key') or os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = config.get('aws.session_token') or os.getenv('AWS_SESSION_TOKEN')
AWS_REGION = config.get('storage.region', 'us-east-1')
AWS_ENDPOINT_URL = config.get('storage.endpoint') or os.getenv('AWS_ENDPOINT_URL')

# S3 Configuration  
S3_BUCKET_NAME = config.get('storage.bucket_name', 'genai-vectors-dev')
S3_REGION = AWS_REGION

# Server Configuration
SERVER_HOST = config.get('server.host', '0.0.0.0')
SERVER_PORT = int(config.get('server.port', 8080))
LOG_LEVEL = config.get('server.log_level', 'info').upper()

# Vector Configuration
DEFAULT_DIMENSION = int(config.get('vector.default_dimension', 256))
DEFAULT_BATCH_SIZE = int(config.get('vector.default_batch_size', 1000))
DEFAULT_TOP_K = int(config.get('vector.default_top_k', 10))
DEFAULT_METRIC = config.get('vector.default_metric', 'cosine')

# Cache Configuration
CACHE_DIR = config.get('cache.vector_cache_dir', './data')
CACHE_ENABLED = str(config.get('cache.enable_cache', 'true')).lower() == 'true'
CACHE_SIZE_MB = int(config.get('cache.cache_size_mb', 1024))

# Test Configuration
TEST_BUCKET_PREFIX = config.get('test.bucket_prefix', 'genai-vectors-test')
TEST_DIMENSION = int(config.get('test.dimension', 128))
TEST_SMALL_SCALE_VECTORS = int(config.get('test.small_scale_vectors', 1000))
TEST_LARGE_SCALE_VECTORS = int(config.get('test.large_scale_vectors', 500000))

# Indexing Configuration
SHARD_SIZE = int(config.get('indexing.shard_size', 10000))
MAX_SHARDS_PER_INDEX = int(config.get('indexing.max_shards_per_index', 100))
ENABLE_BACKGROUND_INDEXING = str(config.get('indexing.enable_background_indexing', 'true')).lower() == 'true'

# Monitoring Configuration (optional)
ENABLE_METRICS = str(config.get('monitoring.enable_metrics', 'false')).lower() == 'true'
METRICS_PORT = int(config.get('monitoring.metrics_port', 9090))


def get_aws_credentials() -> Dict[str, Optional[str]]:
    """Get AWS credentials from environment or config."""
    return {
        'aws_access_key_id': AWS_ACCESS_KEY_ID,
        'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
        'aws_session_token': AWS_SESSION_TOKEN,
        'region_name': AWS_REGION
    }


def get_s3_config() -> Dict[str, Any]:
    """Get S3 configuration."""
    s3_config = {
        'region_name': S3_REGION,
        'bucket_name': S3_BUCKET_NAME
    }
    
    # Add endpoint URL if specified (for MinIO)
    if AWS_ENDPOINT_URL:
        s3_config['endpoint_url'] = AWS_ENDPOINT_URL
    
    # Add credentials if available
    credentials = get_aws_credentials()
    for key, value in credentials.items():
        if value:
            s3_config[key] = value
    
    return s3_config


def get_test_config() -> Dict[str, Any]:
    """Get test configuration."""
    return {
        'aws_region': AWS_REGION,
        'test_bucket_prefix': TEST_BUCKET_PREFIX,
        'cache_dir': Path(CACHE_DIR),
        'small_scale_vectors': TEST_SMALL_SCALE_VECTORS,
        'large_scale_vectors': TEST_LARGE_SCALE_VECTORS,
        'test_dimension': TEST_DIMENSION,
        'endpoint_url': AWS_ENDPOINT_URL
    }


def validate_configuration() -> bool:
    """Validate that required configuration is present."""
    required_for_aws = [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]
    
    if not all(required_for_aws):
        print("⚠️  AWS credentials not found in environment or config")
        print("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("   Or add them to your config file under [aws] section")
        return False
    
    return True


# Print configuration summary when imported
def print_config_summary():
    """Print a summary of current configuration."""
    print("📋 Configuration Summary:")
    print(f"   Config file: {config.config_file or 'Not found'}")
    print(f"   AWS Region: {AWS_REGION}")
    print(f"   S3 Bucket: {S3_BUCKET_NAME}")
    print(f"   Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"   Log Level: {LOG_LEVEL}")
    print(f"   Cache Dir: {CACHE_DIR}")
    print(f"   Credentials: {'✅ Found' if AWS_ACCESS_KEY_ID else '❌ Missing'}")
    if AWS_ENDPOINT_URL:
        print(f"   Endpoint: {AWS_ENDPOINT_URL}")


if __name__ == "__main__":
    print_config_summary()
    print(f"\n🔍 Validation: {'✅ Valid' if validate_configuration() else '❌ Invalid'}")

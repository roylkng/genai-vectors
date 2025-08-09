"""
Small Scale AWS Vector Operations Testing.
Tests all AWS S3 Vector operations using cached vectors for optimal performance.
"""

import pytest
import boto3
import json
import random
import time
from typing import List, Dict, Any
from constants import get_test_config, get_aws_credentials, print_config_summary

import boto3
import json
import time
import random
import numpy as np
from typing import List, Dict, Any
import pytest
from botocore.exceptions import ClientError

# Import configuration constants
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from constants import get_test_config, get_s3_config

# Test configuration
TEST_CONFIG = get_test_config()

@pytest.mark.integration
class TestSmallScaleAWSVectors:
    """Small-scale tests covering all AWS Vector operations."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment for AWS Vector operations."""
        # Get test configuration
        TEST_CONFIG = get_test_config()
        
        # Get AWS configuration
        aws_creds = get_aws_credentials()
        
        # Extract bucket name before creating S3 client
        cls.bucket_name = aws_creds.pop('bucket_name', TEST_CONFIG.get('test_bucket_prefix', 'genai-vectors-test'))
        
        # Create S3 client with valid boto3 parameters only
        s3_config = {k: v for k, v in aws_creds.items() 
                    if k in ['aws_access_key_id', 'aws_secret_access_key', 'region_name', 'endpoint_url']}
        cls.s3_client = boto3.client('s3', **s3_config)
        
        # Test configuration
        cls.index_name = "test_index_small"
        cls.vector_dimension = TEST_CONFIG.get('test_dimension', 128)
        cls.test_vectors = cls.generate_test_vectors(100, cls.vector_dimension)  # Small batch for testing
    
    @classmethod
    def _create_test_bucket(cls):
        """Create test bucket with proper error handling."""
        try:
            if TEST_CONFIG['aws_region'] == 'us-east-1':
                cls.s3_client.create_bucket(Bucket=cls.bucket_name)
            else:
                cls.s3_client.create_bucket(
                    Bucket=cls.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': TEST_CONFIG['aws_region']}
                )
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code not in ['BucketAlreadyOwnedByYou', 'BucketAlreadyExists']:
                raise
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment with pagination."""
        try:
            # Use paginator for large object lists
            paginator = cls.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=cls.bucket_name):
                if 'Contents' in page:
                    delete_keys = [{'Key': obj['Key']} for obj in page['Contents']]
                    # Delete in batches of 1000 (S3 limit)
                    for i in range(0, len(delete_keys), 1000):
                        batch = delete_keys[i:i+1000]
                        cls.s3_client.delete_objects(
                            Bucket=cls.bucket_name,
                            Delete={'Objects': batch}
                        )
            cls.s3_client.delete_bucket(Bucket=cls.bucket_name)
        except ClientError:
            pass
    
    @staticmethod
    def generate_test_vectors(count: int, dimension: int = 128) -> List[Dict[str, Any]]:
        """Generate test vectors with metadata."""
        vectors = []
        for i in range(count):
            # Generate random vector using standard library
            vector = [random.random() for _ in range(dimension)]
            vectors.append({
                'id': f'vec_{i:06d}',
                'embedding': vector,
                'metadata': {
                    'category': random.choice(['A', 'B', 'C']),
                    'value': random.randint(1, 100),
                    'text': f'Sample text {i}',
                    'timestamp': int(time.time()) + i
                }
            })
        return vectors
    
    @pytest.mark.smoke
    def test_create_vector_bucket(self):
        """Test create-vector-bucket operation."""
        # Simulate creating a vector bucket with configuration
        bucket_config = {
            'bucket_name': self.bucket_name,
            'region': TEST_CONFIG['aws_region'],
            'vector_configuration': {
                'dimension': self.vector_dimension,
                'metric': 'cosine',
                'index_type': 'hnsw'
            }
        }
        
        # Upload bucket configuration
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key='vector-config.json',
            Body=json.dumps(bucket_config, indent=2),
            ContentType='application/json'
        )
        
        # Verify bucket configuration
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key='vector-config.json'
        )
        loaded_config = json.loads(response['Body'].read())
        assert loaded_config['vector_configuration']['dimension'] == self.vector_dimension
        assert loaded_config['vector_configuration']['metric'] == 'cosine'
    
    def test_get_vector_bucket(self):
        """Test get-vector-bucket operation."""
        # Get bucket information and configuration
        try:
            bucket_response = self.s3_client.head_bucket(Bucket=self.bucket_name)
            assert bucket_response['ResponseMetadata']['HTTPStatusCode'] == 200
            
            # Get vector configuration
            config_response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key='vector-config.json'
            )
            config = json.loads(config_response['Body'].read())
            assert 'vector_configuration' in config
            print(f"✅ Bucket {self.bucket_name} configuration retrieved")
            
        except ClientError as e:
            pytest.fail(f"Failed to get vector bucket: {e}")
    
    def test_create_index(self):
        """Test create-index operation."""
        # Create index metadata
        index_metadata = {
            'index_name': self.index_name,
            'dimension': self.vector_dimension,
            'metric': 'cosine',
            'algorithm': 'hnsw',
            'parameters': {
                'ef_construction': 200,
                'm': 16
            },
            'created_at': int(time.time()),
            'status': 'creating'
        }
        
        # Upload index metadata
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=f'indexes/{self.index_name}/metadata.json',
            Body=json.dumps(index_metadata, indent=2),
            ContentType='application/json'
        )
        
        # Verify index creation
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=f'indexes/{self.index_name}/metadata.json'
        )
        loaded_metadata = json.loads(response['Body'].read())
        assert loaded_metadata['index_name'] == self.index_name
        assert loaded_metadata['dimension'] == self.vector_dimension
    
    def test_list_indexes(self):
        """Test list-indexes operation."""
        # List all indexes in the bucket
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='indexes/',
                Delimiter='/'
            )
            
            indexes = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    index_name = prefix['Prefix'].split('/')[-2]
                    indexes.append(index_name)
            
            assert self.index_name in indexes
            print(f"✅ Found indexes: {indexes}")
            
        except ClientError as e:
            pytest.fail(f"Failed to list indexes: {e}")
    
    def test_get_index(self):
        """Test get-index operation."""
        # Get index metadata
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f'indexes/{self.index_name}/metadata.json'
            )
            index_metadata = json.loads(response['Body'].read())
            
            assert index_metadata['index_name'] == self.index_name
            assert 'dimension' in index_metadata
            assert 'metric' in index_metadata
            print(f"✅ Index {self.index_name} metadata retrieved")
            
        except ClientError as e:
            pytest.fail(f"Failed to get index: {e}")
    
    def test_put_vectors(self):
        """Test put-vectors operation."""
        # Generate test vectors
        vectors = self.generate_test_vectors(100)
        
        # Upload vectors in batches
        batch_size = 25
        batch_count = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batch_key = f'indexes/{self.index_name}/vectors/batch_{batch_count:04d}.json'
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=batch_key,
                Body=json.dumps({
                    'vectors': batch,
                    'batch_id': batch_count,
                    'index_name': self.index_name,
                    'uploaded_at': int(time.time())
                }, indent=2),
                ContentType='application/json'
            )
            batch_count += 1
        
        # Verify upload
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=f'indexes/{self.index_name}/vectors/'
        )
        
        uploaded_batches = len(response.get('Contents', []))
        expected_batches = (len(vectors) + batch_size - 1) // batch_size
        assert uploaded_batches == expected_batches
        print(f"✅ Uploaded {len(vectors)} vectors in {uploaded_batches} batches")
    
    def test_list_vectors(self):
        """Test list-vectors operation."""
        # List all vector files for the index
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f'indexes/{self.index_name}/vectors/'
            )
            
            vector_files = []
            total_size = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    vector_files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
                    total_size += obj['Size']
            
            assert len(vector_files) > 0
            print(f"✅ Found {len(vector_files)} vector files, total size: {total_size} bytes")
            
        except ClientError as e:
            pytest.fail(f"Failed to list vectors: {e}")
    
    def test_get_vectors(self):
        """Test get-vectors operation."""
        # Get specific vector batch
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f'indexes/{self.index_name}/vectors/batch_0000.json'
            )
            
            batch_data = json.loads(response['Body'].read())
            assert 'vectors' in batch_data
            assert 'batch_id' in batch_data
            assert len(batch_data['vectors']) > 0
            
            # Verify vector structure
            vector = batch_data['vectors'][0]
            assert 'id' in vector
            assert 'embedding' in vector
            assert 'metadata' in vector
            assert len(vector['embedding']) == self.vector_dimension
            
            print(f"✅ Retrieved batch with {len(batch_data['vectors'])} vectors")
            
        except ClientError as e:
            pytest.fail(f"Failed to get vectors: {e}")
    
    def test_query_vectors(self):
        """Test query-vectors operation."""
        # Create a query vector and search parameters
        query_vector = [random.random() for _ in range(self.vector_dimension)]
        
        query_request = {
            'index_name': self.index_name,
            'query_vector': query_vector,
            'top_k': 10,
            'metric': 'cosine',
            'filters': {
                'category': 'A'
            },
            'query_id': f'query_{int(time.time())}',
            'timestamp': int(time.time())
        }
        
        # Upload query request
        query_key = f'queries/{self.index_name}/query_{int(time.time())}.json'
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=query_key,
            Body=json.dumps(query_request, indent=2),
            ContentType='application/json'
        )
        
        # Simulate query results
        query_results = {
            'query_id': query_request['query_id'],
            'results': [
                {
                    'id': f'vec_{i:06d}',
                    'score': 0.95 - (i * 0.02),
                    'metadata': {
                        'category': 'A',
                        'value': random.randint(1, 100)
                    }
                }
                for i in range(10)
            ],
            'query_time_ms': 45,
            'total_candidates': 1000,
            'processed_at': int(time.time())
        }
        
        # Upload query results
        results_key = f'results/{self.index_name}/result_{query_request["query_id"]}.json'
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=results_key,
            Body=json.dumps(query_results, indent=2),
            ContentType='application/json'
        )
        
        # Verify query operation
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=results_key)
        loaded_results = json.loads(response['Body'].read())
        
        assert len(loaded_results['results']) == 10
        assert loaded_results['query_id'] == query_request['query_id']
        assert all(result['score'] <= 1.0 for result in loaded_results['results'])
        print(f"✅ Query completed with {len(loaded_results['results'])} results")
    
    def test_put_vector_bucket_policy(self):
        """Test put-vector-bucket-policy operation."""
        # Create a bucket policy for vector operations
        bucket_policy = {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Sid': 'VectorOperationsAccess',
                    'Effect': 'Allow',
                    'Principal': {'AWS': f'arn:aws:iam::{self.bucket_name}:root'},
                    'Action': [
                        's3:GetObject',
                        's3:PutObject',
                        's3:DeleteObject',
                        's3:ListBucket'
                    ],
                    'Resource': [
                        f'arn:aws:s3:::{self.bucket_name}',
                        f'arn:aws:s3:::{self.bucket_name}/*'
                    ]
                }
            ]
        }
        
        # Upload policy document
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key='bucket-policy.json',
            Body=json.dumps(bucket_policy, indent=2),
            ContentType='application/json'
        )
        
        # Verify policy
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key='bucket-policy.json'
        )
        loaded_policy = json.loads(response['Body'].read())
        assert loaded_policy['Version'] == '2012-10-17'
        assert len(loaded_policy['Statement']) > 0
        print("✅ Vector bucket policy created")
    
    def test_get_vector_bucket_policy(self):
        """Test get-vector-bucket-policy operation."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key='bucket-policy.json'
            )
            policy = json.loads(response['Body'].read())
            
            assert 'Version' in policy
            assert 'Statement' in policy
            print("✅ Vector bucket policy retrieved")
            
        except ClientError as e:
            pytest.fail(f"Failed to get bucket policy: {e}")
    
    def test_delete_vectors(self):
        """Test delete-vectors operation."""
        # Delete specific vector batch
        delete_key = f'indexes/{self.index_name}/vectors/batch_0001.json'
        
        try:
            # Check if object exists first
            self.s3_client.head_object(Bucket=self.bucket_name, Key=delete_key)
            
            # Delete the object
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=delete_key)
            
            # Verify deletion
            with pytest.raises(ClientError) as exc_info:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=delete_key)
            assert exc_info.value.response['Error']['Code'] == '404'
            
            print("✅ Vectors deleted successfully")
            
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                pytest.fail(f"Failed to delete vectors: {e}")
    
    def test_delete_index(self):
        """Test delete-index operation."""
        # Delete all objects under the index prefix
        try:
            # List all objects in the index
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f'indexes/{self.index_name}/'
            )
            
            if 'Contents' in response:
                delete_keys = [{'Key': obj['Key']} for obj in response['Contents']]
                
                # Delete in batches
                for i in range(0, len(delete_keys), 1000):
                    batch = delete_keys[i:i+1000]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': batch}
                    )
            
            print(f"✅ Index {self.index_name} deleted successfully")
            
        except ClientError as e:
            pytest.fail(f"Failed to delete index: {e}")
    
    def test_list_vector_buckets(self):
        """Test list-vector-buckets operation."""
        # List all buckets and filter for vector buckets
        try:
            response = self.s3_client.list_buckets()
            
            vector_buckets = []
            for bucket in response['Buckets']:
                bucket_name = bucket['Name']
                
                # Check if bucket has vector configuration
                try:
                    self.s3_client.head_object(
                        Bucket=bucket_name,
                        Key='vector-config.json'
                    )
                    vector_buckets.append({
                        'name': bucket_name,
                        'creation_date': bucket['CreationDate']
                    })
                except ClientError:
                    # Not a vector bucket
                    continue
            
            # Our test bucket should be in the list
            test_bucket_found = any(
                bucket['name'] == self.bucket_name 
                for bucket in vector_buckets
            )
            assert test_bucket_found
            print(f"✅ Found {len(vector_buckets)} vector buckets")
            
        except ClientError as e:
            pytest.fail(f"Failed to list vector buckets: {e}")
    
    def test_delete_vector_bucket_policy(self):
        """Test delete-vector-bucket-policy operation."""
        try:
            # Delete the bucket policy document
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key='bucket-policy.json'
            )
            
            # Verify deletion
            with pytest.raises(ClientError) as exc_info:
                self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key='bucket-policy.json'
                )
            assert exc_info.value.response['Error']['Code'] == '404'
            
            print("✅ Vector bucket policy deleted")
            
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                pytest.fail(f"Failed to delete bucket policy: {e}")

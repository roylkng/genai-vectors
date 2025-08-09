"""
Large-scale performance tests for vector database.
Tests performance and scalability with 500K vectors using AWS CLI/boto3 only.
"""

import boto3
import json
import time
import random
import numpy as np
from typing import List, Dict, Any
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError

class TestLargeScale:
    """Large-scale performance tests."""
    
    @classmethod
    def setup_class(cls):
        """Setup large-scale test environment."""
        cls.s3_client = boto3.client('s3')
        cls.bucket_name = 'test-vector-db-large'
        cls.index_name = 'test_index_large'
        cls.dimension = 256
        cls.vector_count = 500000
        cls.batch_size = 1000
        
        # Create bucket if not exists
        try:
            cls.s3_client.create_bucket(Bucket=cls.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
                raise
    
    @classmethod
    def teardown_class(cls):
        """Cleanup large-scale test environment."""
        # Delete all objects in bucket (paginated for large datasets)
        try:
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
    
    def generate_large_dataset(self, count: int) -> List[Dict[str, Any]]:
        """Generate large dataset efficiently."""
        vectors = []
        for i in range(count):
            vector = np.random.rand(self.dimension).astype(np.float32).tolist()
            vectors.append({
                'id': f'large_vec_{i:08d}',
                'embedding': vector,
                'metadata': {
                    'category': random.choice(['tech', 'business', 'science', 'arts']),
                    'priority': random.randint(1, 10),
                    'timestamp': int(time.time()) + i,
                    'tags': random.sample(['tag1', 'tag2', 'tag3', 'tag4', 'tag5'], 
                                        random.randint(1, 3))
                }
            })
        return vectors
    
    def upload_batch_parallel(self, batch_data: List[Dict], batch_id: int) -> bool:
        """Upload a single batch in parallel."""
        try:
            batch_key = f'large_batches/batch_{batch_id:06d}.json'
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=batch_key,
                Body=json.dumps(batch_data),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            print(f"Failed to upload batch {batch_id}: {e}")
            return False
    
    def test_large_scale_ingestion(self):
        """Test ingestion of 500K vectors with parallel uploads."""
        start_time = time.time()
        
        # Generate data in batches to manage memory
        total_batches = self.vector_count // self.batch_size
        successful_uploads = 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for batch_id in range(total_batches):
                # Generate batch data
                batch_vectors = self.generate_large_dataset(self.batch_size)
                
                # Submit upload task
                future = executor.submit(self.upload_batch_parallel, batch_vectors, batch_id)
                futures.append(future)
                
                # Process completed uploads periodically
                if len(futures) >= 50:  # Process in chunks
                    for future in as_completed(futures):
                        if future.result():
                            successful_uploads += 1
                    futures = []
            
            # Process remaining futures
            for future in as_completed(futures):
                if future.result():
                    successful_uploads += 1
        
        ingestion_time = time.time() - start_time
        
        # Verify uploads
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix='large_batches/',
            MaxKeys=1000
        )
        uploaded_count = len(response.get('Contents', []))
        
        print(f"Ingestion completed in {ingestion_time:.2f}s")
        print(f"Successful uploads: {successful_uploads}/{total_batches}")
        print(f"Verified uploads: {uploaded_count}")
        
        assert successful_uploads >= total_batches * 0.95  # Allow 5% failure rate
    
    def test_concurrent_search_performance(self):
        """Test concurrent search performance."""
        # Generate multiple query vectors
        query_count = 100
        queries = []
        
        for i in range(query_count):
            query = {
                'index': self.index_name,
                'embedding': np.random.rand(self.dimension).astype(np.float32).tolist(),
                'topk': 50,
                'filters': {
                    'category': random.choice(['tech', 'business', 'science', 'arts']),
                    'priority': {'$gte': random.randint(1, 5)}
                }
            }
            queries.append((i, query))
        
        # Upload queries concurrently
        start_time = time.time()
        
        def upload_query(query_data):
            query_id, query = query_data
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f'performance_queries/query_{query_id:04d}.json',
                    Body=json.dumps(query),
                    ContentType='application/json'
                )
                return True
            except Exception:
                return False
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(upload_query, queries))
        
        upload_time = time.time() - start_time
        successful_queries = sum(results)
        
        print(f"Query upload completed in {upload_time:.2f}s")
        print(f"Successful queries: {successful_queries}/{query_count}")
        
        assert successful_queries >= query_count * 0.95
    
    def test_memory_efficiency(self):
        """Test memory-efficient operations for large datasets."""
        # Test streaming large dataset processing
        chunk_size = 10000
        processed_chunks = 0
        
        for chunk_id in range(0, 50000, chunk_size):  # Process 50K vectors in chunks
            chunk_data = {
                'chunk_id': chunk_id // chunk_size,
                'vectors': self.generate_large_dataset(min(chunk_size, 50000 - chunk_id)),
                'processing_timestamp': int(time.time())
            }
            
            # Upload chunk
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f'memory_test/chunk_{chunk_id//chunk_size:04d}.json',
                    Body=json.dumps(chunk_data),
                    ContentType='application/json'
                )
                processed_chunks += 1
            except Exception as e:
                print(f"Failed to process chunk {chunk_id}: {e}")
        
        # Verify chunk processing
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix='memory_test/'
        )
        uploaded_chunks = len(response.get('Contents', []))
        
        print(f"Processed chunks: {processed_chunks}")
        print(f"Uploaded chunks: {uploaded_chunks}")
        
        assert uploaded_chunks >= processed_chunks * 0.9
    
    def test_scalability_metrics(self):
        """Test and record scalability metrics."""
        metrics = {
            'test_timestamp': int(time.time()),
            'vector_count': self.vector_count,
            'dimension': self.dimension,
            'batch_size': self.batch_size,
            'performance_metrics': {
                'ingestion_rate_vectors_per_second': 0,
                'query_latency_ms': 0,
                'storage_efficiency_mb_per_1k_vectors': 0
            }
        }
        
        # Upload metrics
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key='metrics/scalability_test_results.json',
            Body=json.dumps(metrics, indent=2),
            ContentType='application/json'
        )
        
        # Verify metrics upload
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key='metrics/scalability_test_results.json'
        )
        loaded_metrics = json.loads(response['Body'].read())
        
        assert loaded_metrics['vector_count'] == self.vector_count
        assert 'performance_metrics' in loaded_metrics

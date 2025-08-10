#!/usr/bin/env python3
"""
Performance monitoring and real-time metrics collection for the vector database.
This script demonstrates how to collect and analyze performance data from the Rust application.
"""

import asyncio
import aiohttp
import time
import json
import random
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


class PerformanceMonitor:
    """Real-time performance monitoring for vector database operations."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        self.metrics_history = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_test_vector(self, dim: int = 512) -> List[float]:
        """Generate a random test vector."""
        return [random.random() for _ in range(dim)]
    
    async def create_index(self, index_name: str, dim: int = 512) -> Dict[str, Any]:
        """Create a test index."""
        payload = {
            "name": index_name,
            "dim": dim,
            "metric": "cosine"
        }
        
        async with self.session.post(f"{self.base_url}/indexes", json=payload) as response:
            return await response.json()
    
    async def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert vectors into the index."""
        payload = {
            "index": index_name,
            "vectors": vectors
        }
        
        start_time = time.time()
        async with self.session.post(f"{self.base_url}/vectors", json=payload) as response:
            result = await response.json()
            result['client_latency_ms'] = (time.time() - start_time) * 1000
            return result
    
    async def query_vectors(self, index_name: str, vector: List[float], k: int = 10) -> Dict[str, Any]:
        """Query vectors from the index."""
        payload = {
            "index": index_name,
            "vector": vector,
            "k": k,
            "topk": k
        }
        
        start_time = time.time()
        async with self.session.post(f"{self.base_url}/query", json=payload) as response:
            result = await response.json()
            result['client_latency_ms'] = (time.time() - start_time) * 1000
            return result
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Fetch current metrics from the application."""
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def benchmark_insertion_performance(self, index_name: str, batch_sizes: List[int], 
                                            vector_counts: List[int], dim: int = 512):
        """Benchmark vector insertion performance across different scales."""
        print(f"\n🔄 Benchmarking insertion performance for index '{index_name}'")
        print(f"   Vector dimension: {dim}, Batch sizes: {batch_sizes}, Vector counts: {vector_counts}")
        
        results = []
        
        for vector_count in vector_counts:
            for batch_size in batch_sizes:
                print(f"\n   Testing {vector_count} vectors with batch size {batch_size}")
                
                # Generate test vectors
                test_vectors = []
                for i in range(vector_count):
                    test_vectors.append({
                        "id": f"test_vec_{i}_{int(time.time())}",
                        "vector": self.generate_test_vector(dim),
                        "metadata": {"batch": f"batch_{i // batch_size}", "index": i}
                    })
                
                # Insert in batches
                total_time = 0
                batches_processed = 0
                
                for i in range(0, len(test_vectors), batch_size):
                    batch = test_vectors[i:i + batch_size]
                    result = await self.insert_vectors(index_name, batch)
                    
                    if 'client_latency_ms' in result:
                        total_time += result['client_latency_ms']
                        batches_processed += 1
                    
                    # Small delay to prevent overwhelming the server
                    await asyncio.sleep(0.1)
                
                avg_batch_time = total_time / batches_processed if batches_processed > 0 else 0
                vectors_per_second = (vector_count / (total_time / 1000)) if total_time > 0 else 0
                
                result_record = {
                    "timestamp": datetime.now().isoformat(),
                    "vector_count": vector_count,
                    "batch_size": batch_size,
                    "total_time_ms": total_time,
                    "avg_batch_time_ms": avg_batch_time,
                    "vectors_per_second": vectors_per_second,
                    "batches_processed": batches_processed
                }
                
                results.append(result_record)
                print(f"     ✅ {vector_count} vectors, batch {batch_size}: {vectors_per_second:.2f} vectors/sec")
        
        return results
    
    async def benchmark_query_performance(self, index_name: str, query_counts: List[int], 
                                        k_values: List[int], dim: int = 512):
        """Benchmark query performance across different parameters."""
        print(f"\n🔍 Benchmarking query performance for index '{index_name}'")
        print(f"   Query counts: {query_counts}, K values: {k_values}")
        
        results = []
        
        for query_count in query_counts:
            for k in k_values:
                print(f"\n   Testing {query_count} queries with k={k}")
                
                query_times = []
                successful_queries = 0
                
                for i in range(query_count):
                    query_vector = self.generate_test_vector(dim)
                    result = await self.query_vectors(index_name, query_vector, k)
                    
                    if 'client_latency_ms' in result:
                        query_times.append(result['client_latency_ms'])
                        successful_queries += 1
                    
                    # Small delay between queries
                    await asyncio.sleep(0.05)
                
                if query_times:
                    avg_query_time = sum(query_times) / len(query_times)
                    p95_query_time = np.percentile(query_times, 95)
                    p99_query_time = np.percentile(query_times, 99)
                    queries_per_second = successful_queries / (sum(query_times) / 1000) if sum(query_times) > 0 else 0
                else:
                    avg_query_time = p95_query_time = p99_query_time = queries_per_second = 0
                
                result_record = {
                    "timestamp": datetime.now().isoformat(),
                    "query_count": query_count,
                    "k": k,
                    "avg_query_time_ms": avg_query_time,
                    "p95_query_time_ms": p95_query_time,
                    "p99_query_time_ms": p99_query_time,
                    "queries_per_second": queries_per_second,
                    "successful_queries": successful_queries
                }
                
                results.append(result_record)
                print(f"     ✅ {query_count} queries, k={k}: {queries_per_second:.2f} queries/sec, "
                      f"avg {avg_query_time:.2f}ms, p95 {p95_query_time:.2f}ms")
        
        return results
    
    async def continuous_monitoring(self, duration_minutes: int = 10, interval_seconds: int = 30):
        """Continuously monitor performance metrics."""
        print(f"\n📊 Starting continuous monitoring for {duration_minutes} minutes")
        print(f"   Collecting metrics every {interval_seconds} seconds")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            metrics = await self.get_metrics()
            metrics['timestamp'] = datetime.now().isoformat()
            self.metrics_history.append(metrics)
            
            if 'error' not in metrics:
                print(f"   📈 Metrics collected at {metrics['timestamp']}")
                # Print key metrics if available
                if 'query_total_time_ms' in metrics:
                    print(f"      Query time: {metrics.get('query_total_time_ms', 'N/A')}ms")
                if 'indexer_vectors_loaded' in metrics:
                    print(f"      Vectors loaded: {metrics.get('indexer_vectors_loaded', 'N/A')}")
            else:
                print(f"   ❌ Error collecting metrics: {metrics['error']}")
            
            await asyncio.sleep(interval_seconds)
        
        print(f"\n✅ Monitoring completed. Collected {len(self.metrics_history)} metric snapshots")
        return self.metrics_history
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save performance results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to {filename}")
    
    def generate_performance_report(self, results: Dict[str, Any]):
        """Generate a comprehensive performance report."""
        print("\n" + "="*80)
        print("🎯 VECTOR DATABASE PERFORMANCE REPORT")
        print("="*80)
        
        if 'insertion_results' in results:
            print("\n📥 INSERTION PERFORMANCE:")
            insertion_results = results['insertion_results']
            for result in insertion_results[-5:]:  # Show last 5 results
                print(f"   • {result['vector_count']} vectors, batch {result['batch_size']}: "
                      f"{result['vectors_per_second']:.2f} vectors/sec")
        
        if 'query_results' in results:
            print("\n🔍 QUERY PERFORMANCE:")
            query_results = results['query_results']
            for result in query_results[-5:]:  # Show last 5 results
                print(f"   • {result['query_count']} queries, k={result['k']}: "
                      f"{result['queries_per_second']:.2f} queries/sec "
                      f"(avg: {result['avg_query_time_ms']:.2f}ms)")
        
        if 'metrics_history' in results and results['metrics_history']:
            print("\n📊 SYSTEM METRICS SUMMARY:")
            metrics = results['metrics_history']
            if len(metrics) > 0:
                latest = metrics[-1]
                print(f"   • Latest metrics collected: {latest.get('timestamp', 'N/A')}")
                print(f"   • Total metric snapshots: {len(metrics)}")
        
        print("\n" + "="*80)


async def main():
    parser = argparse.ArgumentParser(description="Monitor vector database performance")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the vector database")
    parser.add_argument("--index", default="performance_test", help="Index name for testing")
    parser.add_argument("--dimension", type=int, default=512, help="Vector dimension")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--monitor-only", action="store_true", help="Only monitor existing system")
    
    args = parser.parse_args()
    
    async with PerformanceMonitor(args.url) as monitor:
        print(f"🚀 Starting performance monitoring for {args.url}")
        
        all_results = {}
        
        if not args.monitor_only:
            # Create test index
            print(f"\n📝 Creating test index '{args.index}'")
            try:
                await monitor.create_index(args.index, args.dimension)
                print(f"✅ Index '{args.index}' created successfully")
            except Exception as e:
                print(f"ℹ️  Index creation result: {e} (may already exist)")
            
            # Benchmark insertion performance
            if args.quick:
                batch_sizes = [10, 50]
                vector_counts = [100, 500]
                query_counts = [10, 50]
                k_values = [5, 10]
            else:
                batch_sizes = [10, 50, 100, 500]
                vector_counts = [100, 500, 1000, 5000]
                query_counts = [10, 50, 100, 500]
                k_values = [5, 10, 20, 50]
            
            insertion_results = await monitor.benchmark_insertion_performance(
                args.index, batch_sizes, vector_counts, args.dimension
            )
            all_results['insertion_results'] = insertion_results
            
            # Benchmark query performance
            query_results = await monitor.benchmark_query_performance(
                args.index, query_counts, k_values, args.dimension
            )
            all_results['query_results'] = query_results
        
        # Continuous monitoring
        monitoring_duration = 2 if args.quick else 5
        metrics_history = await monitor.continuous_monitoring(
            duration_minutes=monitoring_duration, interval_seconds=15
        )
        all_results['metrics_history'] = metrics_history
        
        # Generate report
        monitor.generate_performance_report(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_results_{timestamp}.json"
        monitor.save_results(all_results, filename)
        
        print(f"\n🎉 Performance monitoring completed!")
        print(f"📊 Run this script with --monitor-only to continue monitoring an existing system")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Comprehensive Performance Testing and Parameter Tuning for GenAI Vector Database
This script performs gradual load testing and parameter optimization to achieve production readiness.
"""

import asyncio
import aiohttp
import json
import time
import statistics
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import argparse

@dataclass
class TestConfig:
    """Configuration for performance tests"""
    base_url: str = "http://localhost:8080"
    vector_dim: int = 1536  # OpenAI ada-002 dimension
    num_vectors_stages: List[int] = None  # Will be set in __post_init__
    query_batch_sizes: List[int] = None
    concurrent_queries: List[int] = None
    nlist_values: List[int] = None
    nprobe_values: List[int] = None
    pq_m_values: List[int] = None
    pq_nbits_values: List[int] = None
    
    def __post_init__(self):
        if self.num_vectors_stages is None:
            # Gradual increase: 1K -> 10K -> 100K -> 1M -> 10M
            self.num_vectors_stages = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        if self.query_batch_sizes is None:
            self.query_batch_sizes = [1, 5, 10, 25, 50, 100]
        if self.concurrent_queries is None:
            self.concurrent_queries = [1, 5, 10, 20, 50, 100]
        if self.nlist_values is None:
            # Different clustering strategies
            self.nlist_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if self.nprobe_values is None:
            # Different search quality vs speed trade-offs
            self.nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        if self.pq_m_values is None:
            # Product quantization subspaces
            self.pq_m_values = [4, 8, 16, 32]
        if self.pq_nbits_values is None:
            # Bits per subspace
            self.pq_nbits_values = [4, 6, 8, 10]

@dataclass
class PerformanceMetrics:
    """Performance metrics for a test run"""
    indexing_time: float
    index_size_mb: float
    query_latency_p50: float
    query_latency_p95: float
    query_latency_p99: float
    query_throughput_qps: float
    memory_usage_mb: float
    recall_at_k: float
    error_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'indexing_time': self.indexing_time,
            'index_size_mb': self.index_size_mb,
            'query_latency_p50': self.query_latency_p50,
            'query_latency_p95': self.query_latency_p95,
            'query_latency_p99': self.query_latency_p99,
            'query_throughput_qps': self.query_throughput_qps,
            'memory_usage_mb': self.memory_usage_mb,
            'recall_at_k': self.recall_at_k,
            'error_rate': self.error_rate
        }

class VectorGenerator:
    """Generate realistic test vectors with known similarities"""
    
    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_clustered_vectors(self, num_vectors: int, num_clusters: int = 10) -> List[List[float]]:
        """Generate vectors with known cluster structure for recall testing"""
        vectors = []
        cluster_centers = [self._generate_unit_vector() for _ in range(num_clusters)]
        
        for i in range(num_vectors):
            cluster_idx = i % num_clusters
            center = cluster_centers[cluster_idx]
            
            # Add noise to cluster center
            noise = np.random.normal(0, 0.1, self.dim)
            vector = np.array(center) + noise
            
            # Normalize to unit vector
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector.tolist())
        
        return vectors
    
    def generate_random_vectors(self, num_vectors: int) -> List[List[float]]:
        """Generate random unit vectors"""
        return [self._generate_unit_vector() for _ in range(num_vectors)]
    
    def _generate_unit_vector(self) -> List[float]:
        """Generate a random unit vector"""
        vector = np.random.normal(0, 1, self.dim)
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def generate_query_with_ground_truth(self, vectors: List[List[float]], 
                                       query_vector: List[float], k: int = 10) -> Tuple[List[str], List[float]]:
        """Generate ground truth for recall calculation"""
        similarities = []
        for i, vec in enumerate(vectors):
            similarity = np.dot(query_vector, vec)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        ground_truth_ids = [str(i) for i, _ in similarities[:k]]
        ground_truth_scores = [score for _, score in similarities[:k]]
        
        return ground_truth_ids, ground_truth_scores

class PerformanceTester:
    """Main performance testing class"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.vector_gen = VectorGenerator(config.vector_dim)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def create_index(self, index_name: str, nlist: int, m: int, nbits: int) -> bool:
        """Create an index with specified parameters"""
        payload = {
            "name": index_name,
            "dim": self.config.vector_dim,
            "metric": "cosine",
            "nlist": nlist,
            "m": m,
            "nbits": nbits,
            "default_nprobe": max(1, nlist // 8)
        }
        
        try:
            async with self.session.post(f"{self.config.base_url}/indexes", json=payload) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"❌ Failed to create index: {e}")
            return False
    
    async def insert_vectors(self, index_name: str, vectors: List[List[float]]) -> Tuple[bool, float]:
        """Insert vectors and measure indexing time"""
        start_time = time.time()
        
        # Prepare vector records
        vector_records = []
        for i, vector in enumerate(vectors):
            vector_records.append({
                "id": str(i),
                "embedding": vector,
                "meta": {"cluster": i % 10, "batch": i // 1000}
            })
        
        payload = {
            "index": index_name,
            "vectors": vector_records
        }
        
        try:
            async with self.session.post(f"{self.config.base_url}/vectors", json=payload) as resp:
                success = resp.status == 200
                indexing_time = time.time() - start_time
                return success, indexing_time
        except Exception as e:
            print(f"❌ Failed to insert vectors: {e}")
            return False, 0.0
    
    async def query_vectors(self, index_name: str, query_vector: List[float], 
                          topk: int = 10, nprobe: Optional[int] = None) -> Tuple[Dict, float]:
        """Query vectors and measure latency"""
        payload = {
            "index": index_name,
            "embedding": query_vector,
            "topk": topk
        }
        
        if nprobe is not None:
            payload["nprobe"] = nprobe
        
        start_time = time.time()
        try:
            async with self.session.post(f"{self.config.base_url}/query", json=payload) as resp:
                latency = time.time() - start_time
                if resp.status == 200:
                    result = await resp.json()
                    return result, latency
                else:
                    return {"error": f"HTTP {resp.status}"}, latency
        except Exception as e:
            latency = time.time() - start_time
            return {"error": str(e)}, latency
    
    async def run_concurrent_queries(self, index_name: str, query_vectors: List[List[float]], 
                                   concurrency: int, nprobe: Optional[int] = None) -> List[Tuple[Dict, float]]:
        """Run multiple queries concurrently"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def query_with_semaphore(query_vector):
            async with semaphore:
                return await self.query_vectors(index_name, query_vector, nprobe=nprobe)
        
        tasks = [query_with_semaphore(qv) for qv in query_vectors]
        return await asyncio.gather(*tasks)
    
    def calculate_recall(self, ground_truth_ids: List[str], retrieved_ids: List[str]) -> float:
        """Calculate recall@k"""
        if not ground_truth_ids or not retrieved_ids:
            return 0.0
        
        ground_truth_set = set(ground_truth_ids)
        retrieved_set = set(retrieved_ids)
        
        intersection = ground_truth_set.intersection(retrieved_set)
        return len(intersection) / len(ground_truth_set)
    
    async def benchmark_configuration(self, num_vectors: int, nlist: int, m: int, 
                                    nbits: int, nprobe: int) -> PerformanceMetrics:
        """Benchmark a specific configuration"""
        index_name = f"perf_test_{num_vectors}_{nlist}_{m}_{nbits}_{nprobe}"
        
        print(f"🧪 Testing: {num_vectors} vectors, nlist={nlist}, m={m}, nbits={nbits}, nprobe={nprobe}")
        
        # Create index
        if not await self.create_index(index_name, nlist, m, nbits):
            raise Exception("Failed to create index")
        
        # Generate test vectors
        vectors = self.vector_gen.generate_clustered_vectors(num_vectors)
        
        # Measure indexing performance
        success, indexing_time = await self.insert_vectors(index_name, vectors)
        if not success:
            raise Exception("Failed to insert vectors")
        
        # Wait for indexing to complete (simulate background processing)
        await asyncio.sleep(2)
        
        # Generate query vectors
        num_queries = min(100, num_vectors // 10)
        query_vectors = self.vector_gen.generate_random_vectors(num_queries)
        
        # Measure query performance
        query_results = []
        query_latencies = []
        recall_scores = []
        
        start_time = time.time()
        for query_vector in query_vectors[:20]:  # Test with 20 queries
            result, latency = await self.query_vectors(index_name, query_vector, topk=10, nprobe=nprobe)
            query_latencies.append(latency)
            
            if "results" in result:
                retrieved_ids = [r["id"] for r in result["results"]]
                ground_truth_ids, _ = self.vector_gen.generate_query_with_ground_truth(
                    vectors, query_vector, k=10
                )
                recall = self.calculate_recall(ground_truth_ids, retrieved_ids)
                recall_scores.append(recall)
            else:
                recall_scores.append(0.0)
        
        total_query_time = time.time() - start_time
        
        # Calculate metrics
        if query_latencies:
            p50_latency = statistics.median(query_latencies)
            p95_latency = np.percentile(query_latencies, 95)
            p99_latency = np.percentile(query_latencies, 99)
            throughput = len(query_latencies) / total_query_time
        else:
            p50_latency = p95_latency = p99_latency = throughput = 0.0
        
        avg_recall = statistics.mean(recall_scores) if recall_scores else 0.0
        error_rate = sum(1 for r in query_results if "error" in r) / len(query_results) if query_results else 0.0
        
        # Estimate index size (mock calculation)
        index_size_mb = (num_vectors * self.config.vector_dim * nbits * m / 8) / (1024 * 1024)
        memory_usage_mb = index_size_mb * 1.2  # Include overhead
        
        return PerformanceMetrics(
            indexing_time=indexing_time,
            index_size_mb=index_size_mb,
            query_latency_p50=p50_latency * 1000,  # Convert to ms
            query_latency_p95=p95_latency * 1000,
            query_latency_p99=p99_latency * 1000,
            query_throughput_qps=throughput,
            memory_usage_mb=memory_usage_mb,
            recall_at_k=avg_recall,
            error_rate=error_rate
        )
    
    async def run_gradual_load_test(self) -> Dict[str, List[Dict]]:
        """Run gradual load testing with increasing vector counts"""
        results = {}
        
        for num_vectors in self.config.num_vectors_stages:
            print(f"\n🚀 Testing with {num_vectors:,} vectors")
            
            # Calculate optimal parameters for this scale
            optimal_nlist = max(16, min(4096, int(np.sqrt(num_vectors))))
            optimal_nprobe = max(1, optimal_nlist // 8)
            
            stage_results = []
            
            # Test different parameter combinations
            nlist_values = [optimal_nlist // 2, optimal_nlist, optimal_nlist * 2]
            nprobe_values = [optimal_nprobe // 2, optimal_nprobe, optimal_nprobe * 2]
            
            for nlist in nlist_values:
                for nprobe in nprobe_values:
                    if nprobe > nlist:
                        continue
                    
                    try:
                        metrics = await self.benchmark_configuration(
                            num_vectors, nlist, 8, 8, nprobe
                        )
                        
                        result = {
                            'num_vectors': num_vectors,
                            'nlist': nlist,
                            'nprobe': nprobe,
                            'm': 8,
                            'nbits': 8,
                            **metrics.to_dict()
                        }
                        stage_results.append(result)
                        
                        print(f"  ✅ nlist={nlist}, nprobe={nprobe}: "
                              f"{metrics.query_latency_p50:.1f}ms p50, "
                              f"{metrics.recall_at_k:.3f} recall, "
                              f"{metrics.query_throughput_qps:.1f} QPS")
                        
                    except Exception as e:
                        print(f"  ❌ nlist={nlist}, nprobe={nprobe}: {e}")
                        continue
            
            results[f"{num_vectors}_vectors"] = stage_results
            
            # Brief pause between stages
            await asyncio.sleep(1)
        
        return results
    
    async def run_parameter_sweep(self, num_vectors: int = 100000) -> Dict[str, List[Dict]]:
        """Run comprehensive parameter sweep for optimization"""
        print(f"\n🔍 Parameter sweep with {num_vectors:,} vectors")
        
        results = {
            'nlist_sweep': [],
            'nprobe_sweep': [],
            'pq_sweep': []
        }
        
        # nlist sweep
        print("\n📊 Testing nlist values...")
        for nlist in self.config.nlist_values:
            if nlist > num_vectors // 10:  # Skip if too many clusters
                continue
                
            nprobe = max(1, nlist // 8)
            try:
                metrics = await self.benchmark_configuration(num_vectors, nlist, 8, 8, nprobe)
                result = {
                    'nlist': nlist,
                    'nprobe': nprobe,
                    **metrics.to_dict()
                }
                results['nlist_sweep'].append(result)
                print(f"  nlist={nlist}: {metrics.query_latency_p50:.1f}ms, {metrics.recall_at_k:.3f} recall")
            except Exception as e:
                print(f"  ❌ nlist={nlist}: {e}")
        
        # nprobe sweep (using optimal nlist)
        optimal_nlist = max(16, min(1024, int(np.sqrt(num_vectors))))
        print(f"\n📊 Testing nprobe values (nlist={optimal_nlist})...")
        for nprobe in self.config.nprobe_values:
            if nprobe > optimal_nlist:
                continue
                
            try:
                metrics = await self.benchmark_configuration(num_vectors, optimal_nlist, 8, 8, nprobe)
                result = {
                    'nlist': optimal_nlist,
                    'nprobe': nprobe,
                    **metrics.to_dict()
                }
                results['nprobe_sweep'].append(result)
                print(f"  nprobe={nprobe}: {metrics.query_latency_p50:.1f}ms, {metrics.recall_at_k:.3f} recall")
            except Exception as e:
                print(f"  ❌ nprobe={nprobe}: {e}")
        
        # PQ parameter sweep
        print(f"\n📊 Testing PQ parameters...")
        optimal_nprobe = max(1, optimal_nlist // 8)
        for m in self.config.pq_m_values:
            for nbits in self.config.pq_nbits_values:
                if self.config.vector_dim % m != 0:  # m must divide dimension
                    continue
                    
                try:
                    metrics = await self.benchmark_configuration(num_vectors, optimal_nlist, m, nbits, optimal_nprobe)
                    result = {
                        'nlist': optimal_nlist,
                        'nprobe': optimal_nprobe,
                        'm': m,
                        'nbits': nbits,
                        **metrics.to_dict()
                    }
                    results['pq_sweep'].append(result)
                    print(f"  m={m}, nbits={nbits}: {metrics.query_latency_p50:.1f}ms, "
                          f"{metrics.index_size_mb:.1f}MB, {metrics.recall_at_k:.3f} recall")
                except Exception as e:
                    print(f"  ❌ m={m}, nbits={nbits}: {e}")
        
        return results

def save_results(results: Dict, filename: str):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to {filename}")

def analyze_results(results: Dict):
    """Analyze and provide recommendations"""
    print("\n🎯 Performance Analysis & Recommendations")
    print("=" * 60)
    
    # Find best configurations for different metrics
    all_results = []
    for stage_name, stage_results in results.items():
        if isinstance(stage_results, list):
            all_results.extend(stage_results)
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Best latency
    best_latency = min(all_results, key=lambda x: x.get('query_latency_p50', float('inf')))
    print(f"🏃 Best Latency: {best_latency['query_latency_p50']:.1f}ms")
    print(f"   Configuration: nlist={best_latency['nlist']}, nprobe={best_latency['nprobe']}, "
          f"m={best_latency['m']}, nbits={best_latency['nbits']}")
    
    # Best recall
    best_recall = max(all_results, key=lambda x: x.get('recall_at_k', 0))
    print(f"🎯 Best Recall: {best_recall['recall_at_k']:.3f}")
    print(f"   Configuration: nlist={best_recall['nlist']}, nprobe={best_recall['nprobe']}, "
          f"m={best_recall['m']}, nbits={best_recall['nbits']}")
    
    # Best throughput
    best_throughput = max(all_results, key=lambda x: x.get('query_throughput_qps', 0))
    print(f"⚡ Best Throughput: {best_throughput['query_throughput_qps']:.1f} QPS")
    print(f"   Configuration: nlist={best_throughput['nlist']}, nprobe={best_throughput['nprobe']}, "
          f"m={best_throughput['m']}, nbits={best_throughput['nbits']}")
    
    # Best memory efficiency
    best_memory = min(all_results, key=lambda x: x.get('index_size_mb', float('inf')))
    print(f"💾 Most Memory Efficient: {best_memory['index_size_mb']:.1f}MB")
    print(f"   Configuration: nlist={best_memory['nlist']}, nprobe={best_memory['nprobe']}, "
          f"m={best_memory['m']}, nbits={best_memory['nbits']}")
    
    # Balanced recommendation
    balanced_scores = []
    for result in all_results:
        # Normalize metrics and calculate balanced score
        latency_score = 1.0 / max(result.get('query_latency_p50', 1), 1)
        recall_score = result.get('recall_at_k', 0)
        throughput_score = result.get('query_throughput_qps', 0) / 100
        memory_score = 1.0 / max(result.get('index_size_mb', 1), 1)
        
        balanced_score = (latency_score + recall_score + throughput_score + memory_score) / 4
        balanced_scores.append((balanced_score, result))
    
    best_balanced = max(balanced_scores, key=lambda x: x[0])[1]
    print(f"\n🎖️  Recommended Balanced Configuration:")
    print(f"   nlist={best_balanced['nlist']}, nprobe={best_balanced['nprobe']}, "
          f"m={best_balanced['m']}, nbits={best_balanced['nbits']}")
    print(f"   Latency: {best_balanced['query_latency_p50']:.1f}ms")
    print(f"   Recall: {best_balanced['recall_at_k']:.3f}")
    print(f"   Throughput: {best_balanced['query_throughput_qps']:.1f} QPS")
    print(f"   Memory: {best_balanced['index_size_mb']:.1f}MB")

async def main():
    parser = argparse.ArgumentParser(description="Vector Database Performance Testing")
    parser.add_argument("--mode", choices=["gradual", "sweep", "both"], default="both",
                      help="Testing mode")
    parser.add_argument("--vectors", type=int, default=100000,
                      help="Number of vectors for parameter sweep")
    parser.add_argument("--output", default="performance_results.json",
                      help="Output file for results")
    
    args = parser.parse_args()
    
    config = TestConfig()
    
    print("🚀 Starting Vector Database Performance Testing")
    print(f"📊 Configuration: {config.vector_dim}D vectors, {len(config.num_vectors_stages)} stages")
    
    async with PerformanceTester(config) as tester:
        all_results = {}
        
        if args.mode in ["gradual", "both"]:
            print("\n🏃 Running gradual load testing...")
            gradual_results = await tester.run_gradual_load_test()
            all_results.update(gradual_results)
        
        if args.mode in ["sweep", "both"]:
            print(f"\n🔍 Running parameter sweep with {args.vectors:,} vectors...")
            sweep_results = await tester.run_parameter_sweep(args.vectors)
            all_results.update(sweep_results)
        
        # Save and analyze results
        save_results(all_results, args.output)
        analyze_results(all_results)
        
        print(f"\n✅ Testing complete! Results saved to {args.output}")
        print("🎯 Use the recommended configuration for optimal performance")

if __name__ == "__main__":
    asyncio.run(main())

"""
Comprehensive Benchmark Suite for AECF (Attention Entropy Curriculum Filtering)

This benchmark suite validates performance characteristics against PyTorch baselines
and ensures the AECF implementation meets production performance requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import psutil
import os
import json
import statistics
from typing import Dict, List, Tuple, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import warnings

# Import the modules under test
from proper_aecf_core import (
    CurriculumMasking, 
    MultimodalAttentionPool, 
    multimodal_attention_pool,
    create_fusion_pool
)

# Suppress certain warnings during benchmarking
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    config: Dict[str, Any]
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_allocated_mb: float
    memory_peak_mb: float
    throughput_samples_per_sec: float
    device: str
    dtype: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceProfiler:
    """Lightweight performance profiler for benchmarking."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.cuda_available = torch.cuda.is_available() and device.type == 'cuda'
    
    @contextmanager
    def profile(self, num_warmup: int = 5, num_runs: int = 10):
        """Context manager for timing operations."""
        def run_operation(operation_func):
            # Warmup
            for _ in range(num_warmup):
                operation_func()
                if self.cuda_available:
                    torch.cuda.synchronize()
        
            # Clear cache and reset peak memory
            gc.collect()
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        
            # Actual timing
            start_time = time.perf_counter()
            if self.cuda_available:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
        
            for _ in range(num_runs):
                operation_func()
        
            if self.cuda_available:
                end_event.record()
                torch.cuda.synchronize()
                gpu_time = start_event.elapsed_time(end_event) / num_runs  # ms per run
            else:
                gpu_time = None
        
            end_time = time.perf_counter()
            cpu_time = (end_time - start_time) * 1000 / num_runs  # ms per run
        
            # Memory usage
            if self.cuda_available:
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_allocated = memory_info.rss / 1024**2  # MB
                memory_peak = memory_allocated  # Approximation for CPU
        
            self.last_timing = {
                'cpu_time_ms': cpu_time,
                'gpu_time_ms': gpu_time,
                'memory_allocated_mb': memory_allocated,
                'memory_peak_mb': memory_peak
            }
        
        # Yield the runner function
        yield run_operation


class AECFBenchmarkSuite:
    """Comprehensive benchmark suite for AECF components."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = PerformanceProfiler(self.device)
        self.results: List[BenchmarkResult] = []
        
        print(f"Running benchmarks on {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
    
    def benchmark_operation(
        self,
        operation_name: str,
        operation_fn: Callable,
        config: Dict[str, Any],
        num_warmup: int = 5,
        num_runs: int = 20
    ) -> BenchmarkResult:
        """Benchmark a single operation."""
        
        def run_forward():
            output = operation_fn()
            return output
        
        def run_backward():
            output = operation_fn()
            if isinstance(output, tuple):
                loss = output[0].sum()
            else:
                loss = output.sum()
            loss.backward()
        
        # Forward pass timing
        with self.profiler.profile(num_warmup, num_runs) as profile_runner:
            profile_runner(run_forward)
        
        forward_timing = self.profiler.last_timing.copy()
        
        # Backward pass timing (if gradients are needed)
        try:
            with self.profiler.profile(num_warmup, num_runs) as profile_runner:
                profile_runner(run_backward)
            
            backward_timing = self.profiler.last_timing.copy()
        except Exception:
            # Some operations might not support gradients
            backward_timing = {'cpu_time_ms': 0.0, 'gpu_time_ms': 0.0}
        
        # Calculate throughput
        batch_size = config.get('batch_size', 1)
        time_per_sample = forward_timing['cpu_time_ms'] / batch_size
        throughput = 1000.0 / time_per_sample if time_per_sample > 0 else 0.0
        
        result = BenchmarkResult(
            operation=operation_name,
            config=config,
            forward_time_ms=forward_timing['cpu_time_ms'],
            backward_time_ms=backward_timing['cpu_time_ms'],
            total_time_ms=forward_timing['cpu_time_ms'] + backward_timing['cpu_time_ms'],
            memory_allocated_mb=forward_timing['memory_allocated_mb'],
            memory_peak_mb=forward_timing['memory_peak_mb'],
            throughput_samples_per_sec=throughput,
            device=str(self.device),
            dtype=config.get('dtype', 'float32')
        )
        
        self.results.append(result)
        return result
    
    def benchmark_curriculum_masking(self):
        """Benchmark CurriculumMasking across different configurations."""
        print("\n=== Benchmarking CurriculumMasking ===")
        
        configurations = [
            # Standard configurations
            {'batch_size': 32, 'seq_len': 16, 'base_mask_prob': 0.15},
            {'batch_size': 64, 'seq_len': 32, 'base_mask_prob': 0.15},
            {'batch_size': 128, 'seq_len': 64, 'base_mask_prob': 0.15},
            {'batch_size': 256, 'seq_len': 128, 'base_mask_prob': 0.15},
            
            # Different masking probabilities
            {'batch_size': 64, 'seq_len': 32, 'base_mask_prob': 0.05},
            {'batch_size': 64, 'seq_len': 32, 'base_mask_prob': 0.25},
            {'batch_size': 64, 'seq_len': 32, 'base_mask_prob': 0.50},
            
            # Long sequences
            {'batch_size': 16, 'seq_len': 512, 'base_mask_prob': 0.15},
            {'batch_size': 8, 'seq_len': 1024, 'base_mask_prob': 0.15},
        ]
        
        for config in configurations:
            masking = CurriculumMasking(
                base_mask_prob=config['base_mask_prob']
            ).to(self.device)
            masking.train()
            
            def operation():
                weights = torch.rand(
                    config['batch_size'], 
                    config['seq_len'], 
                    device=self.device,
                    requires_grad=True
                )
                weights = F.softmax(weights, dim=-1)
                masked_weights, info = masking(weights)
                return masked_weights
            
            result = self.benchmark_operation(
                "CurriculumMasking",
                operation,
                config
            )
            
            print(f"Config: {config}")
            print(f"  Forward: {result.forward_time_ms:.2f}ms")
            print(f"  Backward: {result.backward_time_ms:.2f}ms")
            print(f"  Memory: {result.memory_peak_mb:.1f}MB")
            print(f"  Throughput: {result.throughput_samples_per_sec:.0f} samples/sec")
    
    def benchmark_multimodal_attention_pool(self):
        """Benchmark MultimodalAttentionPool across different configurations."""
        print("\n=== Benchmarking MultimodalAttentionPool ===")
        
        configurations = [
            # Standard configurations
            {'batch_size': 32, 'seq_len': 8, 'embed_dim': 256, 'num_heads': 4},
            {'batch_size': 64, 'seq_len': 16, 'embed_dim': 512, 'num_heads': 8},
            {'batch_size': 128, 'seq_len': 32, 'embed_dim': 768, 'num_heads': 12},
            {'batch_size': 256, 'seq_len': 64, 'embed_dim': 1024, 'num_heads': 16},
            
            # Different head counts
            {'batch_size': 64, 'seq_len': 16, 'embed_dim': 512, 'num_heads': 1},
            {'batch_size': 64, 'seq_len': 16, 'embed_dim': 512, 'num_heads': 2},
            {'batch_size': 64, 'seq_len': 16, 'embed_dim': 512, 'num_heads': 16},
            
            # Long sequences
            {'batch_size': 16, 'seq_len': 256, 'embed_dim': 512, 'num_heads': 8},
            {'batch_size': 8, 'seq_len': 512, 'embed_dim': 512, 'num_heads': 8},
        ]
        
        for config in configurations:
            pool = MultimodalAttentionPool(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads']
            ).to(self.device)
            
            def operation():
                query = torch.randn(
                    config['batch_size'], 1, config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                key = torch.randn(
                    config['batch_size'], config['seq_len'], config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                output = pool(query, key)
                return output
            
            result = self.benchmark_operation(
                "MultimodalAttentionPool",
                operation,
                config
            )
            
            print(f"Config: {config}")
            print(f"  Forward: {result.forward_time_ms:.2f}ms")
            print(f"  Backward: {result.backward_time_ms:.2f}ms")
            print(f"  Memory: {result.memory_peak_mb:.1f}MB")
            print(f"  Throughput: {result.throughput_samples_per_sec:.0f} samples/sec")
    
    def benchmark_with_curriculum_masking(self):
        """Benchmark attention pool with curriculum masking."""
        print("\n=== Benchmarking with Curriculum Masking ===")
        
        configurations = [
            {'batch_size': 32, 'seq_len': 8, 'embed_dim': 256, 'num_heads': 4},
            {'batch_size': 64, 'seq_len': 16, 'embed_dim': 512, 'num_heads': 8},
            {'batch_size': 128, 'seq_len': 32, 'embed_dim': 768, 'num_heads': 12},
        ]
        
        for config in configurations:
            # Without curriculum masking
            pool_baseline = MultimodalAttentionPool(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads']
            ).to(self.device)
            
            # With curriculum masking
            masking = CurriculumMasking(base_mask_prob=0.15)
            pool_with_masking = MultimodalAttentionPool(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                curriculum_masking=masking
            ).to(self.device)
            
            def operation_baseline():
                query = torch.randn(
                    config['batch_size'], 1, config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                key = torch.randn(
                    config['batch_size'], config['seq_len'], config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                output = pool_baseline(query, key)
                return output
            
            def operation_with_masking():
                query = torch.randn(
                    config['batch_size'], 1, config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                key = torch.randn(
                    config['batch_size'], config['seq_len'], config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                output, info = pool_with_masking(query, key, return_info=True)
                return output
            
            result_baseline = self.benchmark_operation(
                "AttentionPool_Baseline",
                operation_baseline,
                config
            )
            
            result_with_masking = self.benchmark_operation(
                "AttentionPool_WithMasking",
                operation_with_masking,
                config
            )
            
            overhead_pct = ((result_with_masking.total_time_ms - result_baseline.total_time_ms) 
                          / result_baseline.total_time_ms * 100)
            
            print(f"Config: {config}")
            print(f"  Baseline: {result_baseline.total_time_ms:.2f}ms")
            print(f"  With Masking: {result_with_masking.total_time_ms:.2f}ms")
            print(f"  Overhead: {overhead_pct:.1f}%")
    
    def benchmark_against_pytorch_baseline(self):
        """Benchmark against PyTorch's native MultiheadAttention."""
        print("\n=== Benchmarking Against PyTorch Baseline ===")
        
        configurations = [
            {'batch_size': 32, 'seq_len': 8, 'embed_dim': 256, 'num_heads': 4},
            {'batch_size': 64, 'seq_len': 16, 'embed_dim': 512, 'num_heads': 8},
            {'batch_size': 128, 'seq_len': 32, 'embed_dim': 768, 'num_heads': 12},
        ]
        
        for config in configurations:
            # PyTorch baseline
            pytorch_attn = nn.MultiheadAttention(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                batch_first=True
            ).to(self.device)
            
            # Our implementation
            our_pool = MultimodalAttentionPool(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads']
            ).to(self.device)
            
            # Copy weights for fair comparison
            our_pool.attention.load_state_dict(pytorch_attn.state_dict())
            
            def pytorch_operation():
                query = torch.randn(
                    config['batch_size'], 1, config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                key = torch.randn(
                    config['batch_size'], config['seq_len'], config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                output, _ = pytorch_attn(query, key, key)
                return output
            
            def our_operation():
                query = torch.randn(
                    config['batch_size'], 1, config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                key = torch.randn(
                    config['batch_size'], config['seq_len'], config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                output = our_pool(query, key)
                return output
            
            result_pytorch = self.benchmark_operation(
                "PyTorch_MultiheadAttention",
                pytorch_operation,
                config
            )
            
            result_ours = self.benchmark_operation(
                "AECF_MultimodalAttentionPool",
                our_operation,
                config
            )
            
            overhead_pct = ((result_ours.total_time_ms - result_pytorch.total_time_ms) 
                          / result_pytorch.total_time_ms * 100)
            
            print(f"Config: {config}")
            print(f"  PyTorch: {result_pytorch.total_time_ms:.2f}ms")
            print(f"  AECF: {result_ours.total_time_ms:.2f}ms")
            print(f"  Overhead: {overhead_pct:.1f}%")
    
    def benchmark_scalability(self):
        """Test scalability across different input sizes."""
        print("\n=== Scalability Benchmarks ===")
        
        batch_sizes = [16, 32, 64, 128, 256]
        seq_lens = [8, 16, 32, 64, 128]
        embed_dims = [256, 512, 768, 1024]
        
        print("Batch Size Scaling:")
        for batch_size in batch_sizes:
            config = {'batch_size': batch_size, 'seq_len': 16, 'embed_dim': 512, 'num_heads': 8}
            
            pool = MultimodalAttentionPool(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads']
            ).to(self.device)
            
            def operation():
                query = torch.randn(
                    config['batch_size'], 1, config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                key = torch.randn(
                    config['batch_size'], config['seq_len'], config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                return pool(query, key)
            
            result = self.benchmark_operation(
                "ScalabilityTest_BatchSize",
                operation,
                config,
                num_runs=10
            )
            
            print(f"  Batch {batch_size:3d}: {result.total_time_ms:6.2f}ms, "
                  f"{result.memory_peak_mb:6.1f}MB, "
                  f"{result.throughput_samples_per_sec:6.0f} samples/sec")
        
        print("\nSequence Length Scaling:")
        for seq_len in seq_lens:
            config = {'batch_size': 64, 'seq_len': seq_len, 'embed_dim': 512, 'num_heads': 8}
            
            pool = MultimodalAttentionPool(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads']
            ).to(self.device)
            
            def operation():
                query = torch.randn(
                    config['batch_size'], 1, config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                key = torch.randn(
                    config['batch_size'], config['seq_len'], config['embed_dim'],
                    device=self.device, requires_grad=True
                )
                return pool(query, key)
            
            result = self.benchmark_operation(
                "ScalabilityTest_SeqLen",
                operation,
                config,
                num_runs=10
            )
            
            print(f"  SeqLen {seq_len:3d}: {result.total_time_ms:6.2f}ms, "
                  f"{result.memory_peak_mb:6.1f}MB")
    
    def benchmark_memory_efficiency(self):
        """Test memory efficiency and detect potential memory leaks."""
        print("\n=== Memory Efficiency Tests ===")
        
        pool = MultimodalAttentionPool(embed_dim=512, num_heads=8).to(self.device)
        masking = CurriculumMasking().to(self.device)
        
        config = {'batch_size': 32, 'seq_len': 64, 'embed_dim': 512}
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        memory_samples = []
        
        # Run multiple iterations to check for memory leaks
        for i in range(50):
            query = torch.randn(
                config['batch_size'], 1, config['embed_dim'],
                device=self.device, requires_grad=True
            )
            key = torch.randn(
                config['batch_size'], config['seq_len'], config['embed_dim'],
                device=self.device, requires_grad=True
            )
            
            # Forward pass
            output = pool(query, key)
            loss = output.sum()
            
            # Backward pass
            loss.backward()
            
            # Curriculum masking test
            weights = F.softmax(torch.randn_like(key[:, :, 0]), dim=-1)
            masked_weights, info = masking(weights)
            
            if self.device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated()
                memory_samples.append(current_memory)
            
            # Clean up
            del query, key, output, loss, weights, masked_weights, info
            
            if i % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        if self.device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            
            print(f"Initial memory: {initial_memory / 1024**2:.1f} MB")
            print(f"Final memory: {final_memory / 1024**2:.1f} MB")
            print(f"Memory growth: {memory_growth / 1024**2:.1f} MB")
            print(f"Max memory during test: {max(memory_samples) / 1024**2:.1f} MB")
            
            if memory_growth > 100 * 1024**2:  # 100MB threshold
                print("WARNING: Potential memory leak detected!")
            else:
                print("✓ Memory usage appears stable")
    
    def benchmark_different_dtypes(self):
        """Benchmark performance across different data types."""
        print("\n=== Data Type Performance ===")
        
        dtypes = [torch.float32]
        if self.device.type == 'cuda':
            dtypes.extend([torch.float16, torch.bfloat16])
        
        config = {'batch_size': 64, 'seq_len': 32, 'embed_dim': 512, 'num_heads': 8}
        
        for dtype in dtypes:
            try:
                pool = MultimodalAttentionPool(
                    embed_dim=config['embed_dim'],
                    num_heads=config['num_heads'],
                    dtype=dtype
                ).to(self.device)
                
                def operation():
                    query = torch.randn(
                        config['batch_size'], 1, config['embed_dim'],
                        device=self.device, dtype=dtype, requires_grad=True
                    )
                    key = torch.randn(
                        config['batch_size'], config['seq_len'], config['embed_dim'],
                        device=self.device, dtype=dtype, requires_grad=True
                    )
                    return pool(query, key)
                
                test_config = config.copy()
                test_config['dtype'] = str(dtype).split('.')[-1]
                
                result = self.benchmark_operation(
                    f"DTypeTest_{dtype}",
                    operation,
                    test_config,
                    num_runs=15
                )
                
                print(f"  {str(dtype):15s}: {result.total_time_ms:6.2f}ms, "
                      f"{result.memory_peak_mb:6.1f}MB")
                
            except Exception as e:
                print(f"  {str(dtype):15s}: FAILED ({e})")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group results by operation
        grouped_results = {}
        for result in self.results:
            op_name = result.operation
            if op_name not in grouped_results:
                grouped_results[op_name] = []
            grouped_results[op_name].append(result)
        
        # Compute statistics
        report = {
            "device": str(self.device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "summary": {},
            "detailed_results": [result.to_dict() for result in self.results]
        }
        
        for op_name, results in grouped_results.items():
            forward_times = [r.forward_time_ms for r in results]
            backward_times = [r.backward_time_ms for r in results]
            total_times = [r.total_time_ms for r in results]
            memory_usage = [r.memory_peak_mb for r in results]
            
            report["summary"][op_name] = {
                "count": len(results),
                "forward_time_ms": {
                    "mean": statistics.mean(forward_times),
                    "median": statistics.median(forward_times),
                    "min": min(forward_times),
                    "max": max(forward_times),
                    "std": statistics.stdev(forward_times) if len(forward_times) > 1 else 0
                },
                "total_time_ms": {
                    "mean": statistics.mean(total_times),
                    "median": statistics.median(total_times),
                    "min": min(total_times),
                    "max": max(total_times)
                },
                "memory_mb": {
                    "mean": statistics.mean(memory_usage),
                    "max": max(memory_usage),
                    "min": min(memory_usage)
                }
            }
        
        return report
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🚀 Starting AECF Comprehensive Benchmark Suite")
        print("=" * 60)
        
        # Core component benchmarks
        self.benchmark_curriculum_masking()
        self.benchmark_multimodal_attention_pool()
        
        # Integration benchmarks
        self.benchmark_with_curriculum_masking()
        self.benchmark_against_pytorch_baseline()
        
        # Performance characteristics
        self.benchmark_scalability()
        self.benchmark_memory_efficiency()
        self.benchmark_different_dtypes()
        
        print("\n" + "=" * 60)
        print("🎯 Benchmark Suite Complete!")
        
        # Generate and return report
        report = self.generate_report()
        return report


class RegressionTestSuite:
    """Regression testing against previous performance baselines."""
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Optional[Dict]:
        """Load baseline performance data."""
        if self.baseline_file and os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_baseline(self, results: Dict[str, Any], filename: str):
        """Save current results as baseline for future comparisons."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Baseline saved to {filename}")
    
    def check_regression(self, current_results: Dict[str, Any], threshold: float = 0.1) -> bool:
        """Check for performance regressions."""
        if not self.baseline_data:
            print("No baseline data available for regression testing")
            return True
        
        regressions_found = False
        
        for op_name, current_stats in current_results.get("summary", {}).items():
            if op_name in self.baseline_data.get("summary", {}):
                baseline_stats = self.baseline_data["summary"][op_name]
                
                # Check total time regression
                current_time = current_stats["total_time_ms"]["mean"]
                baseline_time = baseline_stats["total_time_ms"]["mean"]
                
                if current_time > baseline_time * (1 + threshold):
                    regression_pct = ((current_time - baseline_time) / baseline_time) * 100
                    print(f"❌ REGRESSION in {op_name}: {regression_pct:.1f}% slower")
                    regressions_found = True
                else:
                    improvement_pct = ((baseline_time - current_time) / baseline_time) * 100
                    if improvement_pct > 5:  # Notable improvement
                        print(f"✅ IMPROVEMENT in {op_name}: {improvement_pct:.1f}% faster")
        
        return not regressions_found


def main():
    """Run the benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AECF Benchmark Suite")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to run benchmarks on")
    parser.add_argument("--baseline", help="Baseline file for regression testing")
    parser.add_argument("--save-baseline", help="Save results as baseline")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run benchmarks
    suite = AECFBenchmarkSuite(device)
    
    if args.quick:
        # Quick benchmarks for CI
        suite.benchmark_curriculum_masking()
        suite.benchmark_multimodal_attention_pool()
    else:
        # Full benchmark suite
        results = suite.run_all_benchmarks()
        
        # Save results
        if args.save_baseline:
            suite_regression = RegressionTestSuite()
            suite_regression.save_baseline(results, args.save_baseline)
        
        # Check for regressions
        if args.baseline:
            regression_suite = RegressionTestSuite(args.baseline)
            no_regressions = regression_suite.check_regression(results)
            
            if not no_regressions:
                print("\n❌ Performance regressions detected!")
                exit(1)
            else:
                print("\n✅ No performance regressions found!")
        
        # Print summary
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 40)
        for op_name, stats in results.get("summary", {}).items():
            mean_time = stats["total_time_ms"]["mean"]
            mean_memory = stats["memory_mb"]["mean"]
            print(f"{op_name:25s}: {mean_time:6.2f}ms, {mean_memory:6.1f}MB")


if __name__ == "__main__":
    main()
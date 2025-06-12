#!/usr/bin/env python3
"""
AECF Integration Test & Benchmark Runner

This script runs both the comprehensive test suite and benchmark suite
for the AECF implementation, providing a complete validation pipeline
suitable for PyTorch repository integration.

Usage:
    python aecf_integration_runner.py [--mode MODE] [--device DEVICE] [--output DIR]
    
Modes:
    - test: Run only the test suite
    - benchmark: Run only the benchmark suite  
    - both: Run both test and benchmark suites (default)
    - ci: Run CI-friendly quick tests and benchmarks
"""

import argparse
import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import pytest


class AECFIntegrationRunner:
    """Comprehensive test and benchmark runner for AECF."""
    
    def __init__(self, output_dir: Optional[str] = None, device: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("aecf_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.results = {}
        
    def run_tests(self, mode: str = "full") -> bool:
        """Run the test suite."""
        print("🧪 Running AECF Test Suite")
        print("=" * 50)
        
        # Configure pytest arguments based on mode
        if mode == "ci":
            pytest_args = [
                "-v", 
                "--tb=short",
                "-x",  # Stop on first failure
                "aecf/test_aecf.py::TestCurriculumMasking::test_forward_training_mode",
                "aecf/test_aecf.py::TestCurriculumMasking::test_compute_entropy_correctness", 
                "aecf/test_aecf.py::TestMultimodalAttentionPool::test_forward_basic",
                "aecf/test_aecf.py::TestMultimodalAttentionPool::test_curriculum_masking_integration",
                "aecf/test_aecf.py::TestIntegrationAndPerformance::test_convergence_simple_task"
            ]
        elif mode == "quick":
            pytest_args = [
                "-v",
                "--tb=short", 
                "aecf/test_aecf.py::TestCurriculumMasking",
                "aecf/test_aecf.py::TestMultimodalAttentionPool::test_forward_basic",
                "aecf/test_aecf.py::TestMultimodalAttentionPool::test_curriculum_masking_integration"
            ]
        else:  # full
            pytest_args = ["-v", "--tb=short", "aecf/test_aecf.py"]
        
        # Add device-specific markers
        if self.device.type == 'cuda':
            pytest_args.extend(["-m", "not cpu_only"])
        
        # Run tests
        try:
            exit_code = pytest.main(pytest_args)
            success = exit_code == 0
            
            self.results['tests'] = {
                'success': success,
                'exit_code': exit_code,
                'mode': mode,
                'device': str(self.device)
            }
            
            if success:
                print("✅ All tests passed!")
            else:
                print("❌ Some tests failed!")
                
            return success
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            self.results['tests'] = {
                'success': False,
                'error': str(e),
                'mode': mode,
                'device': str(self.device)
            }
            return False
    
    def run_benchmarks(self, mode: str = "full") -> bool:
        """Run the benchmark suite."""
        print("\n🚀 Running AECF Benchmark Suite")
        print("=" * 50)
        
        try:
            # Import benchmark suite (assumes it's in the same directory)
            from aecf_benchmark_suite import AECFBenchmarkSuite, RegressionTestSuite
            
            # Create benchmark suite
            suite = AECFBenchmarkSuite(self.device)
            
            if mode == "ci":
                # Quick benchmarks for CI
                print("Running CI benchmarks...")
                suite.benchmark_curriculum_masking()
                suite.benchmark_multimodal_attention_pool()
                benchmark_results = suite.generate_report()
                
            elif mode == "quick":
                # Essential benchmarks
                print("Running quick benchmarks...")
                suite.benchmark_curriculum_masking()
                suite.benchmark_multimodal_attention_pool()
                suite.benchmark_against_pytorch_baseline()
                benchmark_results = suite.generate_report()
                
            else:  # full
                # Complete benchmark suite
                print("Running full benchmark suite...")
                benchmark_results = suite.run_all_benchmarks()
            
            # Save benchmark results
            benchmark_file = self.output_dir / f"benchmarks_{int(time.time())}.json"
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            print(f"📊 Benchmark results saved to {benchmark_file}")
            
            # Check for performance regressions if baseline exists
            baseline_file = self.output_dir / "baseline_benchmarks.json"
            if baseline_file.exists():
                regression_suite = RegressionTestSuite(str(baseline_file))
                no_regressions = regression_suite.check_regression(benchmark_results, threshold=0.15)
                
                if not no_regressions:
                    print("⚠️  Performance regressions detected!")
                    self.results['benchmarks'] = {
                        'success': False,
                        'regressions': True,
                        'results_file': str(benchmark_file)
                    }
                    return False
            
            self.results['benchmarks'] = {
                'success': True,
                'regressions': False,
                'results_file': str(benchmark_file),
                'summary': benchmark_results.get('summary', {})
            }
            
            print("✅ Benchmarks completed successfully!")
            return True
            
        except ImportError as e:
            print(f"❌ Could not import benchmark suite: {e}")
            print("Make sure aecf_benchmark_suite.py is in the same directory")
            return False
        except Exception as e:
            print(f"❌ Benchmark execution failed: {e}")
            print(f"Error details: {traceback.format_exc()}")
            self.results['benchmarks'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def run_smoke_tests(self) -> bool:
        """Run basic smoke tests to verify installation."""
        print("💨 Running Smoke Tests")
        print("=" * 30)
        
        try:
            # Test imports
            print("Testing imports...")
            from proper_aecf_core import (
                CurriculumMasking, 
                MultimodalAttentionPool, 
                multimodal_attention_pool,
                create_fusion_pool
            )
            print("✅ Imports successful")
            
            # Test basic functionality
            print("Testing basic functionality...")
            
            # Test CurriculumMasking
            masking = CurriculumMasking().to(self.device)
            weights = torch.rand(2, 5, device=self.device)
            weights = torch.softmax(weights, dim=-1)
            result, info = masking(weights)
            assert result.shape == weights.shape
            print("✅ CurriculumMasking working")
            
            # Test MultimodalAttentionPool
            pool = MultimodalAttentionPool(embed_dim=64).to(self.device)
            query = torch.randn(2, 1, 64, device=self.device)
            key = torch.randn(2, 8, 64, device=self.device)
            output = pool(query, key)
            assert output.shape == query.shape
            print("✅ MultimodalAttentionPool working")
            
            # Test functional interface
            output_func = multimodal_attention_pool(query, key)
            assert output_func.shape == query.shape
            print("✅ Functional interface working")
            
            # Test factory function
            fusion_query, fusion_pool = create_fusion_pool(embed_dim=64, num_modalities=3)
            assert fusion_query.shape == (1, 1, 64)
            print("✅ Factory function working")
            
            print("✅ All smoke tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Smoke test failed: {e}")
            print(f"Error details: {traceback.format_exc()}")
            return False
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        report = {
            'timestamp': time.time(),
            'device': str(self.device),
            'torch_version': torch.__version__,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'results': self.results
        }
        
        # Add system info
        if self.device.type == 'cuda':
            report['cuda_version'] = torch.version.cuda
            report['gpu_name'] = torch.cuda.get_device_name()
            report['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory // 1024**2
        
        return report
    
    def save_report(self, report: Dict[str, Any]):
        """Save integration report."""
        report_file = self.output_dir / f"integration_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📋 Integration report saved to {report_file}")
    
    def run_integration_pipeline(self, mode: str = "full") -> bool:
        """Run the complete integration pipeline."""
        print("🔄 Starting AECF Integration Pipeline")
        print("=" * 60)
        print(f"Mode: {mode}")
        print(f"Device: {self.device}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 60)
        
        overall_success = True
        
        # 1. Smoke tests
        if not self.run_smoke_tests():
            print("❌ Smoke tests failed - aborting pipeline")
            return False
        
        # 2. Run tests
        test_success = self.run_tests(mode)
        overall_success = overall_success and test_success
        
        # 3. Run benchmarks (only if tests pass for strict mode)
        if mode == "ci" and not test_success:
            print("⚠️  Skipping benchmarks due to test failures in CI mode")
        else:
            benchmark_success = self.run_benchmarks(mode)
            overall_success = overall_success and benchmark_success
        
        # 4. Generate and save report
        report = self.generate_integration_report()
        self.save_report(report)
        
        # 5. Summary
        print("\n" + "=" * 60)
        print("🎯 AECF Integration Pipeline Complete!")
        print("=" * 60)
        
        if overall_success:
            print("✅ All checks passed - Ready for PyTorch integration!")
        else:
            print("❌ Some checks failed - Please review results")
            
        print(f"Results saved to: {self.output_dir}")
        
        return overall_success


def main():
    parser = argparse.ArgumentParser(
        description="AECF Integration Test & Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python aecf_integration_runner.py                    # Run full pipeline
    python aecf_integration_runner.py --mode ci         # CI-friendly quick run
    python aecf_integration_runner.py --mode test       # Tests only
    python aecf_integration_runner.py --mode benchmark  # Benchmarks only
    python aecf_integration_runner.py --device cpu      # Force CPU usage
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["test", "benchmark", "both", "ci", "quick"],
        default="both",
        help="Which tests to run (default: both)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto", 
        help="Device to run on (default: auto)"
    )
    parser.add_argument(
        "--output",
        help="Output directory for results (default: aecf_results)"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current benchmark results as baseline"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = None  # Let the runner decide
    else:
        device = args.device
    
    # Create runner
    runner = AECFIntegrationRunner(output_dir=args.output, device=device)
    
    # Run based on mode
    try:
        if args.mode == "test":
            success = runner.run_smoke_tests() and runner.run_tests()
        elif args.mode == "benchmark":
            success = runner.run_smoke_tests() and runner.run_benchmarks()
        elif args.mode in ["both", "ci", "quick"]:
            success = runner.run_integration_pipeline(args.mode)
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
        
        # Save baseline if requested
        if args.save_baseline and 'benchmarks' in runner.results:
            baseline_file = runner.output_dir / "baseline_benchmarks.json"
            if runner.results['benchmarks'].get('results_file'):
                import shutil
                shutil.copy(runner.results['benchmarks']['results_file'], baseline_file)
                print(f"📁 Baseline saved to {baseline_file}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Integration pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Integration pipeline crashed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
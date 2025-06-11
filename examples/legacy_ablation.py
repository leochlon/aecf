#!/usr/bin/env python3
"""
AECF Ablation Suite - Main Entry Point
This script runs ablation experiments for the AECF model.
"""
import argparse
import sys
import os
import logging
from pathlib import Path

# Add test directory to Python path
test_dir = str(Path(__file__).parent.absolute())
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

# Import our modularized components
from utils.setup import setup_environment, setup_logging
from ablation.config import AblationConfig
from ablation.suite import AblationSuite


def main():
    """Main entry point for the ablation suite."""
    
    # Setup environment (GPU optimizations, warnings suppression, etc.)
    setup_environment()
    
    # Ensure output directory exists
    output_dir = Path("./ablation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging (both file and console)
    setup_logging(output_dir)
    
    print(f"🚀 AECF Ablation Suite Starting...")
    
    # Parse arguments (with defaults for Colab)
    try:
        parser = argparse.ArgumentParser(description="AECF Ablation Suite")
        parser.add_argument("--ablations", nargs="+", 
                           choices=["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"],
                           default=["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"],
                           help="Specific ablations to run")
        parser.add_argument("--quick", action="store_true", default=False,
                           help="Run quick ablation with 80 epochs (instead of full training)")
        parser.add_argument("--parallel", action="store_true",
                           help="Run ablations in parallel (experimental)")
        parser.add_argument("--output-dir", type=str, default="./ablation_results",
                           help="Output directory for results")
        
        # Parse args (handle Colab notebook context)
        if 'ipykernel' in sys.modules:
            # Running in Jupyter/Colab - use defaults
            args = parser.parse_args([])
        else:
            args = parser.parse_args()
            
        args.output_dir = Path(args.output_dir)
        
    except Exception as e:
        print(f"⚠️ Argument parsing issue: {e}")
        # Use defaults for Colab
        class DefaultArgs:
            ablations = ["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"]
            quick = False
            parallel = False
            output_dir = Path("./ablation_results")
        args = DefaultArgs()
    
    # Create and configure ablation suite
    try:
        suite = AblationSuite(
            data_root=Path("/content/coco2014"),
            cache_dir=Path("./cache"),
            output_dir=args.output_dir,
            parallel=args.parallel
        )
        
        print(f"📋 Configuration:")
        print(f"   • Ablations: {args.ablations}")
        print(f"   • Quick mode: {args.quick}")
        print(f"   • Parallel: {args.parallel}")
        print(f"   • Output: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Suite initialization failed: {e}")
        return 1

    # Run the ablation suite
    try:
        if args.quick:
            from ablation.convenience import run_quick_ablation
            results = run_quick_ablation(args.ablations)
        else:
            results = suite.run_ablations(args.ablations)
        
        print(f"✅ Ablation suite completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Ablation suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
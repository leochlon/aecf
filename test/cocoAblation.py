#!/usr/bin/env python3
"""
AECF Ablation Suite - Main Entry Point

This is the main entry point for the AECF ablation study suite.
The code has been modularized for better maintainability while preserving
all original functionality.

Usage:
    python cocoAblation.py --help
    python cocoAblation.py --ablations full no_gate
    python cocoAblation.py --quick
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure test directory is in Python path
import sys
from pathlib import Path
test_dir = str(Path(__file__).parent.absolute())
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

# Add current directory to Python path for local imports
import sys
from pathlib import Path
current_dir = str(Path(__file__).parent.absolute())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / 'suite.log')
        ]
    )
    setup_logging(output_dir)
    
    print(f"🚀 AECF Ablation Suite Starting...")
    
    # Parse arguments (with defaults for Colab)
    try:
        parser = argparse.ArgumentParser(description="AECF Ablation Suite")
        parser.add_argument("--ablations", nargs="+", 
                           choices=["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"],
                           default=["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"],  # Run all ablations by default
                           help="Specific ablations to run")
        parser.add_argument("--quick", action="store_true", default=False,  # Changed default to False
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
        # Configure for quick mode if enabled
        if args.quick:
            print("⚡ Quick mode enabled: 80 epochs, batch_size=256, optimized for A100")
            # Create quick configs
            quick_configs = {}
            for name in args.ablations:
                if name in suite.STANDARD_ABLATIONS:
                    config = suite.STANDARD_ABLATIONS[name]
                    # Modify for quick execution with A100 optimizations
                    config.max_epochs = 80  # Changed from 5 to 80 epochs
                    config.batch_size = 256  # Larger batch for A100 efficiency
                    config.patience = 15  # Increased patience for longer training
                    quick_configs[name] = config
            results = suite.run_ablations(custom_configs=quick_configs)
        else:
            results = suite.run_ablations(ablation_names=args.ablations)
        
        print(f"\n🎉 Ablation suite completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Ablation suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a minimal test instead
        print(f"\n🔬 Attempting minimal test...")
        try:
            # Simple data loading test
            from aecf import make_clip_tensor_loaders_from_cache
            cache_dir = Path("./cache")
            if cache_dir.exists():
                print("✅ Cache directory exists, basic setup working")
            else:
                print("❌ Cache directory not found")
        except Exception as test_e:
            print(f"❌ Minimal test also failed: {test_e}")
        
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
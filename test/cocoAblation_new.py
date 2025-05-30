#!/usr/bin/env python3
"""
Refactored COCO Ablation Suite - Main Entry Point

This is the main entry point for the modularized COCO ablation suite.
All functionality has been split into logical modules while preserving
complete original behavior.
"""

import argparse
import sys
import os
from pathlib import Path

# Setup environment before importing other modules
from utils.setup import setup_environment, patch_aecf_logging
setup_environment()
patch_aecf_logging()

# Import modularized components
from ablation import AblationSuite, run_quick_ablation, run_full_ablation_suite


def main():
    """Main entry point with argument parsing and execution."""
    # Ensure output directory exists
    os.makedirs("./ablation_results", exist_ok=True)
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('./ablation_results/suite.log')
        ]
    )
    
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
            ablations = ["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"]  # All ablations
            quick = False  # Changed default to False
            parallel = False
            output_dir = Path("./ablation_results")
        args = DefaultArgs()
    
    print(f"📊 Configuration:")
    print(f"   - Ablations: {args.ablations}")
    print(f"   - Quick mode: {args.quick}")
    print(f"   - Output: {args.output_dir}")
    
    try:
        # Create ablation suite with local-friendly paths
        print(f"\n🔧 Initializing AblationSuite...")
        suite = AblationSuite(
            data_root=Path("./coco2014"),  # Local path instead of /content/coco2014
            cache_dir=Path("./cache"), 
            output_dir=args.output_dir
        )
        
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
        else:
            quick_configs = None
        
        # Run ablations
        print(f"\n{'='*60}")
        print("🚀 STARTING ABLATION EXPERIMENTS")
        print(f"{'='*60}")
        
        if quick_configs:
            results = suite.run_ablations(custom_configs=quick_configs)
        else:
            results = suite.run_ablations(args.ablations)
        
        # Display results
        print(f"\n{'='*80}")
        print("📊 ABLATION RESULTS SUMMARY")
        print(f"{'='*80}")
        
        if hasattr(results, 'to_string'):
            # Format the results table for better readability
            print(results.to_string(index=False, float_format='%.4f'))
            
            # Show top performers
            if len(results) > 1:
                print(f"\n🏆 TOP PERFORMERS:")
                if 'Test_Accuracy' in results.columns:
                    best_acc = results.loc[results['Test_Accuracy'].idxmax()]
                    print(f"   Best Accuracy: {best_acc['Ablation']} ({best_acc['Test_Accuracy']:.4f})")
                
                if 'Test_ECE' in results.columns:
                    best_ece = results.loc[results['Test_ECE'].idxmin()]
                    print(f"   Best Calibration (lowest ECE): {best_ece['Ablation']} ({best_ece['Test_ECE']:.4f})")
                
                if 'Test_MAP' in results.columns:
                    best_map = results.loc[results['Test_MAP'].idxmax()]
                    print(f"   Best mAP: {best_map['Ablation']} ({best_map['Test_MAP']:.4f})")
        else:
            print(results)
        
        print(f"\n✅ Ablation suite completed successfully!")
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
            from pathlib import Path
            cache_dir = Path("./cache")
            if cache_dir.exists():
                print("✅ Cache directory exists, basic setup working")
            else:
                print("❌ Cache directory not found")
        except Exception as test_e:
            print(f"❌ Minimal test also failed: {test_e}")


if __name__ == "__main__":
    main()

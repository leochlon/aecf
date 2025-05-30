#!/usr/bin/env python3
"""
Colab entry point for AECF ablation suite.
Run this script in Google Colab to execute ablation experiments.
"""
import os
import sys
from pathlib import Path

def main():
    """Set up environment and run ablation suite."""
    # Get the test directory path
    test_dir = Path(__file__).parent.absolute()
    
    # Add test directory to Python path
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    
    # Run the ablation suite
    from cocoAblation import main as run_ablation
    return run_ablation()

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Create dummy cached CLIP features for testing the AECF model.
This simulates the output of the feature extraction pipeline from oldleg.py.
"""

import torch
from pathlib import Path
import json
import sys

def create_dummy_coco_cache(root_path: str = "/content/coco2014"):
    """Create dummy cached CLIP features matching oldleg.py format."""
    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy cache files at {root}")
    
    # Create split file
    split = {
        "train60k": list(range(60000)),
        "val5k": list(range(60000, 65000)),
        "test5k": list(range(70000, 75000))
    }
    
    split_file = root / "splits_60k5k5k.json"
    with open(split_file, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"✓ Created split file: {split_file}")
    
    # Create dummy feature caches
    cache_configs = [
        ("train_60k", 1000),  # Smaller for testing
        ("val_5k", 100),
        ("test_5k", 100)
    ]
    
    for name, size in cache_configs:
        # Create features matching oldleg.py format exactly
        features = {
            "img": torch.randn(size, 512, dtype=torch.bfloat16),  # CLIP image features
            "txt": torch.randn(size, 512, dtype=torch.bfloat16),  # CLIP text features  
            "y": torch.randint(0, 2, (size, 80), dtype=torch.float16)  # Multi-label (80 COCO classes)
        }
        
        cache_file = root / f"{name}_clip_feats.pt"
        torch.save(features, cache_file)
        print(f"✓ Created cache: {cache_file} with {size} samples")
    
    print("✅ All dummy cache files created successfully!")
    return root

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create dummy COCO cache for testing")
    parser.add_argument("--root", default="/content/coco2014", help="Root directory for cache")
    args = parser.parse_args()
    
    create_dummy_coco_cache(args.root)

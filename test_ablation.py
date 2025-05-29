#!/usr/bin/env python3
"""
Quick test of the COCO ablation suite using cached features.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import from aecf package
from aecf.datasets import make_clip_tensor_loaders_from_cache
from aecf.model import AECF_CLIP
import torch

def test_cached_pipeline():
    """Test that our cached features pipeline works."""
    
    root = Path("./test_coco2014")
    
    print("Testing cached features pipeline...")
    
    # Test data loaders
    try:
        dl_tr, dl_va, dl_te = make_clip_tensor_loaders_from_cache(
            root, 
            batch_size=32,  # Small batch for testing
            num_workers=0   # No multiprocessing for testing
        )
        print(f"✓ Loaded {len(dl_tr.dataset)} train, {len(dl_va.dataset)} val, {len(dl_te.dataset)} test samples")
    except Exception as e:
        print(f"❌ Error loading cached features: {e}")
        return False
    
    # Test model initialization
    try:
        cfg = {
            "task_type": "classification",
            "num_classes": 80,
            "modalities": ["image", "text"],
            "image_encoder_cfg": {"output_dim": 512, "input_dim": 512},
            "text_encoder_cfg": {"output_dim": 512, "input_dim": 512},
            "feature_norm": True,
            "gate_hidden": 2048,
            "masking_mode": "random",
            "p_missing": 0.40,
            "tau": 0.4,
            "entropy_max": 0.05,
            "cec_coef": 0.10,
            "lr": 1e-4,
            "gate_lr": 1e-3,
            "wd": 1e-2,
            "label_freq": torch.ones(80) * 0.0125,  # Uniform frequencies
        }
        
        model = AECF_CLIP(cfg)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False
    
    # Test forward pass
    try:
        # Get one batch
        batch = next(iter(dl_tr))
        print(f"✓ Batch keys: {list(batch.keys())}")
        print(f"✓ Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
        
        # Test model forward pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        features = {m: batch[m] for m in model.modalities}
        logits, gates = model(features)
        
        print(f"✓ Forward pass successful:")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Gates shape: {gates.shape}")
        print(f"  - Expected: logits=[{batch['image'].size(0)}, 80], gates=[{batch['image'].size(0)}, 3]")
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 All tests passed! The ablation suite should work.")
    return True

if __name__ == "__main__":
    test_cached_pipeline()

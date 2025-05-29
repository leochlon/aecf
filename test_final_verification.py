#!/usr/bin/env python3
"""
Final test to verify the COCO ablation suite is ready to run.
"""

import sys
from pathlib import Path
import torch

# Add the current directory to Python path  
sys.path.insert(0, str(Path(__file__).parent))

def test_ablation_ready():
    """Test that all components for the ablation suite are working."""
    
    print("=== AECF COCO Ablation Suite - Final Verification ===")
    
    # Test 1: Cached features exist and load correctly
    print("\n1. Testing cached features...")
    try:
        from aecf.datasets import make_clip_tensor_loaders_from_cache
        
        root = Path("./test_coco2014")
        dl_tr, dl_va, dl_te = make_clip_tensor_loaders_from_cache(
            root, batch_size=16, num_workers=0
        )
        
        # Get a sample batch from each loader
        train_batch = next(iter(dl_tr))
        val_batch = next(iter(dl_va))
        test_batch = next(iter(dl_te))
        
        print(f"✓ Train: {len(dl_tr.dataset)} samples, batch shape: {[(k, v.shape) for k, v in train_batch.items()]}")
        print(f"✓ Val: {len(dl_va.dataset)} samples, batch shape: {[(k, v.shape) for k, v in val_batch.items()]}")
        print(f"✓ Test: {len(dl_te.dataset)} samples, batch shape: {[(k, v.shape) for k, v in test_batch.items()]}")
        
    except Exception as e:
        print(f"❌ Cached features test failed: {e}")
        return False
    
    # Test 2: Model initialization and forward pass
    print("\n2. Testing model...")
    try:
        from aecf.model import AECF_CLIP
        
        cfg = {
            "task_type": "classification",
            "num_classes": 80,
            "modalities": ["image", "text"],
            "image_encoder_cfg": {"output_dim": 512, "input_dim": 512},
            "text_encoder_cfg": {"output_dim": 512, "input_dim": 512},
            "feature_norm": True,
            "gate_hidden": 256,
            "masking_mode": "random",
            "p_missing": 0.40,
            "tau": 0.4,
            "entropy_max": 0.05,
            "cec_coef": 0.10,
            "lr": 1e-4,
            "gate_lr": 1e-3,
            "wd": 1e-2,
            "label_freq": torch.ones(80) * 0.0125,
        }
        
        model = AECF_CLIP(cfg)
        
        # Test forward pass
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        model = model.to(device)
        batch = {k: v.to(device) for k, v in train_batch.items()}
        
        features = {m: batch[m] for m in model.modalities}
        logits, gates = model(features)
        
        print(f"✓ Model forward pass: logits {logits.shape}, gates {gates.shape}")
        
        # Test loss computation
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"])
        print(f"✓ Loss computation: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Ablation suite imports
    print("\n3. Testing ablation suite components...")
    try:
        from test.coco_ablation_suite import (
            evaluate, mk_zero_mod, mk_rnd_mask, 
            learn_global_temperature, apply_temperature,
            build_ablation_table
        )
        
        print("✓ All ablation functions imported successfully")
        
        # Test corruption functions
        zero_img = mk_zero_mod(0)
        zero_txt = mk_zero_mod(1) 
        rnd_mask = mk_rnd_mask(0.3)
        
        # Test corruption application
        corrupted_batch = zero_img(batch.copy())
        print(f"✓ Image zeroing: {corrupted_batch['image'].sum().item():.1f} (should be 0)")
        
        corrupted_batch = zero_txt(batch.copy())
        print(f"✓ Text zeroing: {corrupted_batch['text'].sum().item():.1f} (should be 0)")
        
        corrupted_batch = rnd_mask(batch.copy())
        print("✓ Random masking applied successfully")
        
    except Exception as e:
        print(f"❌ Ablation suite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: PyTorch Lightning integration
    print("\n4. Testing PyTorch Lightning integration...")
    try:
        import pytorch_lightning as pl
        
        # Test trainer creation (don't actually train)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_model_summary=False,
            fast_dev_run=True  # Just validate the setup
        )
        
        print("✓ PyTorch Lightning trainer created successfully")
        
    except Exception as e:
        print(f"❌ PyTorch Lightning test failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("\n" + "="*60)
    print("AECF COCO Ablation Suite is ready to run!")
    print("="*60)
    print("\nTo run the full ablation suite:")
    print("  cd /Users/leo/aecf")
    print("  python test/coco_ablation_suite.py")
    print("\nThe suite will:")
    print("  • Load cached CLIP features from test_coco2014/")
    print("  • Train AECF models with different configurations")
    print("  • Evaluate under multiple input corruptions")
    print("  • Generate ablation results table")
    print("\nNote: The current setup uses dummy data for testing.")
    print("For real COCO evaluation, you'll need actual COCO 2014 dataset.")
    
    return True

if __name__ == "__main__":
    success = test_ablation_ready()
    if not success:
        sys.exit(1)

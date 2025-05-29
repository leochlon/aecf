#!/usr/bin/env python3
"""
Test script to verify the AECF model can handle the batch format from oldleg.py
Creates dummy data matching the ClipTensor format and tests the model forward pass.
"""

import torch
import torch.nn.functional as F
from aecf.model import AECF_CLIP

def create_dummy_clip_batch(batch_size=4, feat_dim=512, num_classes=80):
    """Create a dummy batch matching the ClipTensor format from oldleg.py."""
    
    # Create random CLIP-like features (bfloat16 to match oldleg.py cache format)
    img_features = torch.randn(batch_size, feat_dim, dtype=torch.bfloat16)
    txt_features = torch.randn(batch_size, feat_dim, dtype=torch.bfloat16)
    
    # Create random multi-label targets (80 COCO classes)
    labels = torch.zeros(batch_size, num_classes, dtype=torch.float16)
    for i in range(batch_size):
        # Randomly set 1-3 classes as positive
        num_positive = torch.randint(1, 4, (1,)).item()
        positive_classes = torch.randperm(num_classes)[:num_positive]
        labels[i, positive_classes] = 1.0
    
    # Return batch in the exact format from ClipTensor.__getitem__
    batch = {
        "image": img_features,
        "text": txt_features,
        "label": labels
    }
    
    return batch

def test_model_with_dummy_data():
    """Test AECF_CLIP model with dummy data matching oldleg.py format."""
    
    print("=== Testing AECF Model with Dummy ClipTensor Data ===")
    
    # Create model configuration
    cfg = {
        "task_type": "classification",
        "num_classes": 80,
        "modalities": ["image", "text"],
        "feat_dim": 512,
        "image_encoder_cfg": {"output_dim": 512, "input_dim": 512},
        "text_encoder_cfg": {"output_dim": 512, "input_dim": 512},
        "feature_norm": True,
        "gate_hidden": 256,  # Smaller for testing
        "masking_mode": "none",  # Disable masking for simple test
        "entropy_max": 0.0,     # Disable entropy regularization for simple test
        "cec_coef": 0.0,        # Disable consistency loss for simple test
        "lr": 1e-4,
    }
    
    print(f"Model config: {cfg}")
    
    # Initialize model
    try:
        model = AECF_CLIP(cfg)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False
    
    # Create dummy batch
    try:
        batch = create_dummy_clip_batch(batch_size=4)
        print(f"✓ Created dummy batch with shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape} {v.dtype}")
    except Exception as e:
        print(f"✗ Dummy batch creation failed: {e}")
        return False
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            # Extract features as the model expects
            features = {m: batch[m].float() for m in model.modalities}
            
            # Forward pass
            logits, weights = model(features)
            
            print(f"✓ Forward pass successful:")
            print(f"  logits shape: {logits.shape}")
            print(f"  weights shape: {weights.shape}")
            
            # Test sigmoid probabilities
            probs = torch.sigmoid(logits)
            print(f"  probs range: [{probs.min():.3f}, {probs.max():.3f}]")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test training step format (matching oldleg.py training_step)
    try:
        model.train()
        
        # Test the _shared_step method which processes batches
        loss, metrics = model._shared_step(batch, "train")
        
        print(f"✓ Training step successful:")
        print(f"  loss: {loss.item():.4f}")
        print(f"  metrics: {metrics}")
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! Model is compatible with oldleg.py batch format.")
    return True

if __name__ == "__main__":
    success = test_model_with_dummy_data()
    exit(0 if success else 1)

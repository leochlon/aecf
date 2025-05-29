#!/usr/bin/env python3
"""
Simple test to verify the AECF model can handle batches in oldleg.py format.
Creates dummy cached features to test the pipeline.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add aecf to path
sys.path.insert(0, str(Path(__file__).parent))

def create_dummy_cached_features(root_path: Path, split_name: str, num_samples: int = 100):
    """Create dummy cached CLIP features matching oldleg.py format."""
    root_path = Path(root_path)
    root_path.mkdir(exist_ok=True)
    
    # Create dummy features matching oldleg.py format
    features = {
        "img": torch.randn(num_samples, 512, dtype=torch.bfloat16),  # Image features
        "txt": torch.randn(num_samples, 512, dtype=torch.bfloat16),  # Text features  
        "y": torch.randint(0, 2, (num_samples, 80), dtype=torch.float16)  # Multi-label (80 classes)
    }
    
    cache_file = root_path / f"{split_name}_clip_feats.pt"
    torch.save(features, cache_file)
    print(f"✓ Created dummy cache: {cache_file}")
    return cache_file

def test_batch_format():
    """Test that our model can handle batches from cached features."""
    from aecf.datasets import ClipTensor
    from aecf.model import AECF_CLIP
    from torch.utils.data import DataLoader
    
    print("=== Testing Batch Format ===")
    
    # Create dummy cache files
    cache_dir = Path("/tmp/dummy_coco")
    create_dummy_cached_features(cache_dir, "train_60k", 1000)
    create_dummy_cached_features(cache_dir, "val_5k", 100)
    create_dummy_cached_features(cache_dir, "test_5k", 100)
    
    # Load cached features
    train_obj = torch.load(cache_dir / "train_60k_clip_feats.pt")
    val_obj = torch.load(cache_dir / "val_5k_clip_feats.pt")
    
    # Create datasets
    train_dataset = ClipTensor(train_obj)
    val_dataset = ClipTensor(val_obj)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Check sample format
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
    print(f"Text shape: {sample['text'].shape}, dtype: {sample['text'].dtype}")
    print(f"Label shape: {sample['label'].shape}, dtype: {sample['label'].dtype}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Test batch format
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch text shape: {batch['text'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    
    # Test model initialization
    cfg = {
        "modalities": ["image", "text"],
        "feat_dim": 512,
        "num_classes": 80,
        "task_type": "classification",
        "gate_hidden": 256,
        "masking_mode": "none",
        "p_missing": 0.0,
        "lr": 1e-4
    }
    
    model = AECF_CLIP(cfg)
    model.eval()
    
    print(f"\nModel modalities: {model.modalities}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            # Extract features for model
            features = {m: batch[m] for m in model.modalities}
            print(f"Features keys: {list(features.keys())}")
            print(f"Features['image'] shape: {features['image'].shape}")
            print(f"Features['text'] shape: {features['text'].shape}")
            
            # Forward pass
            logits, weights = model(features)
            print(f"✓ Forward pass successful!")
            print(f"Logits shape: {logits.shape}")
            print(f"Weights shape: {weights.shape}")
            
            # Test loss computation
            labels = batch["label"]
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            print(f"✓ Loss computation successful! Loss: {loss.item():.4f}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! The model can handle oldleg.py batch format.")
    return True

if __name__ == "__main__":
    test_batch_format()

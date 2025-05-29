#!/usr/bin/env python3
"""
Minimal test of the COCO ablation suite with one quick ablation.
"""

import sys
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import required components
from aecf.model import AECF_CLIP
from aecf.datasets import make_clip_tensor_loaders_from_cache, compute_label_freq
from utils.quick_meter import QuickMeter

def test_mini_ablation():
    """Run a minimal ablation test with cached features."""
    
    root = Path("./test_coco2014")
    
    print("=== Mini COCO Ablation Test ===")
    
    # Load datasets using cached CLIP features
    print("Loading cached features...")
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
    
    # Compute label frequency for class weights
    print("Computing label frequencies...")
    try:
        label_freq = compute_label_freq(dl_tr, num_classes=80)
        print(f"✓ Computed label frequencies (min: {label_freq.min():.4f}, max: {label_freq.max():.4f})")
    except Exception as e:
        print(f"Error computing label frequencies: {e}")
        # Use uniform frequencies as fallback
        label_freq = torch.ones(80) * 0.0125  # 1/80 for each class
        print("Using uniform label frequencies as fallback")
    
    # Define minimal configuration for testing
    cfg = {
        # Architecture
        "task_type": "classification",
        "num_classes": 80,
        "modalities": ["image", "text"],
        "image_encoder_cfg": {"output_dim": 512, "input_dim": 512},
        "text_encoder_cfg": {"output_dim": 512, "input_dim": 512},
        "feature_norm": True,
        "gate_hidden": 256,  # Smaller for faster testing

        # Masking curriculum
        "masking_mode": "random",
        "p_missing": 0.40,
        "curr_warmup_epochs": 2,  # Shorter for testing

        # Entropy regularization
        "tau": 0.4,
        "entropy_free": 0,
        "entropy_warmup": 1,  # Shorter for testing
        "entropy_max": 0.05,

        # Expert calibration
        "cec_coef": 0.10,
        "cec_ramp_epochs": 2,  # Shorter for testing

        # Optimization
        "lr": 1e-4,
        "gate_lr": 1e-3,
        "wd": 1e-2,

        # Misc
        "label_freq": label_freq,
    }
    
    print("Initializing model...")
    model = AECF_CLIP(cfg)
    
    # Test training for just 1 epoch
    print("Testing training for 1 epoch...")
    
    # Setup logging and checkpointing
    meter = QuickMeter()
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_mAP", 
        mode="max", 
        filename="test-{epoch:02d}-{val_mAP:.3f}"
    )

    # Lightning callback for metric logging
    class MetricLoggerCallback(pl.Callback):
        def __init__(self, meter, tag):
            self.meter, self.tag = meter, tag

        def on_train_epoch_end(self, trainer, pl_module):
            self.meter.add(trainer.current_epoch, "train",
                          getattr(pl_module, "last_train_metrics", {}))

        def on_validation_epoch_end(self, trainer, pl_module):
            self.meter.add(trainer.current_epoch, "val",
                          getattr(pl_module, "last_val_metrics", {}))

        def on_fit_end(self, trainer, pl_module):
            self.meter.save(f"{self.tag}_diag.csv")

    # Configure trainer for minimal test
    trainer = pl.Trainer(
        max_epochs=1,  # Just 1 epoch for testing
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_model_summary=False,
        callbacks=[ckpt_cb, MetricLoggerCallback(meter, "test")],
        fast_dev_run=False,  # Run full epoch but with limited data
    )

    try:
        print("Starting training...")
        trainer.fit(model, dl_tr, dl_va)
        print("✓ Training completed successfully!")
        
        # Test evaluation
        print("Testing evaluation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Simple evaluation loop
        total_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(dl_te):
                if i >= 3:  # Just test first 3 batches
                    break
                    
                batch = {k: v.to(device) for k, v in batch.items()}
                features = {m: batch[m] for m in model.modalities}
                logits, _ = model(features)
                
                # Simple loss computation
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["label"]
                )
                
                total_samples += batch["label"].size(0)
                total_loss += loss.item() * batch["label"].size(0)
        
        avg_loss = total_loss / total_samples
        print(f"✓ Evaluation completed. Avg loss: {avg_loss:.4f} on {total_samples} samples")
        
        print("🎉 Mini ablation test successful!")
        print("The full ablation suite should work correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during training/evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mini_ablation()

"""Training entry‑point. Usage::

    python -m aecf.train --root ~/coco2014 --epochs 30 --gpus 1
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pytorch_lightning as pl

from .datasets import make_coco_loaders
from .model import AECF_CLIP

def cli_main() -> None:
    p = argparse.ArgumentParser(description="Train Adaptive Ensemble CLIP Fusion (AECF).")
    p.add_argument("--root", type=str, default="~/coco2014",
                   help="Path to COCO‑2014 train/val + annotations.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()

    root = Path(args.root).expanduser()
    train_dl, val_dl = make_coco_loaders(root,
                                        batch_size=args.batch_size,
                                        num_workers=args.workers)
    
    # Create default configuration
    cfg = {
        "task_type": "classification",
        "num_classes": 80,
        "modalities": ["image", "text"],
        "feature_norm": True,
        "gate_hidden": 2048,
        "masking_mode": "random",
        "p_missing": 0.40,
        "curr_warmup_epochs": 20,
        "tau": 0.4,
        "entropy_free": 0,
        "entropy_warmup": 3,
        "entropy_max": 0.05,
        "cec_coef": 0.10,
        "cec_ramp_epochs": 5,
        "lr": 1e-4,
        "gate_lr": 1e-3,
        "wd": 1e-2,
        "epochs": args.epochs,
    }

    model = AECF_CLIP(cfg)
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus or 1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_mAP",
                mode="max",
                filename="best-{epoch:02d}-{val_mAP:.3f}"
            )
        ]
    )
    
    trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    cli_main()

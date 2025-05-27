"""
AV-MNIST training script using AECF components.

This script demonstrates the AECF architecture on the Audio-Visual MNIST dataset.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models

from aecf.model import GatingNet, CurriculumMasker
from aecf.datasets import compute_ece, AVMNISTDataModule

# ------------------------------------------------------------------ encoders
class ImageEncoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.net = backbone
        self.proj = nn.Linear(512, dim)
        
    def forward(self, x):
        return self.proj(self.net(x))

class TextEncoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.emb = nn.Embedding(11, 256)
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.ReLU(), nn.Linear(dim, dim))
        
    def forward(self, x):
        return self.mlp(self.emb(x))

# ------------------------------------------------------------------ Lightning
class AECF_AVMNIST(pl.LightningModule):
    """
    Adaptive Ensemble CLIP Fusion for AV-MNIST.
    Implements modality fusion with gating and masking.
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        
        # Encoders
        self.encoders = nn.ModuleDict({
            "image": ImageEncoder(cfg.get("feat_dim", 512)),
            "text": TextEncoder(cfg.get("feat_dim", 512))
        })
        
        # Gating network
        self.gate = GatingNet(
            input_dim=cfg.get("feat_dim", 512) * 2,
            n_modalities=2,
            hidden_dims=cfg.get("gate_hidden", 1024)
        )
        
        # Classification head
        self.classifier = nn.Linear(cfg.get("feat_dim", 512), 10)
        
        # Masking for training
        self.masker = CurriculumMasker(
            cfg.get("masking_mode", "random"),
            cfg.get("p_missing", 0.5),
            cfg.get("tau", 0.3)
        )

    def forward(self, batch):
        """Forward pass through the model."""
        # Process each modality
        features = {}
        for modality, encoder in self.encoders.items():
            if batch[modality] is not None:
                features[modality] = encoder(batch[modality])
            else:
                features[modality] = torch.zeros(
                    (batch["image"].size(0), self.cfg.get("feat_dim", 512)), 
                    device=self.device
                )
        
        # Concatenate features for gating
        concat_features = torch.cat([features[m] for m in ["image", "text"]], dim=-1)
        
        # Get modality weights
        weights, entropy = self.gate(concat_features)
        
        # Apply masking during training
        if self.training:
            mask = self.masker(weights)
            for i, modality in enumerate(["image", "text"]):
                features[modality] = features[modality] * mask[:, i:i+1]
        
        # Weighted fusion
        fused = torch.zeros_like(features["image"])
        for i, modality in enumerate(["image", "text"]):
            fused += weights[:, i:i+1] * features[modality]
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, weights

    def _shared_step(self, batch, stage):
        """Shared step for training and validation."""
        logits, weights = self(batch)
        loss = F.cross_entropy(logits, batch["label"])
        
        # Calculate metrics
        acc = (logits.argmax(dim=-1) == batch["label"]).float().mean()
        probs = F.softmax(logits, dim=-1)
        ece = compute_ece(probs, batch["label"], n_bins=15)
        
        # Log metrics
        metrics = {
            "loss": loss,
            "acc": acc,
            "ece": ece,
            "w_image": weights[:, 0].mean(),
            "w_text": weights[:, 1].mean()
        }
        
        self.log_dict({f"{stage}_{k}": v for k, v in metrics.items()},
                      prog_bar=(stage == "val"),
                      on_step=False, on_epoch=True)
        
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")
    
    def configure_optimizers(self):
        """Configure optimizers with separate learning rates for gate and other parameters."""
        param_groups = [
            {"params": self.gate.parameters(), 
             "lr": self.cfg.get("gate_lr", 1e-3), 
             "weight_decay": 0.0},
            {"params": [p for n, p in self.named_parameters() if not n.startswith("gate")], 
             "lr": self.cfg.get("lr", 1e-4), 
             "weight_decay": self.cfg.get("weight_decay", 1e-3)},
        ]
        
        opt = torch.optim.AdamW(param_groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.get("epochs", 30))
        return {"optimizer": opt, "lr_scheduler": sched}

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Train AECF on AV-MNIST")
    parser.add_argument("--root", type=str, default="~/data", 
                       help="Root directory for dataset")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    args = parser.parse_args()
    
    # Configuration
    cfg = {
        "feat_dim": 512,
        "gate_hidden": 1024,
        "masking_mode": "random",
        "p_missing": 0.5,
        "tau": 0.3,
        "lr": 1e-4,
        "gate_lr": 1e-3,
        "weight_decay": 1e-3,
        "epochs": args.epochs
    }
    
    # Data module
    dm = AVMNISTDataModule(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Model
    model = AECF_AVMNIST(cfg)
    
    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="avmnist-{epoch:02d}-{val_acc:.3f}",
        save_top_k=1
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus,
        callbacks=[checkpoint_callback]
    )
    
    # Train
    trainer.fit(model, dm)
    
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()

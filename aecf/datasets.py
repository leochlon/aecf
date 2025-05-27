"""
Consolidated data handling module for AECF.

This module provides a unified interface for all datasets used in AECF:
- COCO dataset handling
- AV-MNIST dataset handling
- ClipTensorDataset for pre-extracted features
- Data loading utilities
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import CocoCaptions
import pytorch_lightning as pl

from .masking import apply_adaptive_mask

__all__ = [
    "ensure_coco", 
    "ClipTensorDataset", 
    "CocoClipDataset", 
    "AVMNISTDataModule",
    "make_coco_loaders",
    "make_clip_tensor_loaders",
    "compute_label_freq",
    "compute_ece",
    "batch_map1",
    "focal_bce_loss"
]

# ============================================================================
# COCO Dataset Utilities
# ============================================================================

def ensure_coco(root: Union[str, Path]) -> Path:
    """Checks that *root* contains the 2014 splits."""
    root = Path(root).expanduser().resolve()
    required = [
        root / "train2014",
        root / "val2014",
        root / "annotations" / "captions_train2014.json",
        root / "annotations" / "captions_val2014.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_str = "\n".join(map(str, missing))
        raise OSError(
            f"COCO‑2014 not found in '{root}'.\n"
            f"Missing paths:\n{missing_str}\n"
            "Please download train/val images and caption annotations from "
            "https://cocodataset.org/#download."
        )
    return root

class CocoClipDataset(Dataset):
    """Minimal wrapper around *torchvision* `CocoCaptions`."""

    def __init__(self, root: Union[str, Path], split: str = "train") -> None:
        root = ensure_coco(root)
        img_root = root / f"{split}2014"
        ann_file = root / "annotations" / f"captions_{split}2014.json"
        self.ds = CocoCaptions(str(img_root), str(ann_file),
                               transform=transforms.ToTensor())

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, caps = self.ds[idx]
        caption = caps[0] if isinstance(caps, list) else caps
        return {"image": img, "text": caption, "label": torch.tensor(0)}

def collate_fn(batch):
    """Simple collate that pads nothing (masking operates on features)."""
    return {k: [d[k] for d in batch] for k in batch[0]}

def make_coco_loaders(root: Union[str, Path],
                     batch_size: int = 32,
                     num_workers: int = 4
                     ) -> Tuple[DataLoader, DataLoader]:
    """Factory that returns COCO *train* and *val* dataloaders."""
    train_ds = CocoClipDataset(root, "train")
    val_ds   = CocoClipDataset(root, "val")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_fn)
    return train_dl, val_dl

# ============================================================================
# CLIP Tensor Dataset
# ============================================================================

class ClipTensorDataset(Dataset):
    """Dataset for pre-extracted CLIP features stored in .pt files."""
    
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        """Initialize dataset from a dictionary of tensors."""
        self.data = data_dict
        self.keys = list(data_dict.keys())
        
        # Validate data
        assert 'image' in self.data, "Dataset must contain 'image' features"
        assert 'text' in self.data, "Dataset must contain 'text' features"
        assert 'label' in self.data, "Dataset must contain 'label' tensors"
        
        # Get dataset size
        self.size = len(self.data[self.keys[0]])
        
        # Check all tensors have the same first dimension
        for k, v in self.data.items():
            assert len(v) == self.size, f"Tensor {k} has inconsistent size"
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.data.items()}

def make_clip_tensor_loaders(
    root: Union[str, Path],
    train_file: str = "train_60k_clip_feats.pt",
    val_file: str = "val_5k_clip_feats.pt",
    test_file: str = "test_5k_clip_feats.pt",
    batch_size: int = 512,
    num_workers: int = 4,
    weighted_sampling: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for pre-extracted CLIP features."""
    root = Path(root).expanduser()
    
    # Load datasets
    train_dataset = ClipTensorDataset(torch.load(root / train_file))
    val_dataset = ClipTensorDataset(torch.load(root / val_file))
    test_dataset = ClipTensorDataset(torch.load(root / test_file))
    
    # Create sampler for training set if needed
    train_sampler = None
    if weighted_sampling:
        weights = []
        for sample in train_dataset:
            lbl = sample["label"]
            w = 0.5 if lbl[0] == 1 else 1.0
            weights.append(w)
        
        weights = torch.tensor(weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def compute_label_freq(dataloader, num_classes=80):
    """Compute label frequency for weighting."""
    counts = torch.zeros(num_classes)
    total = 0
    for batch in dataloader:
        labels = batch["label"]
        counts += labels.sum(0)
        total += labels.size(0)
    return (counts / total).clamp_min(1e-4)

# ============================================================================
# AV-MNIST Dataset
# ============================================================================

class AVMNISTDataModule(pl.LightningDataModule):
    """DataModule for Audio-Visual MNIST dataset."""
    
    def __init__(self, root="data", batch_size=512, num_workers=4):
        super().__init__()
        self.root = Path(root)
        self.bs, self.nw = batch_size, num_workers
        tf = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self._train = datasets.MNIST(self.root, train=True,
                                     download=True, transform=tf)
        self._val   = datasets.MNIST(self.root, train=False,
                                     download=True, transform=tf)

    def _wrap(self, ds):
        """Wrap MNIST dataset to return dict with 'image', 'text', and 'label'."""
        class _W:
            def __init__(self, ds): self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                img, lab = self.ds[i]
                return {"image": img, "text": lab, "label": lab}
        return _W(ds)

    def train_dataloader(self):
        return DataLoader(self._wrap(self._train), self.bs,
                          shuffle=True, num_workers=self.nw, pin_memory=True)
                          
    def val_dataloader(self):
        return DataLoader(self._wrap(self._val), self.bs,
                          shuffle=False, num_workers=self.nw, pin_memory=True)

# ============================================================================
# Utility functions (moved from utils.py)
# ============================================================================

def batch_map1(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mini-batch mAP@1 (mean precision-at-1 across classes).

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits (before sigmoid) of shape (B, C)
    labels : torch.Tensor
        Ground truth labels (0/1) of shape (B, C)

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the mAP@1 value.
    """
    B, C = logits.shape
    top1 = logits.argmax(dim=1)                 # (B,)

    # 1 if the predicted class is actually present in the GT for that sample
    correct = labels.gather(1, top1.unsqueeze(1)).squeeze(1).float()  # (B,)

    # per-class precision
    preds_per_class   = torch.bincount(top1, minlength=C).float()     # (C,)
    correct_per_class = torch.bincount(top1, weights=correct, minlength=C)

    # avoid div-by-zero; mask classes never predicted in this batch
    mask = preds_per_class > 0
    precision_per_class = torch.zeros_like(preds_per_class)
    precision_per_class[mask] = correct_per_class[mask] / preds_per_class[mask]

    if mask.any():
        return precision_per_class[mask].mean()
    else:                          # rare corner-case: batch too small
        return torch.tensor(0.0, device=logits.device)

def focal_bce_loss(logits, targets, *, alpha=0.25, gamma=2.0, smooth=0.05, pos_weight=None):
    """
    Focal BCE loss with label smoothing.
    
    Parameters
    ----------
    logits : torch.Tensor
        Raw logits of shape (B, C)
    targets : torch.Tensor
        Ground truth labels (0/1) of shape (B, C)
    alpha : float
        Class balancing factor
    gamma : float
        Focusing parameter for hard examples
    smooth : float
        Label smoothing factor
    pos_weight : torch.Tensor, optional
        Per-class positive weights
        
    Returns
    -------
    torch.Tensor
        Scalar tensor containing the loss value.
    """
    targets = targets.float() * (1 - smooth) + 0.5 * smooth
    p  = torch.sigmoid(logits)
    pt_pos, pt_neg = p, 1 - p
    w_pos = pos_weight if pos_weight is not None else 1.0
    loss = (
        -alpha     * (pt_neg ** gamma) * w_pos * targets     * torch.log(pt_pos + 1e-8)
        -(1-alpha) * (pt_pos ** gamma)        * (1-targets) * torch.log(pt_neg + 1e-8)
    )
    return loss.mean()

def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error.
    
    Parameters
    ----------
    probs : torch.Tensor
        Predicted probabilities
    labels : torch.Tensor
        Ground truth labels
    n_bins : int
        Number of bins for binning confidence scores
        
    Returns
    -------
    torch.Tensor
        Scalar tensor with ECE value
    """
    conf = probs.flatten()
    corr = labels.bool().flatten().float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)

    err = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            err += (conf[m].mean() - corr[m].mean()).abs() * m.float().mean()

    return err

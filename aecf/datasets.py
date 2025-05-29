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

__all__ = [
    "ensure_coco", 
    "ClipTensorDataset", 
    "CocoClipDataset", 
    "AVMNISTDataModule",
    "make_coco_loaders",
    "make_clip_tensor_loaders",
    "make_clip_tensor_loaders_from_cache",
    "setup_coco_cache_pipeline", 
    "CocoPairDataset",
    "ClipTensor",
    "build_clip_cache",
    "make_coco_split",
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

def make_clip_tensor_loaders_from_cache(
    root: Union[str, Path],
    train_file: str = "train_60k_clip_feats.pt",
    val_file: str = "val_5k_clip_feats.pt", 
    test_file: str = "test_5k_clip_feats.pt",
    batch_size: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for pre-cached CLIP features (oldleg.py format)."""
    root = Path(root)
    
    # Load the cached feature files
    train_obj = torch.load(root / train_file)
    val_obj = torch.load(root / val_file)
    test_obj = torch.load(root / test_file)
    
    # Create ClipTensor datasets 
    train_dataset = ClipTensor(train_obj)
    val_dataset = ClipTensor(val_obj)
    test_dataset = ClipTensor(test_obj)
    
    # Create standard DataLoaders (no custom collate_fn needed for tensors)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
# COCO Data Processing Logic (from oldleg.py)
# ============================================================================

import json
import random

# COCO thing class IDs (80 classes)
COCO_THING_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
                  22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
                  41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
                  62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,
                  84,85,86,87,88,89,90]
ID2IDX = {cid: i for i, cid in enumerate(COCO_THING_IDS)}
NUM_CLASSES = len(ID2IDX)

def make_coco_split(root: Union[str, Path], seed: int = 42):
    """Create deterministic COCO split: 60k train, 5k val, 5k test."""
    root = Path(root)
    split_json = root / "splits_60k5k5k.json"
    
    if split_json.exists():
        return json.loads(split_json.read_text())
    
    random.seed(seed)
    
    def sample_ids(ann_file, k):
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError("pycocotools required for COCO dataset processing")
        
        coco = COCO(str(ann_file))
        ids = list(coco.imgToAnns.keys())
        random.shuffle(ids)
        return ids[:k]
    
    train_ann = root / "annotations" / "instances_train2014.json"
    val_ann = root / "annotations" / "instances_val2014.json"
    
    ids = sample_ids(train_ann, 65_000)  # 60k train + 5k val
    split = {
        "train60k": ids[:60_000],
        "val5k": ids[60_000:],
        "test5k": sample_ids(val_ann, 5_000)
    }
    
    split_json.write_text(json.dumps(split, indent=2))
    print("✓ wrote", split_json)
    return split

class CocoPairDataset(Dataset):
    """(img_path, caption, 80-hot label) tuples restricted to a set of IDs."""
    
    def __init__(self, root: Union[str, Path], split_label: str, id_set: List[int]):
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError("pycocotools required for COCO dataset processing")
            
        year = "2014"
        self.root = Path(root)
        self.img_dir = self.root / f"{split_label}{year}"
        self.det = COCO(str(self.root / "annotations" / f"instances_{split_label}{year}.json"))
        self.cap = COCO(str(self.root / "annotations" / f"captions_{split_label}{year}.json"))
        self.ids = [i for i in self.det.imgToAnns.keys() if i in id_set]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        file_name = self.det.loadImgs(img_id)[0]["file_name"]
        caption = self.cap.imgToAnns[img_id][0]["caption"]
        
        # Create multi-label vector
        lab = torch.zeros(NUM_CLASSES, dtype=torch.float16)
        for ann in self.det.imgToAnns[img_id]:
            cid = ann["category_id"]
            if cid in ID2IDX:
                lab[ID2IDX[cid]] = 1.0
        
        return str(self.img_dir / file_name), caption, lab

def build_clip_cache(root: Union[str, Path], subset: str, dataset: Dataset, 
                     clip_arch: str = "ViT-B-32", clip_pretrained: str = "openai",
                     batch_gpu: int = 512, num_workers: int = 8):
    """Extract CLIP features from dataset and save to .pt file."""
    root = Path(root)
    dest = root / f"{subset}_clip_feats.pt"
    
    if dest.exists():
        print(f"⏩ {dest.name} already exists")
        return dest
    
    try:
        import open_clip
        from PIL import Image
        from tqdm.auto import tqdm
    except ImportError:
        raise ImportError("open_clip, PIL, and tqdm required for feature extraction")
    
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_arch, pretrained=clip_pretrained)
    model = model.cuda().eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(clip_arch)
    
    # Create dataloader with custom collate function
    def collate_batch(batch):
        return list(zip(*batch))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_gpu, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=collate_batch
    )
    
    feats_i, feats_t, labs = [], [], []
    
    for img_paths, caps, lab_batch in tqdm(dataloader, leave=False, desc=dest.name):
        # Process images
        imgs = torch.stack([
            preprocess(Image.open(p).convert("RGB")) 
            for p in img_paths
        ]).cuda(non_blocking=True)
        
        # Extract features
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            fi = model.encode_image(imgs)
            ft = model.encode_text(tokenizer(caps).cuda(non_blocking=True))
        
        feats_i.append(fi.cpu())
        feats_t.append(ft.cpu()) 
        labs.append(torch.stack(lab_batch))
    
    # Save features in the exact format from oldleg.py
    obj = {
        "img": torch.cat(feats_i).bfloat16(),
        "txt": torch.cat(feats_t).bfloat16(),
        "y": torch.cat(labs)
    }
    
    torch.save(obj, dest)
    print(f"✓ saved {dest} ({len(dataset):,})")
    return dest

class ClipTensor(Dataset):
    """Dataset for pre-extracted CLIP features (matches oldleg.py format)."""
    
    def __init__(self, obj):
        self.img = obj["img"]
        self.txt = obj["txt"] 
        self.y = obj["y"]
    
    def __len__(self):
        return self.y.size(0)
    
    def __getitem__(self, i):
        return {
            "image": self.img[i],
            "text": self.txt[i], 
            "label": self.y[i]
        }

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

def setup_coco_cache_pipeline(root: Union[str, Path], seed: int = 42):
    """Setup the complete COCO caching pipeline from oldleg.py."""
    root = Path(root)
    
    # 1. Create the deterministic split
    split = make_coco_split(root, seed)
    
    # 2. Create raw datasets for each split
    train_dataset = CocoPairDataset(root, "train", split["train60k"])
    val_dataset = CocoPairDataset(root, "train", split["val5k"])  # Note: uses train2014 imgs
    test_dataset = CocoPairDataset(root, "val", split["test5k"])  # Note: uses val2014 imgs
    
    # 3. Build CLIP feature caches
    build_clip_cache(root, "train_60k", train_dataset)
    build_clip_cache(root, "val_5k", val_dataset) 
    build_clip_cache(root, "test_5k", test_dataset)
    
    return split

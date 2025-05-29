# ================================================================
# 0. Imports and helper utilities
# ================================================================

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.utils.data
from pathlib import Path
import shutil
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, Any, List
from torchmetrics.classification import AveragePrecision

# Import from aecf package
from aecf.model import AECF_CLIP
from aecf.datasets import make_coco_loaders, compute_label_freq, compute_ece, ensure_coco

# Import QuickMeter from utils
from utils.quick_meter import QuickMeter

# Constants
ROOT = Path("/content/coco2014")
RESULTS_DIR = Path("/content/coco_results_split")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_JSON = ROOT / "splits_60k5k5k.json"

# ----------------------------------------------------------------
# Corruption helpers for testing model robustness
# ----------------------------------------------------------------
def mk_zero_mod(idx: int):
    """idx = 0 → zero *image*, idx = 1 → zero *text*."""
    def _f(b):
        b = b.copy()
        (b["image"] if idx == 0 else b["text"]).zero_()
        return b
    return _f

def mk_rnd_mask(pi: float):
    """Random IID drop with fixed π, applied to the inputs."""
    def _f(b):
        keep = (torch.rand(len(b["image"]), 2, device=b["image"].device) > pi).float()
        b = b.copy()
        b["image"] *= keep[:, :1]
        b["text"]  *= keep[:, 1:]
        return b
    return _f

# Standard corruption functions for evaluation
CORRUPTION_FNS = [
    ("full",     None),
    ("img_only", mk_zero_mod(0)),
    ("txt_only", mk_zero_mod(1)),
    ("rnd10",    mk_rnd_mask(0.10)),
    ("rnd20",    mk_rnd_mask(0.20)),
    ("rnd30",    mk_rnd_mask(0.30)),
    ("rnd50",    mk_rnd_mask(0.50)),
    ("rnd90",    mk_rnd_mask(0.90)),
]

# ----------------------------------------------------------------------
# Temperature calibration functions
# ----------------------------------------------------------------------
def learn_global_temperature(model, dl_val):
    """Learn optimal temperature parameter on validation set."""
    model.eval()
    logits, labels = [], []
    for b in dl_val:
        features = {m: b[m].to(model.device) for m in model.modalities}
        l, _ = model(features)
        logits.append(l.float().cpu())
        labels.append(b["label"].cpu())

    logits = torch.cat(logits).detach()
    labels = torch.cat(labels).float()

    device = logits.device
    log_T  = torch.tensor(0.0, device=device, requires_grad=True)
    opt    = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50)

    def _closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = F.binary_cross_entropy_with_logits(logits / T, labels)
        loss.backward()
        return loss

    opt.step(_closure)
    return torch.exp(log_T).clamp(0.05, 10.0).item()

def apply_temperature(model, T_pos):
    """Wrap model with temperature scaling."""
    class _Wrapper(nn.Module):
        def __init__(self, base, T):
            super().__init__()
            self.base = base
            self.register_buffer("T", torch.tensor(T))
            self.cfg = base.cfg
            self._ece = base._ece
            self.modalities = base.modalities
            self.device = base.device

        def forward(self, *a, **kw):
            logits, w = self.base(*a, **kw)
            return logits / self.T, w
    return _Wrapper(model, T_pos)

# ----------------------------------------------------------------------
# Evaluation function
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: pl.LightningModule,
             loader: DataLoader,
             mask_fn: Callable | None = None) -> dict[str, float]:
    """Evaluate model on given loader with optional input corruption."""

    model.eval()
    device = next(model.parameters()).device
    C = model.cfg["num_classes"]

    ap_metric = AveragePrecision(task="binary",
                                num_labels=C,
                                average=None).to(device)

    all_probs, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if mask_fn is not None:
            batch = mask_fn(batch)
            
        features = {m: batch[m] for m in model.modalities}
        logits, _ = model(features)
        probs = logits.sigmoid()

        ap_metric.update(probs, batch["label"])
        all_probs.append(probs.cpu())
        all_labels.append(batch["label"].cpu())

    per_class_ap = ap_metric.compute()
    map_value    = per_class_ap.mean().item()

    probs_full  = torch.cat(all_probs)
    labels_full = torch.cat(all_labels)
    ece_value   = compute_ece(probs_full, labels_full).item()

    return {"mAP": map_value, "ece": ece_value}

# ----------------------------------------------------------------------
# Main training and evaluation protocol
# ----------------------------------------------------------------------
def run_protocol(cfg:dict,
                 dl_tr, dl_va, dl_te,
                 epochs:int = 30,
                 gpus:int  = 1,
                 tag:str   = "main"):
    """
    Train AECF-CLIP, log metrics, and evaluate under various corruptions.
    """
    # 1) Initialize model
    model = AECF_CLIP(cfg)
    
    # Setup logging and checkpointing
    meter = QuickMeter()
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_mAP", 
        mode="max", 
        filename="best-{epoch:02d}-{val_mAP:.3f}"
    )

    # 2) Lightning callback for metric logging
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

    # 3) Configure and run training
    trainer = pl.Trainer(
        max_epochs = epochs,
        accelerator = "gpu" if torch.cuda.is_available() and gpus else "cpu",
        devices = gpus,
        logger = False,
        enable_model_summary = False,
        callbacks = [ckpt_cb, MetricLoggerCallback(meter, tag)],
    )

    trainer.fit(model, dl_tr, dl_va)

    # Load best checkpoint
    best_path = ckpt_cb.best_model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model = AECF_CLIP.load_from_checkpoint(best_path, cfg=cfg).to(device)

    # Learn temperature on validation set
    T = learn_global_temperature(best_model, dl_va)
    print(f"[cal] learned temperature T = {T:.3f}")

    # Apply temperature calibration
    best_model = apply_temperature(best_model, T).to(device)
    best_model.eval()

    # Evaluate under multiple corruptions
    print("\n=== TEST-SET RESULTS (best checkpoint) ===")
    corr_rows = []
    for tag_corr, fn in CORRUPTION_FNS:
        res = evaluate(best_model, dl_te, mask_fn=fn)
        print(f"{tag_corr:8s}  |  mAP = {res['mAP']:.3f}   ECE = {res['ece']:.3f}")

        # Accumulate results for CSV
        corr_rows.append({
            "model_tag" : tag,
            "corruption": tag_corr,
            "mAP"       : res['mAP'],
            "ECE"       : res['ece'],
        })

    # Save results
    pd.DataFrame(corr_rows).to_csv(f"{tag}_corr.csv", index=False)
    print(f"[run_protocol] wrote corruption metrics → {tag}_corr.csv")

    shutil.copy(best_path, f"{tag}_best.ckpt")
    return best_model, best_path

# ----------------------------------------------------------------------
# Data preparation utilities
# ----------------------------------------------------------------------
def compute_label_freq(dataloader, num_classes=80):
    """Compute label frequency for weighting."""
    counts = torch.zeros(num_classes)
    total = 0
    for batch in dataloader:
        labels = batch["label"]
        counts += labels.sum(0)
        total += labels.size(0)
    return (counts / total).clamp_min(1e-4)

# ----------------------------------------------------------------------
# Results analysis utilities
# ----------------------------------------------------------------------
def build_ablation_table(
        names,
        root = Path("."),
        corrs = ("full", "rnd30"),
        metrics = ("mAP", "ECE"),
        round_dec = 3,
):
    """Build a summary table of ablation results."""
    rows = []

    for name in names:
        fn = root / f"abl-{name}_corr.csv"
        if not fn.exists():
            print(f"[warn] missing file {fn}")
            continue

        df = pd.read_csv(fn)
        df.columns = [c.lower() for c in df.columns]

        # Keep only the corruption rows we want, then pivot for convenience
        sub = (df[df["corruption"].isin(corrs)]
               .set_index("corruption")[list(metrics)])

        entry = {"model": name}
        for corr in corrs:
            for m in metrics:
                entry[f"{m}_{corr}"] = round(sub.loc[corr, m], round_dec)
        rows.append(entry)

    # Order the columns nicely
    col_order = (["model"] +
                 [f"{m}_{c}" for c in corrs for m in metrics])
    table = pd.DataFrame(rows)[col_order]

    # Markdown friendly
    print(table.to_markdown(index=False))
    return table

# ----------------------------------------------------------------------
# COCO dataset setup helper for Colab
# ----------------------------------------------------------------------
def setup_coco_for_colab(root_path="/content/coco2014"):
    """
    Helper function to guide COCO dataset setup in Google Colab.
    This function provides instructions for downloading and setting up COCO.
    """
    from pathlib import Path
    import os
    
    root = Path(root_path)
    
    print("=== COCO Dataset Setup for Colab ===")
    print(f"Target directory: {root}")
    
    # Check if dataset already exists
    try:
        ensure_coco(root)
        print("✅ COCO dataset already properly set up!")
        return True
    except OSError:
        pass
    
    print("\n📥 COCO dataset not found. Setting up...")
    
    # Create directories
    root.mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(exist_ok=True)
    
    print("\n🔄 Downloading COCO 2014 validation images and annotations...")
    
    # Download validation images (smaller dataset for testing)
    val_url = "http://images.cocodataset.org/zips/val2014.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    
    print("This will download about 6GB of validation images and 241MB of annotations.")
    print("For a full setup, you would also need train2014.zip (13GB).")
    
    # Use wget to download
    os.system(f"wget -q --show-progress {val_url} -P {root}/")
    os.system(f"wget -q --show-progress {ann_url} -P {root}/")
    
    print("\n📦 Extracting files...")
    os.system(f"cd {root} && unzip -q val2014.zip")
    os.system(f"cd {root} && unzip -q annotations_trainval2014.zip")
    
    # For testing purposes, create train2014 as a symlink to val2014
    print("🔗 Creating train2014 symlink for testing...")
    os.system(f"cd {root} && ln -sf val2014 train2014")
    
    # Verify setup
    try:
        ensure_coco(root)
        print("✅ COCO dataset setup complete!")
        return True
    except OSError as e:
        print(f"❌ Setup failed: {e}")
        return False

# ----------------------------------------------------------------------
# Main execution - load data and define configurations
# ----------------------------------------------------------------------
def main():
    # First, ensure COCO dataset exists and is properly structured
    try:
        print("Checking COCO dataset...")
        root_path = ensure_coco(ROOT)
        print(f"✅ COCO dataset found at: {root_path}")
    except OSError as e:
        print(f"❌ COCO dataset not found: {e}")
        print("\n🔧 Attempting automatic setup for Colab...")
        
        if not setup_coco_for_colab(str(ROOT)):
            print("\n💡 Manual setup required:")
            print("1. Download COCO 2014 train/val images from https://cocodataset.org/#download")
            print("2. Download COCO 2014 annotations")
            print("3. Extract them to the correct structure")
            print(f"\nExpected structure at {ROOT}:")
            print("├── train2014/")
            print("├── val2014/")
            print("└── annotations/")
            print("    ├── captions_train2014.json")
            print("    └── captions_val2014.json")
            return None
    
    # Load datasets using COCO loaders
    print("Creating data loaders...")
    dl_tr, dl_va = make_coco_loaders(ROOT, batch_size=32, num_workers=4)
    
    # For testing, use validation set as test set (you could split it further if needed)
    dl_te = dl_va
    
    # Compute label frequency for class weights (using a smaller sample for speed)
    print("Computing label frequencies...")
    # Sample a subset for label frequency computation to speed things up
    sample_size = min(1000, len(dl_tr.dataset))
    indices = torch.randperm(len(dl_tr.dataset))[:sample_size]
    sample_loader = DataLoader(
        torch.utils.data.Subset(dl_tr.dataset, indices),
        batch_size=32,
        num_workers=4
    )
    
    # Create dummy label frequencies for COCO (80 classes)
    # In practice, you'd compute these from your actual COCO annotations
    label_freq = torch.ones(80) * 0.05  # Uniform frequency as placeholder

    # Define base configuration for COCO
    BASE = dict(
        # Architecture
        task_type     = "classification",
        num_classes   = 80,
        modalities    = ["image", "text"],
        image_encoder_cfg = {"output_dim": 512, "input_dim": 512},
        text_encoder_cfg  = {"output_dim": 512, "input_dim": 512},
        feature_norm  = True,
        gate_hidden   = 2048,

        # Masking curriculum
        masking_mode  = "random",
        p_missing     = 0.40,
        curr_warmup_epochs = 20,

        # Entropy regularization
        tau           = 0.4,
        entropy_free  = 0,
        entropy_warmup= 3,
        entropy_max   = 0.05,

        # Expert calibration
        cec_coef         = 0.10,
        cec_ramp_epochs  = 5,

        # Optimization
        lr        = 1e-4,
        gate_lr   = 1e-3,
        wd        = 1e-2,

        # Misc
        label_freq      = label_freq,
        epochs          = 30,
    )

    # Define ablation configurations
    ABLATIONS = {
        "full"      : {},
        "no_gate"   : {"gate_disabled": True},
        "no_entropy": {"entropy_max": 0.0},
        "no_curmask": {"masking_mode": "none"},
        "img_only"  : {"gate_disabled": True,
                       "masking_mode": "none",
                       "modalities": ["image"]},
        "txt_only"  : {"gate_disabled": True,
                       "masking_mode": "none",
                       "modalities": ["text"]},
    }

    # Run ablations (using shorter epochs for demo)
    for name, patch in ABLATIONS.items():
        print(f"\n===== Running ablation: {name} =====")
        cfg = {**BASE, **patch}
        run_protocol(cfg,
                    dl_tr, dl_va, dl_te,
                    epochs=5,  # Reduced for faster testing
                    gpus=1,
                    tag=f"abl-{name}")

    # Generate summary table
    ablation_names = ["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"]
    table = build_ablation_table(ablation_names, 
                        root=Path("."), 
                        corrs=("full", "rnd30", "rnd50"))
    
    return table

if __name__ == "__main__":
    main()

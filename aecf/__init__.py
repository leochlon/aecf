"""AECF – Adaptive Ensemble CLIP Fusion (NeurIPS 2025).

Import conveniences::

    from aecf import AECF_CLIP, make_loaders, GatingNet, CurriculumMasker
"""
from .model import AECF_CLIP, GatingNet, CurriculumMasker
from .datasets import (
    make_coco_loaders,
    ClipTensorDataset,
    AVMNISTDataModule,
    compute_label_freq,
    batch_map1,
    focal_bce_loss,
    compute_ece
)

__all__ = [
    "AECF_CLIP", 
    "make_coco_loaders",
    "ClipTensorDataset",
    "AVMNISTDataModule",
    "CurriculumMasker",
    "GatingNet",
    "batch_map1",
    "focal_bce_loss",
    "compute_ece",
    "compute_label_freq"
]

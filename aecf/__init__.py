"""AECF – Adaptive Ensemble CLIP Fusion (NeurIPS 2025).

Import conveniences::

    from aecf import AECF_CLIP, AECFConfig, create_aecf_model
"""

# Import refactored components
from .config import AECFConfig, create_default_config, validate_config_compatibility
from .model_refactored import AECF_CLIP, create_aecf_model, validate_model_inputs, test_model_forward
from .training import AECFTrainer, MemoryManagementCallback, MetricsAccumulator
from .components import (
    AdaptiveGate, CurriculumMasker, EncoderFactory, OutputHeadFactory,
    validate_feature_dict, normalize_features, validate_tensor_input
)
from .losses import AECFLoss
from .metrics import AECFMetrics

# Legacy compatibility - components now fully modular

# Import datasets
from .datasets import (
    make_coco_loaders,
    make_clip_tensor_loaders_from_cache,
    setup_coco_cache_pipeline,
    ClipTensorDataset,
    ClipTensor,
    CocoPairDataset,
    build_clip_cache,
    make_coco_split,
    AVMNISTDataModule,
    compute_label_freq,
    batch_map1,
    focal_bce_loss,
    compute_ece
)

__all__ = [
    # Core refactored components
    "AECF_CLIP",
    "AECFConfig", 
    "AECFTrainer",
    "create_aecf_model",
    "create_default_config",
    "validate_config_compatibility",
    "validate_model_inputs",
    "test_model_forward",
    
    # Component modules
    "AdaptiveGate",
    "CurriculumMasker",
    "EncoderFactory",
    "OutputHeadFactory",
    "validate_feature_dict",
    "normalize_features",
    "validate_tensor_input",
    
    # Loss and metrics
    "AECFLoss",
    "AECFMetrics",
    
    # Training utilities
    "MemoryManagementCallback",
    "MetricsAccumulator",
    
    # Dataset components
    "make_coco_loaders",
    "make_clip_tensor_loaders_from_cache",
    "setup_coco_cache_pipeline",
    "ClipTensorDataset",
    "ClipTensor",
    "CocoPairDataset",
    "build_clip_cache",
    "make_coco_split",
    "AVMNISTDataModule",
    "batch_map1",
    "focal_bce_loss",
    "compute_ece",
    "compute_label_freq"
]

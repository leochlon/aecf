"""
Constants and default configurations for COCO Ablation Suite.
"""
from pathlib import Path

# Default paths configuration
DEFAULT_PATHS = {
    'data_root': Path("./coco2014"),
    'cache_dir': Path("./cache"),
    'output_dir': Path("./ablation_results")
}

# GPU optimization settings
GPU_OPTIMIZATIONS = {
    'precision': '16-mixed',
    'accumulate_grad_batches': 1,
    'strategy': 'auto',
    'benchmark': True,
    'allow_tf32': True,
    'float32_matmul_precision': 'medium'
}

# COCO dataset specifications
COCO_SPECS = {
    'num_classes': 80,
    'feature_dim': 512,
    'splits': ["train", "val", "test"],
    'expected_dtypes': {
        "image": "torch.float32",  # CLIP features normalized to float32
        "text": "torch.float32",   # CLIP features normalized to float32
        "label": "torch.float32"   # Multi-label targets as float32
    },
    'expected_shapes': {
        "image": [512],   # CLIP ViT-B/32 features
        "text": [512],    # CLIP text features
        "label": [80]     # 80 COCO classes
    }
}

# Required COCO files structure
COCO_FILES_REQUIRED = {
    "annotations/instances_train2014.json": "",
    "annotations/instances_val2014.json": "",
    "annotations/captions_train2014.json": "",
    "annotations/captions_val2014.json": "",
    "train2014": "",
    "val2014": ""
}

# Cache file names
CACHE_FILES = {
    "coco_clip_cache_train.pt": "",
    "coco_clip_cache_val.pt": "",
    "coco_clip_cache_test.pt": "",
    "coco_manifest.json": ""
}

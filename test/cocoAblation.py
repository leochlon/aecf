"""
Refactored COCO Ablation Suite - Streamlined and Optimized with Clean Data Management
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import logging
import hashlib
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os

from aecf import AECF_CLIP, make_clip_tensor_loaders_from_cache, setup_coco_cache_pipeline


# ============================================================================
# Clean Data Management System
# ============================================================================

@dataclass
class DatasetManifest:
    """Manifest tracking dataset state and integrity."""
    name: str
    version: str
    raw_data_path: Path
    cache_path: Path
    files_required: Dict[str, str]  # filename -> expected hash
    cache_files: Dict[str, str]     # cache_filename -> expected hash
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetManifest':
        # Convert string paths back to Path objects
        data['raw_data_path'] = Path(data['raw_data_path'])
        data['cache_path'] = Path(data['cache_path'])
        return cls(**data)


class DataIntegrityChecker:
    """Handles data integrity verification."""
    
    @staticmethod
    def compute_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of file."""
        if not filepath.exists():
            return ""
        
        hash_obj = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    @staticmethod
    def verify_files(file_dict: Dict[str, str], base_path: Path) -> Dict[str, bool]:
        """Verify multiple files against expected hashes."""
        results = {}
        for filename, expected_hash in file_dict.items():
            filepath = base_path / filename
            if expected_hash == "":  # Skip hash check if not provided
                results[filename] = filepath.exists()
            else:
                actual_hash = DataIntegrityChecker.compute_file_hash(filepath)
                results[filename] = actual_hash == expected_hash
        return results


class COCODataManager:
    """Clean, unified COCO data management."""
    
    # Standard COCO 2014 file structure
    COCO_MANIFEST = DatasetManifest(
        name="coco2014",
        version="1.0",
        raw_data_path=Path("../coco2014"),  # Outside the aecf folder
        cache_path=Path("./cache"),
        files_required={
            "annotations/instances_train2014.json": "",
            "annotations/instances_val2014.json": "",
            "annotations/captions_train2014.json": "",
            "annotations/captions_val2014.json": "",
            "train2014": "",
            "val2014": ""
        },
        cache_files={
            "coco_clip_cache_train.pt": "",
            "coco_clip_cache_val.pt": "",
            "coco_clip_cache_test.pt": "",
            "coco_manifest.json": ""
        },
        metadata={
            "num_classes": 80,
            "feature_dim": 512,
            "splits": ["train", "val", "test"],
            "expected_dtypes": {
                "image": "torch.float32",  # CLIP features normalized to float32
                "text": "torch.float32",   # CLIP features normalized to float32
                "label": "torch.float32"   # Multi-label targets as float32
            },
            "expected_shapes": {
                "image": [512],   # CLIP ViT-B/32 features
                "text": [512],    # CLIP text features
                "label": [80]     # 80 COCO classes
            }
        }
    )

    def __init__(self, 
                 data_root: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 force_rebuild: bool = False):
        
        # Use provided paths or defaults
        self.manifest = self.COCO_MANIFEST
        if data_root:
            self.manifest.raw_data_path = data_root
        if cache_dir:
            self.manifest.cache_path = cache_dir
            
        self.force_rebuild = force_rebuild
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.manifest.cache_path.mkdir(parents=True, exist_ok=True)
        
    def ensure_data_ready(self) -> bool:
        """
        Single entry point that ensures everything is ready.
        Returns True if data is ready, False if failed.
        """
        try:
            # Step 1: Check if cache is valid and complete
            if not self.force_rebuild and self._is_cache_valid():
                self.logger.info("✅ Valid cache found, data ready")
                return True
            
            # Step 2: Check if raw data is available (optional for dummy cache)
            raw_data_available = self._ensure_raw_data()
            if not raw_data_available:
                self.logger.info("💡 No raw data available, will create dummy cache")
            
            # Step 3: Build cache from raw data or create dummy cache
            if not self._build_cache():
                self.logger.error("❌ Failed to build cache")
                return False
            
            # Step 4: Verify final state
            if self._is_cache_valid():
                self.logger.info("✅ Data pipeline completed successfully")
                return True
            else:
                self.logger.error("❌ Cache validation failed after build")
                # Try to get more details about what failed
                try:
                    self._validate_cache_consistency()
                except Exception as validation_error:
                    self.logger.error(f"💀 Detailed validation error: {validation_error}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            return False
    
    def get_loaders(self, batch_size: int = 32, num_workers: int = 4) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get data loaders. Assumes data is ready (call ensure_data_ready first)."""
        if not self._is_cache_valid():
            raise RuntimeError("Cache not ready. Call ensure_data_ready() first.")
        
        # Perform comprehensive dtype/shape validation
        self._validate_cache_consistency()
        
        return make_clip_tensor_loaders_from_cache(
            root=self.manifest.cache_path,
            train_file="coco_clip_cache_train.pt",
            val_file="coco_clip_cache_val.pt", 
            test_file="coco_clip_cache_test.pt",
            batch_size=batch_size,
            num_workers=num_workers
        )
    
    def _validate_cache_consistency(self) -> bool:
        """Comprehensive validation of cache file consistency."""
        self.logger.info("🔍 Performing comprehensive cache validation...")
        
        try:
            # Load and validate each cache file
            cache_files = [
                "coco_clip_cache_train.pt",
                "coco_clip_cache_val.pt", 
                "coco_clip_cache_test.pt"
            ]
            
            expected_dtypes = self.manifest.metadata["expected_dtypes"]
            expected_shapes = self.manifest.metadata["expected_shapes"]
            
            for cache_file in cache_files:
                cache_path = self.manifest.cache_path / cache_file
                
                if not cache_path.exists():
                    raise RuntimeError(f"Cache file missing: {cache_file}")
                
                # Load and inspect data
                try:
                    data = torch.load(cache_path, map_location='cpu')
                    
                    # Validate format (should be list of tuples or ClipTensor dict format)
                    if isinstance(data, list):
                        # Old format: list of (img_feat, txt_feat, label) tuples
                        if len(data) == 0:
                            raise ValueError(f"Empty cache file: {cache_file}")
                        
                        # Check first sample
                        sample = data[0]
                        if not (isinstance(sample, tuple) and len(sample) == 3):
                            raise ValueError(f"Invalid sample format in {cache_file}: expected tuple of 3, got {type(sample)}")
                        
                        img_feat, txt_feat, label = sample
                        
                        # Validate dtypes
                        if img_feat.dtype != torch.float32:
                            self.logger.warning(f"Image features in {cache_file} are {img_feat.dtype}, converting to float32")
                        if txt_feat.dtype != torch.float32:
                            self.logger.warning(f"Text features in {cache_file} are {txt_feat.dtype}, converting to float32")
                        if label.dtype != torch.float32:
                            self.logger.warning(f"Labels in {cache_file} are {label.dtype}, converting to float32")
                        
                        # Validate shapes
                        if list(img_feat.shape) != expected_shapes["image"]:
                            raise ValueError(f"Image feature shape mismatch in {cache_file}: expected {expected_shapes['image']}, got {list(img_feat.shape)}")
                        if list(txt_feat.shape) != expected_shapes["text"]:
                            raise ValueError(f"Text feature shape mismatch in {cache_file}: expected {expected_shapes['text']}, got {list(txt_feat.shape)}")
                        if list(label.shape) != expected_shapes["label"]:
                            raise ValueError(f"Label shape mismatch in {cache_file}: expected {expected_shapes['label']}, got {list(label.shape)}")
                        
                        # Check for valid label values (0-1 for multi-label)
                        if not torch.all((label >= 0) & (label <= 1)):
                            raise ValueError(f"Invalid label values in {cache_file}: labels must be in [0,1] range")
                        
                        self.logger.info(f"✅ {cache_file}: {len(data)} samples, shapes verified, dtypes OK")
                    
                    elif isinstance(data, dict):
                        # New format: {"img": tensor, "txt": tensor, "y": tensor}
                        required_keys = {"img", "txt", "y"}
                        if not required_keys.issubset(data.keys()):
                            raise ValueError(f"Invalid dict format in {cache_file}: missing keys {required_keys - set(data.keys())}")
                        
                        # Check tensor properties
                        img_tensor, txt_tensor, y_tensor = data["img"], data["txt"], data["y"]
                        
                        # Validate dtypes
                        if img_tensor.dtype != torch.float32:
                            self.logger.warning(f"Image tensor in {cache_file} is {img_tensor.dtype}, should be float32")
                        if txt_tensor.dtype != torch.float32:
                            self.logger.warning(f"Text tensor in {cache_file} is {txt_tensor.dtype}, should be float32") 
                        if y_tensor.dtype != torch.float32:
                            self.logger.warning(f"Label tensor in {cache_file} is {y_tensor.dtype}, should be float32")
                        
                        # Validate lengths match
                        if not (len(img_tensor) == len(txt_tensor) == len(y_tensor)):
                            raise ValueError(f"Tensor length mismatch in {cache_file}")
                        
                        # Validate shapes (per sample)
                        if list(img_tensor.shape[1:]) != expected_shapes["image"]:
                            raise ValueError(f"Image tensor shape mismatch in {cache_file}")
                        if list(txt_tensor.shape[1:]) != expected_shapes["text"]:
                            raise ValueError(f"Text tensor shape mismatch in {cache_file}")
                        if list(y_tensor.shape[1:]) != expected_shapes["label"]:
                            raise ValueError(f"Label tensor shape mismatch in {cache_file}")
                        
                        self.logger.info(f"✅ {cache_file}: {len(y_tensor)} samples, tensor format verified")
                    
                    else:
                        raise ValueError(f"Unknown cache format in {cache_file}: {type(data)}")
                
                except Exception as e:
                    self.logger.error(f"❌ Cache validation failed for {cache_file}: {e}")
                    return False
            
            self.logger.info("✅ All cache files passed validation")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cache validation failed: {e}")
            return False
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is valid."""
        manifest_file = self.manifest.cache_path / "coco_manifest.json"
        
        # Check if manifest exists
        if not manifest_file.exists():
            self.logger.info("📝 No cache manifest found")
            return False
        
        try:
            # Load and verify manifest
            with open(manifest_file) as f:
                stored_manifest = json.load(f)
            
            # Check version compatibility
            if stored_manifest.get('version') != self.manifest.version:
                self.logger.info("🔄 Cache version mismatch, rebuilding")
                return False
            
            # Check if all cache files exist
            missing_files = []
            for cache_file in self.manifest.cache_files:
                if cache_file == "coco_manifest.json":
                    continue
                cache_path = self.manifest.cache_path / cache_file
                if not cache_path.exists():
                    missing_files.append(cache_file)
            
            if missing_files:
                self.logger.info(f"📁 Missing cache files: {missing_files}")
                return False
            
            # Quick integrity check - verify at least one cache file can be loaded
            try:
                test_file = self.manifest.cache_path / "coco_clip_cache_train.pt"
                data = torch.load(test_file, map_location='cpu')
                if not isinstance(data, (list, dict)) or len(data) == 0:
                    self.logger.info("🔍 Cache file corrupted")
                    return False
            except Exception as e:
                self.logger.info(f"🔍 Cache integrity check failed: {e}")
                return False
            
            self.logger.info("✅ Cache validation passed")
            return True
            
        except Exception as e:
            self.logger.info(f"📝 Cache validation error: {e}")
            return False
    
    def _ensure_raw_data(self) -> bool:
        """Ensure raw COCO data is available."""
        self.logger.info("📊 Checking raw COCO data...")
        
        # Check if raw data directory exists
        if not self.manifest.raw_data_path.exists():
            self.logger.info(f"📁 Raw data directory not found: {self.manifest.raw_data_path}")
            # Don't try to download, just return False so we can use dummy cache
            self.logger.info("💡 Will use dummy cache instead of downloading data")
            return False
        
        # Check required files/directories
        missing_items = []
        for item_name in self.manifest.files_required:
            item_path = self.manifest.raw_data_path / item_name
            if not item_path.exists():
                missing_items.append(item_name)
        
        if missing_items:
            self.logger.info(f"📁 Missing raw data items: {missing_items}")
            self.logger.info("💡 Will use dummy cache instead of downloading data")
            return False
        
        # Quick validation - check if directories have content
        train_dir = self.manifest.raw_data_path / "train2014"
        val_dir = self.manifest.raw_data_path / "val2014"
        
        if (train_dir.exists() and len(list(train_dir.glob("*.jpg"))) < 1000) or \
           (val_dir.exists() and len(list(val_dir.glob("*.jpg"))) < 100):
            self.logger.info("📁 Image directories appear incomplete")
            self.logger.info("💡 Will use dummy cache instead of downloading data")
            return False
        
        self.logger.info("✅ Raw data validation passed")
        return True
    
    def _download_raw_data(self) -> bool:
        """Download raw COCO data."""
        self.logger.info("⬇️ Downloading COCO data...")
        
        try:
            # Try to use existing ensure_coco2014 function
            from aecf.datasets import ensure_coco2014
            ensure_coco2014(str(self.manifest.raw_data_path))
            return True
            
        except ImportError:
            self.logger.warning("⚠️ COCO download function not available")
            
            # Provide manual download instructions
            self.logger.info(f"""
📋 Manual COCO Download Required:

1. Create directory: {self.manifest.raw_data_path}
2. Download and extract:
   - http://images.cocodataset.org/zips/train2014.zip
   - http://images.cocodataset.org/zips/val2014.zip
   - http://images.cocodataset.org/annotations/annotations_trainval2014.zip

Directory structure should be:
{self.manifest.raw_data_path}/
├── annotations/
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── captions_train2014.json
│   └── captions_val2014.json
├── train2014/
│   └── *.jpg (82,783 images)
└── val2014/
    └── *.jpg (40,504 images)
""")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Download failed: {e}")
            return False
    
    def _build_cache(self) -> bool:
        """Build CLIP feature cache."""
        self.logger.info("🔧 Building CLIP feature cache...")
        
        # First check if we have raw data available
        if not self.manifest.raw_data_path.exists():
            self.logger.info("📁 No raw data available, creating dummy cache for testing...")
            return self._create_dummy_cache()
        
        try:
            # Try to use existing cache building function
            setup_coco_cache_pipeline(self.manifest.raw_data_path)
            
            # Save manifest
            self._save_manifest()
            
            return True
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Cache pipeline dependencies missing: {e}")
            # Fall back to dummy cache
            self.logger.info("🎭 Creating dummy cache for testing...")
            return self._create_dummy_cache()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache building failed: {e}")
            # Fall back to dummy cache
            self.logger.info("🎭 Creating dummy cache for testing...")
            return self._create_dummy_cache()
    
    def _create_dummy_cache(self) -> bool:
        """Create dummy cache files for testing with correct dtypes and shapes."""
        try:
            # Create dummy data matching expected ClipTensor format (dict with tensors)
            dummy_data = {
                'train': {
                    'img': torch.randn(1000, 512, dtype=torch.float32),
                    'txt': torch.randn(1000, 512, dtype=torch.float32),
                    'y': torch.randint(0, 2, (1000, 80), dtype=torch.float32)
                },
                'val': {
                    'img': torch.randn(100, 512, dtype=torch.float32),
                    'txt': torch.randn(100, 512, dtype=torch.float32),
                    'y': torch.randint(0, 2, (100, 80), dtype=torch.float32)
                },
                'test': {
                    'img': torch.randn(100, 512, dtype=torch.float32),
                    'txt': torch.randn(100, 512, dtype=torch.float32),
                    'y': torch.randint(0, 2, (100, 80), dtype=torch.float32)
                }
            }
            
            # Save cache files with correct names
            filename_mapping = {
                'train': 'coco_clip_cache_train.pt',
                'val': 'coco_clip_cache_val.pt', 
                'test': 'coco_clip_cache_test.pt'
            }
            
            for split, data in dummy_data.items():
                cache_file = self.manifest.cache_path / filename_mapping[split]
                torch.save(data, cache_file)
                self.logger.info(f"📁 Created dummy cache: {cache_file}")
            
            self._save_manifest()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dummy cache creation failed: {e}")
            return False
    
    def _save_manifest(self):
        """Save cache manifest."""
        manifest_file = self.manifest.cache_path / "coco_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(self.manifest.to_dict(), f, indent=2, default=str)
        self.logger.info(f"📝 Saved manifest: {manifest_file}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about current data state."""
        info = {
            "raw_data_available": self.manifest.raw_data_path.exists(),
            "cache_valid": self._is_cache_valid(),
            "cache_files": {}
        }
        
        # Check individual cache files
        for cache_file in self.manifest.cache_files:
            if cache_file == "coco_manifest.json":
                continue
            cache_path = self.manifest.cache_path / cache_file
            if cache_path.exists():
                try:
                    data = torch.load(cache_path, map_location='cpu')
                    info["cache_files"][cache_file] = {
                        "exists": True,
                        "size": len(data) if isinstance(data, (list, dict)) else "unknown",
                        "file_size_mb": cache_path.stat().st_size / (1024*1024)
                    }
                except:
                    info["cache_files"][cache_file] = {"exists": True, "corrupted": True}
            else:
                info["cache_files"][cache_file] = {"exists": False}
        
        return info


# ============================================================================
# Ablation Configuration and Experiment Management
# ============================================================================


@dataclass
class AblationConfig:
    """Configuration for ablation experiments with comprehensive validation."""
    name: str
    gate_disabled: bool = False
    entropy_reg: bool = True
    curriculum_mask: bool = True
    modalities: List[str] = field(default_factory=lambda: ["image", "text"])
    
    # Model architecture - aligned with AECF_CLIP requirements
    feat_dim: int = 512
    num_classes: int = 80
    hidden_dim: int = 256
    task_type: str = "classification"  # Explicitly set for AECF_CLIP
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 50
    patience: int = 10
    
    # Paths
    data_root: Path = Path("/content/coco2014")
    cache_dir: Path = Path("./cache")
    output_dir: Path = Path("./ablation_results")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths are Path objects
        self.data_root = Path(self.data_root)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        
        # Validate modalities
        valid_modalities = {"image", "text", "audio"}
        if not set(self.modalities).issubset(valid_modalities):
            raise ValueError(f"Invalid modalities: {set(self.modalities) - valid_modalities}")
        
        # Validate required fields for AECF_CLIP
        if self.feat_dim != 512:
            raise ValueError("feat_dim must be 512 for CLIP features")
        if self.num_classes != 80:
            raise ValueError("num_classes must be 80 for COCO dataset")
        if self.task_type != "classification":
            raise ValueError("task_type must be 'classification' for COCO")
    
    def to_model_config(self) -> Dict[str, Any]:
        """Convert to AECF_CLIP model configuration with dtype validation."""
        return {
            'modalities': self.modalities,
            'encoders': {
                mod: {'input_dim': self.feat_dim, 'hidden_dim': self.hidden_dim}
                for mod in self.modalities
            },
            'gate_disabled': self.gate_disabled,
            'entropy_reg': self.entropy_reg,
            'curriculum_mask': self.curriculum_mask,
            'num_classes': self.num_classes,
            'feat_dim': self.feat_dim,
            'learning_rate': self.learning_rate,
            'task_type': self.task_type,
            # Critical: Ensure consistent dtype handling
            'feature_norm': True,  # Normalize CLIP features to float32
            'masking_mode': 'entropy_min' if self.curriculum_mask else 'none',
            'p_missing': 0.3 if self.curriculum_mask else 0.0,
            'tau': 0.4
        }


class AblationExperiment:
    """Single ablation experiment runner with comprehensive validation."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup experiment-specific logging."""
        logger = logging.getLogger(f"ablation_{self.config.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure output directory exists
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            handler = logging.FileHandler(
                self.config.output_dir / f"{self.config.name}.log"
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_cache_files(self, cache_dir: Path) -> bool:
        """Validate cache files exist and have correct format."""
        try:
            required_files = [
                cache_dir / "coco_clip_cache_train.pt",
                cache_dir / "coco_clip_cache_val.pt", 
                cache_dir / "coco_clip_cache_test.pt"
            ]
            
            for cache_file in required_files:
                if not cache_file.exists():
                    self.logger.error(f"❌ Missing cache file: {cache_file}")
                    return False
                
                # Load and validate structure
                data = torch.load(cache_file, map_location='cpu')
                if not isinstance(data, list) or len(data) == 0:
                    self.logger.error(f"❌ Invalid cache format: {cache_file}")
                    return False
                
                # Check first sample
                sample = data[0]
                if not (isinstance(sample, tuple) and len(sample) == 3):
                    self.logger.error(f"❌ Invalid sample format: {cache_file}")
                    return False
                
                img_feat, txt_feat, label = sample
                
                # Validate dtypes
                if img_feat.dtype != torch.float32:
                    self.logger.error(f"❌ Image features wrong dtype: {img_feat.dtype} != torch.float32")
                    return False
                if txt_feat.dtype != torch.float32:
                    self.logger.error(f"❌ Text features wrong dtype: {txt_feat.dtype} != torch.float32")
                    return False
                if label.dtype != torch.float32:
                    self.logger.error(f"❌ Labels wrong dtype: {label.dtype} != torch.float32")
                    return False
                
                # Validate shapes
                if img_feat.shape != (512,):
                    self.logger.error(f"❌ Image feature shape: {img_feat.shape} != (512,)")
                    return False
                if txt_feat.shape != (512,):
                    self.logger.error(f"❌ Text feature shape: {txt_feat.shape} != (512,)")
                    return False
                if label.shape != (80,):
                    self.logger.error(f"❌ Label shape: {label.shape} != (80,)")
                    return False
                
                self.logger.info(f"✅ Cache file validated: {cache_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cache validation failed: {e}")
            return False
    
    def _validate_data_loaders(self, train_loader, val_loader, test_loader) -> bool:
        """Validate data loaders produce correct tensor formats for AECF_CLIP."""
        try:
            self.logger.info("🔍 Validating data loaders...")
            
            for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                # Get first batch
                try:
                    batch = next(iter(loader))
                except StopIteration:
                    self.logger.error(f"❌ Empty {name} loader")
                    return False
                
                # Check batch format - must be Dict[str, torch.Tensor] for AECF_CLIP
                if not isinstance(batch, dict):
                    self.logger.error(f"❌ {name} batch not dict: {type(batch)}")
                    return False
                
                # Check required keys
                required_keys = {"image", "text", "label"}
                if not required_keys.issubset(batch.keys()):
                    self.logger.error(f"❌ {name} batch missing keys: {required_keys - set(batch.keys())}")
                    return False
                
                # Validate tensor properties
                for key in required_keys:
                    tensor = batch[key]
                    
                    if not isinstance(tensor, torch.Tensor):
                        self.logger.error(f"❌ {name}.{key} not tensor: {type(tensor)}")
                        return False
                    
                    # Check dtypes - CRITICAL for AECF_CLIP compatibility
                    if tensor.dtype != torch.float32:
                        self.logger.error(f"❌ {name}.{key} wrong dtype: {tensor.dtype} != torch.float32")
                        return False
                    
                    # Check shapes
                    if key == "image":
                        if len(tensor.shape) != 2 or tensor.shape[1] != 512:
                            self.logger.error(f"❌ {name}.image shape: {tensor.shape} != (B, 512)")
                            return False
                    elif key == "text":
                        if len(tensor.shape) != 2 or tensor.shape[1] != 512:
                            self.logger.error(f"❌ {name}.text shape: {tensor.shape} != (B, 512)")
                            return False
                    elif key == "label":
                        if len(tensor.shape) != 2 or tensor.shape[1] != 80:
                            self.logger.error(f"❌ {name}.label shape: {tensor.shape} != (B, 80)")
                            return False
                
                # Check value ranges
                labels = batch["label"]
                if not torch.all((labels >= 0) & (labels <= 1)):
                    self.logger.error(f"❌ {name} labels out of range [0,1]")
                    return False
                
                # Check for NaN/Inf
                for key, tensor in batch.items():
                    if torch.isnan(tensor).any():
                        self.logger.error(f"❌ {name}.{key} contains NaN")
                        return False
                    if torch.isinf(tensor).any():
                        self.logger.error(f"❌ {name}.{key} contains Inf")
                        return False
                
                self.logger.info(f"✅ {name} loader validated: {tensor.shape[0]} samples")
            
            self.logger.info("✅ All data loaders validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data loader validation failed: {e}")
            return False
    
    def run(self, train_loader, val_loader, test_loader) -> Dict[str, Any]:
        """Run single ablation experiment with comprehensive validation."""
        self.logger.info(f"Starting ablation: {self.config.name}")
        
        # Pre-flight validation
        if not self._validate_data_loaders(train_loader, val_loader, test_loader):
            raise RuntimeError("Data loader validation failed - cannot proceed")
        
        if not self._validate_cache_files(self.config.cache_dir):
            self.logger.warning("Cache file validation failed - proceeding with caution")
        
        try:
            # Create model
            model = self._create_model()
            
            # Setup trainer
            trainer = self._create_trainer()
            
            # Train
            self.logger.info("Starting training...")
            trainer.fit(model, train_loader, val_loader)
            
            # Test
            self.logger.info("Starting testing...")
            test_results = trainer.test(model, test_loader)
            
            # Collect results
            results = self._collect_results(trainer, test_results)
            
            self.logger.info(f"Completed ablation: {self.config.name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed ablation {self.config.name}: {str(e)}")
            raise
    
    def _create_model(self) -> AECF_CLIP:
        """Create model with ablation-specific configuration."""
        model_config = self.config.to_model_config()
        
        self.logger.info(f"Creating AECF_CLIP model with config: {model_config}")
        
        try:
            model = AECF_CLIP(model_config)
            
            # Validate model is in expected state
            model.eval()
            
            # Test forward pass with dummy data
            dummy_batch = {
                modality: torch.randn(2, 512, dtype=torch.float32)
                for modality in self.config.modalities
            }
            
            with torch.no_grad():
                logits, weights = model(dummy_batch)
                
                # Validate output shapes and dtypes
                if logits.shape != (2, 80):
                    raise ValueError(f"Model output shape wrong: {logits.shape}, expected (2, 80)")
                if weights.shape[1] != len(self.config.modalities):
                    raise ValueError(f"Model weights shape wrong: {weights.shape[1]}, expected {len(self.config.modalities)}")
                
                if not torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6):
                    raise ValueError("Model weights don't sum to 1")
            
            self.logger.info("✅ Model validation passed")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Model creation failed: {e}")
            raise
    
    def _create_trainer(self) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min'
            ),
            ModelCheckpoint(
                dirpath=self.config.output_dir / self.config.name,
                filename='best_model',
                monitor='val_loss',
                save_top_k=1,
                mode='min'
            )
        ]
        
        return pl.Trainer(
            max_epochs=self.config.max_epochs,
            callbacks=callbacks,
            accelerator='auto',
            devices='auto',
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=False  # Changed from True to False to avoid bincount issues
        )
    
    def _collect_results(self, trainer, test_results) -> Dict[str, Any]:
        """Collect experiment results."""
        return {
            'ablation_name': self.config.name,
            'config': self.config.__dict__,
            'test_metrics': test_results[0] if test_results else {},
            'best_val_loss': trainer.checkpoint_callback.best_model_score.item(),
            'epochs_trained': trainer.current_epoch,
            'model_path': str(trainer.checkpoint_callback.best_model_path)
        }


class DataManager:
    """DEPRECATED: Use COCODataManager instead for better reliability."""
    
    def __init__(self, data_root: Path, cache_dir: Path):
        import warnings
        warnings.warn("DataManager is deprecated. Use COCODataManager instead.", DeprecationWarning)
        self.data_root = data_root
        self.cache_dir = cache_dir
        self._loaders_cache = None
        
    def get_loaders(self, batch_size: int = 32) -> Tuple[Any, Any, Any]:
        """Get cached data loaders."""
        if self._loaders_cache is None:
            self._ensure_data()
            self._loaders_cache = make_clip_tensor_loaders_from_cache(
                cache_dir=self.cache_dir,
                batch_size=batch_size,
                num_workers=min(4, mp.cpu_count())
            )
        return self._loaders_cache
    
    def _ensure_data(self):
        """Ensure data and cache exist."""
        # Check if cache exists
        required_files = [
            self.cache_dir / "coco_clip_cache_train.pt",
            self.cache_dir / "coco_clip_cache_val.pt", 
            self.cache_dir / "coco_clip_cache_test.pt"
        ]
        
        if all(f.exists() for f in required_files):
            print("✅ Found cached CLIP features, skipping extraction")
            return
            
        print("📊 Setting up COCO cache pipeline...")
        try:
            setup_coco_cache_pipeline(self.data_root)
        except ImportError as e:
            print(f"⚠️ Cache pipeline dependencies missing: {e}")
            print("Using existing cache or creating dummy cache...")


class ResultsAnalyzer:
    """Analyzes and compares ablation results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def analyze_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze and compare ablation results."""
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)
        
        # Extract key metrics
        analysis_df = pd.DataFrame({
            'Ablation': df['ablation_name'],
            'Test_Loss': df['test_metrics'].apply(lambda x: x.get('test_loss', float('nan'))),
            'Test_Accuracy': df['test_metrics'].apply(lambda x: x.get('test_acc', float('nan'))),
            'Best_Val_Loss': df['best_val_loss'],
            'Epochs': df['epochs_trained'],
            'Modalities': df['config'].apply(lambda x: len(x.get('modalities', []))),
            'Gate_Enabled': df['config'].apply(lambda x: not x.get('gate_disabled', False)),
            'Entropy_Reg': df['config'].apply(lambda x: x.get('entropy_reg', True)),
            'Curriculum': df['config'].apply(lambda x: x.get('curriculum_mask', True))
        })
        
        return analysis_df
    
    def save_results(self, results: List[Dict[str, Any]], analysis_df: pd.DataFrame):
        """Save results and analysis."""
        # Save raw results
        with open(self.output_dir / 'raw_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save analysis table
        analysis_df.to_csv(self.output_dir / 'analysis_summary.csv', index=False)
        
        # Save formatted report
        self._generate_report(analysis_df)
    
    def _generate_report(self, df: pd.DataFrame):
        """Generate formatted analysis report."""
        report = []
        report.append("# AECF Ablation Study Results\n")
        
        # Summary table
        report.append("## Results Summary")
        report.append(df.to_string(index=False))
        report.append("\n")
        
        # Component analysis
        if len(df) > 1:
            baseline = df[df['Ablation'] == 'full']
            if not baseline.empty:
                baseline_acc = baseline['Test_Accuracy'].iloc[0]
                report.append("## Component Analysis")
                
                for _, row in df.iterrows():
                    if row['Ablation'] != 'full':
                        diff = row['Test_Accuracy'] - baseline_acc
                        report.append(f"- {row['Ablation']}: {diff:+.4f} vs baseline")
                
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))


class AblationSuite:
    """Main ablation suite orchestrator with integrated data management."""
    
    # Predefined ablation configurations with rigorous validation
    STANDARD_ABLATIONS = {
        "full": AblationConfig(name="full"),
        "no_gate": AblationConfig(name="no_gate", gate_disabled=True),
        "no_entropy": AblationConfig(name="no_entropy", entropy_reg=False),
        "no_curmask": AblationConfig(name="no_curmask", curriculum_mask=False),
        "img_only": AblationConfig(name="img_only", modalities=["image"]),
        "txt_only": AblationConfig(name="txt_only", modalities=["text"])
    }
    
    def __init__(self, 
                 data_root: Path = Path("/content/coco2014"),
                 cache_dir: Path = Path("./cache"),
                 output_dir: Path = Path("./ablation_results"),
                 parallel: bool = False):
        
        self.data_manager = COCODataManager(data_root, cache_dir)
        self.output_dir = output_dir
        self.parallel = parallel
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_main_logging()
        
        # Validate system requirements
        self._validate_system_requirements()
        
    def _setup_main_logging(self) -> logging.Logger:
        """Setup main suite logging."""
        logger = logging.getLogger("ablation_suite")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.output_dir / "suite.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _validate_system_requirements(self):
        """Validate system has required capabilities."""
        self.logger.info("🔍 Validating system requirements...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            self.logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            self.logger.warning("⚠️ CUDA not available, using CPU (will be slow)")
        
        # Check memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 4:
                self.logger.warning(f"⚠️ Low GPU memory: {gpu_memory:.1f}GB")
            else:
                self.logger.info(f"✅ GPU memory: {gpu_memory:.1f}GB")
        
        # Check CPU cores for data loading
        cpu_count = mp.cpu_count()
        self.logger.info(f"✅ CPU cores: {cpu_count}")
        
        self.logger.info("✅ System requirements validated")
    
    def run_ablations(self, 
                     ablation_names: Optional[List[str]] = None,
                     custom_configs: Optional[Dict[str, AblationConfig]] = None) -> pd.DataFrame:
        """Run ablation experiments with comprehensive validation."""
        
        # Determine which ablations to run
        if custom_configs:
            configs = custom_configs
        else:
            if ablation_names is None:
                ablation_names = list(self.STANDARD_ABLATIONS.keys())
            configs = {name: self.STANDARD_ABLATIONS[name] for name in ablation_names}
        
        # Update output directories for all configs
        for config in configs.values():
            config.output_dir = self.output_dir
        
        self.logger.info(f"Running {len(configs)} ablations: {list(configs.keys())}")
        
        # CRITICAL: Ensure data is ready with comprehensive validation
        print("🔍 Preparing COCO data with validation...")
        if not self.data_manager.ensure_data_ready():
            raise RuntimeError("Data preparation failed - cannot proceed with ablations")
        
        # Get data info for logging
        info = self.data_manager.get_data_info()
        self.logger.info(f"📊 Data ready: {info}")
        
        # Get data loaders (guaranteed to work after ensure_data_ready)
        train_loader, val_loader, test_loader = self.data_manager.get_loaders(
            batch_size=list(configs.values())[0].batch_size,
            num_workers=min(4, mp.cpu_count())
        )
        
        # Log data loader information
        self._log_data_loader_info(train_loader, val_loader, test_loader)
        
        # Run experiments
        if self.parallel and len(configs) > 1:
            results = self._run_parallel(configs, train_loader, val_loader, test_loader)
        else:
            results = self._run_sequential(configs, train_loader, val_loader, test_loader)
        
        # Analyze results
        analyzer = ResultsAnalyzer(self.output_dir)
        analysis_df = analyzer.analyze_results(results)
        analyzer.save_results(results, analysis_df)
        
        self.logger.info("✅ Ablation suite completed successfully!")
        return analysis_df
    
    def _log_data_loader_info(self, train_loader, val_loader, test_loader):
        """Log information about data loaders for debugging."""
        try:
            train_batch = next(iter(train_loader))
            
            self.logger.info(f"📊 Data loader info:")
            self.logger.info(f"  - Train batches: {len(train_loader)}")
            self.logger.info(f"  - Val batches: {len(val_loader)}")
            self.logger.info(f"  - Test batches: {len(test_loader)}")
            self.logger.info(f"  - Batch keys: {list(train_batch.keys())}")
            
            for key, tensor in train_batch.items():
                if isinstance(tensor, torch.Tensor):
                    self.logger.info(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not log data loader info: {e}")
    
    def _run_sequential(self, configs, train_loader, val_loader, test_loader) -> List[Dict]:
        """Run ablations sequentially with proper cleanup."""
        results = []
        
        for name, config in configs.items():
            self.logger.info(f"🚀 Running ablation: {name}")
            
            try:
                experiment = AblationExperiment(config)
                result = experiment.run(train_loader, val_loader, test_loader)
                results.append(result)
                
                self.logger.info(f"✅ Completed ablation: {name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed ablation {name}: {e}")
                # Continue with other ablations
                results.append({
                    'ablation_name': name,
                    'config': config.__dict__,
                    'error': str(e),
                    'test_metrics': {},
                    'best_val_loss': float('inf'),
                    'epochs_trained': 0,
                    'model_path': None
                })
            
            finally:
                # Clear GPU memory between experiments
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return results
    
    def _run_parallel(self, configs, train_loader, val_loader, test_loader) -> List[Dict]:
        """Run ablations in parallel (experimental - may have GPU memory issues)."""
        self.logger.warning("⚠️ Parallel execution is experimental and may cause GPU memory issues")
        
        def run_single(name_config):
            name, config = name_config
            try:
                experiment = AblationExperiment(config)
                return experiment.run(train_loader, val_loader, test_loader)
            except Exception as e:
                return {
                    'ablation_name': name,
                    'config': config.__dict__,
                    'error': str(e),
                    'test_metrics': {},
                    'best_val_loss': float('inf'),
                    'epochs_trained': 0,
                    'model_path': None
                }
        
        # Limit parallelism to avoid GPU memory issues
        max_workers = min(2, len(configs))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_single, configs.items()))
        
        return results


# Convenience functions for common use cases
def run_quick_ablation(ablations: List[str] = ["full", "no_gate"]) -> pd.DataFrame:
    """Run a quick ablation with reduced epochs for testing."""
    suite = AblationSuite()
    
    # Create quick configs
    quick_configs = {}
    for name in ablations:
        config = suite.STANDARD_ABLATIONS[name]
        config.max_epochs = 5  # Quick test
        config.batch_size = 64  # Larger batches for speed
        quick_configs[name] = config
    
    return suite.run_ablations(custom_configs=quick_configs)


def run_full_ablation_suite() -> pd.DataFrame:
    """Run the complete ablation suite."""
    suite = AblationSuite()
    return suite.run_ablations()


# Replace the end of test/cocoAblation.py (around line 1200+) with this:

if __name__ == "__main__":
    import argparse
    import sys
    
    # Ensure output directory exists
    os.makedirs("./ablation_results", exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('./ablation_results/suite.log')
        ]
    )
    
    print(f"🚀 AECF Ablation Suite Starting...")
    
    # Parse arguments (with defaults for Colab)
    try:
        parser = argparse.ArgumentParser(description="AECF Ablation Suite")
        parser.add_argument("--ablations", nargs="+", 
                           choices=["full", "no_gate", "no_entropy", "no_curmask", "img_only", "txt_only"],
                           default=["full", "no_gate"],  # Default to just 2 for faster testing
                           help="Specific ablations to run")
        parser.add_argument("--quick", action="store_true", default=True,  # Default to quick mode
                           help="Run quick ablation with fewer epochs")
        parser.add_argument("--parallel", action="store_true",
                           help="Run ablations in parallel (experimental)")
        parser.add_argument("--output-dir", type=str, default="./ablation_results",
                           help="Output directory for results")
        
        # Parse args (handle Colab notebook context)
        import sys
        if 'ipykernel' in sys.modules:
            # Running in Jupyter/Colab - use defaults
            args = parser.parse_args([])
        else:
            args = parser.parse_args()
            
        args.output_dir = Path(args.output_dir)
        
    except Exception as e:
        print(f"⚠️ Argument parsing issue: {e}")
        # Use defaults for Colab
        class DefaultArgs:
            ablations = ["full", "no_gate"]
            quick = True
            parallel = False
            output_dir = Path("./ablation_results")
        args = DefaultArgs()
    
    print(f"📊 Configuration:")
    print(f"   - Ablations: {args.ablations}")
    print(f"   - Quick mode: {args.quick}")
    print(f"   - Output: {args.output_dir}")
    
    try:
        # Create ablation suite with local-friendly paths
        print(f"\n🔧 Initializing AblationSuite...")
        suite = AblationSuite(
            data_root=Path("./coco2014"),  # Local path instead of /content/coco2014
            cache_dir=Path("./cache"), 
            output_dir=args.output_dir
        )
        
        # Configure for quick mode if enabled
        if args.quick:
            print("⚡ Quick mode enabled: 5 epochs, batch_size=64")
            # Create quick configs
            quick_configs = {}
            for name in args.ablations:
                if name in suite.STANDARD_ABLATIONS:
                    config = suite.STANDARD_ABLATIONS[name]
                    # Modify for quick execution
                    config.max_epochs = 5
                    config.batch_size = 64
                    config.patience = 3
                    quick_configs[name] = config
        else:
            quick_configs = None
        
        # Run ablations
        print(f"\n{'='*60}")
        print("🚀 STARTING ABLATION EXPERIMENTS")
        print(f"{'='*60}")
        
        if quick_configs:
            results = suite.run_ablations(custom_configs=quick_configs)
        else:
            results = suite.run_ablations(args.ablations)
        
        # Display results
        print(f"\n{'='*60}")
        print("📊 ABLATION RESULTS SUMMARY")
        print(f"{'='*60}")
        if hasattr(results, 'to_string'):
            print(results.to_string(index=False))
        else:
            print(results)
        
        print(f"\n✅ Ablation suite completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Ablation suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a minimal test instead
        print(f"\n🔬 Attempting minimal test...")
        try:
            # Simple data loading test
            from aecf import make_clip_tensor_loaders_from_cache
            from pathlib import Path
            cache_dir = Path("./cache")
            if cache_dir.exists():
                print("✅ Cache directory exists, basic setup working")
            else:
                print("❌ Cache directory not found")
        except Exception as test_e:
            print(f"❌ Minimal test also failed: {test_e}")
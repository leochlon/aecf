"""
Single ablation experiment execution with validation.
"""
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from typing import Dict, Any

from .config import AblationConfig
from test.ui.progress import CleanProgressBar
from test.utils.constants import GPU_OPTIMIZATIONS


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
                
                # Handle different cache formats
                if isinstance(data, dict):
                    # New dict format: {"img": tensor, "txt": tensor, "y": tensor}
                    if not all(key in data for key in ['img', 'txt', 'y']):
                        self.logger.error(f"❌ Invalid dict format: {cache_file}")
                        return False
                    
                    # Convert bfloat16 to float32 if needed and optimize memory layout
                    data_changed = False
                    for key in ['img', 'txt', 'y']:
                        if data[key].dtype == torch.bfloat16:
                            self.logger.info(f"🔄 Converting {key} from bfloat16 to float32 in {cache_file}")
                            # Convert and make contiguous for better performance
                            data[key] = data[key].to(torch.float32).contiguous()
                            data_changed = True
                        elif not data[key].is_contiguous():
                            # Ensure memory layout is optimal
                            data[key] = data[key].contiguous()
                            data_changed = True
                    
                    # Save the corrected version only if changes were made
                    if data_changed:
                        torch.save(data, cache_file)
                    
                elif isinstance(data, list):
                    # Old list format: [(img_feat, txt_feat, label), ...]
                    if len(data) == 0:
                        self.logger.error(f"❌ Empty cache file: {cache_file}")
                        return False
                    
                    # Convert to float32 if needed and optimize for performance
                    data_changed = False
                    corrected_data = []
                    for img_feat, txt_feat, label in data:
                        # Convert dtypes if needed
                        if img_feat.dtype == torch.bfloat16:
                            img_feat = img_feat.to(torch.float32).contiguous()
                            data_changed = True
                        elif not img_feat.is_contiguous():
                            img_feat = img_feat.contiguous()
                            data_changed = True
                            
                        if txt_feat.dtype == torch.bfloat16:
                            txt_feat = txt_feat.to(torch.float32).contiguous()
                            data_changed = True
                        elif not txt_feat.is_contiguous():
                            txt_feat = txt_feat.contiguous()
                            data_changed = True
                            
                        if label.dtype == torch.bfloat16:
                            label = label.to(torch.float32).contiguous()
                            data_changed = True
                        elif not label.is_contiguous():
                            label = label.contiguous()
                            data_changed = True
                            
                        corrected_data.append((img_feat, txt_feat, label))
                    
                    # Save corrected version only if changes were made
                    if data_changed:
                        self.logger.info(f"🔄 Converting cache file to float32 and optimizing layout: {cache_file}")
                        torch.save(corrected_data, cache_file)
                else:
                    self.logger.error(f"❌ Unknown cache format: {cache_file}")
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
                    
                    # Check dtypes - accept both float32 and bfloat16 for AECF_CLIP compatibility
                    valid_dtypes = {torch.float32, torch.bfloat16}
                    if tensor.dtype not in valid_dtypes:
                        self.logger.error(f"❌ {name}.{key} wrong dtype: {tensor.dtype} not in {valid_dtypes}")
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
    
    def _create_model(self):
        """Create model with ablation-specific configuration."""
        from aecf import AECF_CLIP
        
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
        """Create PyTorch Lightning trainer with optimized GPU utilization."""
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
            ),
            CleanProgressBar()  # Use our custom progress bar
        ]
        
        # Set tensor core precision for better GPU utilization
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision(GPU_OPTIMIZATIONS['float32_matmul_precision'])
            
            # Additional GPU optimizations
            torch.backends.cudnn.benchmark = GPU_OPTIMIZATIONS['benchmark']
            torch.backends.cuda.matmul.allow_tf32 = GPU_OPTIMIZATIONS['allow_tf32']
        
        trainer_kwargs = {
            'max_epochs': self.config.max_epochs,
            'callbacks': callbacks,
            'accelerator': 'auto',
            'devices': 'auto',
            'log_every_n_steps': 500,  # Significantly reduced logging frequency
            'enable_progress_bar': True,  # Keep epoch-level TQDM progress bar
            'enable_model_summary': False,
            'deterministic': False,  # Allow non-deterministic ops for better performance
            'num_sanity_val_steps': 0,  # Skip sanity validation to reduce output
            'sync_batchnorm': False,  # Disable for single GPU
            'logger': False,  # Disable default logging to reduce output
            'enable_checkpointing': True,
        }
        
        # Add GPU-specific optimizations for A100
        if torch.cuda.is_available():
            trainer_kwargs.update({
                'precision': GPU_OPTIMIZATIONS['precision'],
                'accumulate_grad_batches': GPU_OPTIMIZATIONS['accumulate_grad_batches'],
                'strategy': GPU_OPTIMIZATIONS['strategy'],
                'benchmark': GPU_OPTIMIZATIONS['benchmark'],
            })
        else:
            trainer_kwargs['precision'] = 32
        
        return pl.Trainer(**trainer_kwargs)
    
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

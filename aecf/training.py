"""
Training utilities and components for AECF model.

This module contains training-specific components separated from the core model
for better modularity and maintainability.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .config import AECFConfig
from .model_refactored import AECF_CLIP


class AECFTrainer:
    """
    Production-ready trainer for AECF models with proper memory management
    and comprehensive logging.
    """
    
    def __init__(
        self,
        config: AECFConfig,
        logger_type: str = "tensorboard",
        logger_config: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.logger_type = logger_type
        self.logger_config = logger_config or {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = AECF_CLIP(config)
        self.trainer = None
        self.callbacks = []
        
        self.logger.info(f"Initialized AECF trainer with config: {config}")
    
    def setup_logging(self) -> None:
        """Setup comprehensive logging."""
        self.logger = logging.getLogger(f"AECFTrainer_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def setup_callbacks(self) -> List[pl.Callback]:
        """Setup training callbacks for monitoring and checkpointing."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename='{epoch:02d}-{val_loss:.3f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.early_stopping_patience > 0:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=self.config.early_stopping_min_delta,
                patience=self.config.early_stopping_patience,
                verbose=True,
                mode='min'
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # Memory management callback
        memory_callback = MemoryManagementCallback()
        callbacks.append(memory_callback)
        
        self.callbacks = callbacks
        return callbacks
    
    def setup_logger(self) -> Optional[pl.loggers.Logger]:
        """Setup PyTorch Lightning logger."""
        if self.logger_type == "tensorboard":
            return TensorBoardLogger(
                save_dir=self.config.log_dir,
                name="aecf_experiment",
                **self.logger_config
            )
        elif self.logger_type == "wandb":
            return WandbLogger(
                project="aecf_clip",
                save_dir=self.config.log_dir,
                **self.logger_config
            )
        else:
            return None
    
    def setup_trainer(self) -> pl.Trainer:
        """Setup PyTorch Lightning trainer with proper configuration."""
        trainer_config = {
            "max_epochs": self.config.epochs,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "precision": "16-mixed" if self.config.mixed_precision else "32-true",
            "gradient_clip_val": self.config.gradient_clip_val,
            "gradient_clip_algorithm": "norm",
            "callbacks": self.setup_callbacks(),
            "logger": self.setup_logger(),
            "log_every_n_steps": 50,
            "val_check_interval": self.config.val_check_interval,
            "num_sanity_val_steps": 2,
            "enable_progress_bar": True,
            "enable_model_summary": True,
        }
        
        # Add additional trainer arguments from config
        if hasattr(self.config, 'trainer_kwargs'):
            trainer_config.update(self.config.trainer_kwargs)
        
        self.trainer = pl.Trainer(**trainer_config)
        return self.trainer
    
    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        ckpt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit the model with proper memory management.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            ckpt_path: Path to checkpoint for resuming training
            
        Returns:
            Training results dictionary
        """
        if self.trainer is None:
            self.setup_trainer()
        
        self.logger.info("Starting training...")
        self.logger.info(f"Model info: {self.model.get_model_info()}")
        
        try:
            self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=ckpt_path
            )
            
            # Get training results
            results = {
                "best_model_path": self.trainer.checkpoint_callback.best_model_path,
                "best_model_score": self.trainer.checkpoint_callback.best_model_score.item(),
                "current_epoch": self.trainer.current_epoch,
                "global_step": self.trainer.global_step,
                "logged_metrics": self.trainer.logged_metrics,
            }
            
            self.logger.info(f"Training completed. Best model: {results['best_model_path']}")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def test(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test the model."""
        if self.trainer is None:
            self.setup_trainer()
        
        self.logger.info("Starting testing...")
        
        try:
            test_results = self.trainer.test(
                self.model,
                dataloaders=test_dataloader,
                ckpt_path=ckpt_path
            )
            
            self.logger.info(f"Testing completed. Results: {test_results}")
            return test_results[0] if test_results else {}
            
        except Exception as e:
            self.logger.error(f"Testing failed: {str(e)}")
            raise
    
    def validate(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate the model."""
        if self.trainer is None:
            self.setup_trainer()
        
        self.logger.info("Starting validation...")
        
        try:
            val_results = self.trainer.validate(
                self.model,
                dataloaders=val_dataloader,
                ckpt_path=ckpt_path
            )
            
            self.logger.info(f"Validation completed. Results: {val_results}")
            return val_results[0] if val_results else {}
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise


class MemoryManagementCallback(pl.Callback):
    """Callback for managing memory usage during training."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("MemoryManagementCallback")
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clear memory at the start of each training epoch."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clear memory at the start of each validation epoch."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clear accumulated outputs and memory at the end of training epoch."""
        # Clear any accumulated outputs to prevent memory leaks
        if hasattr(pl_module, '_train_outputs'):
            pl_module._train_outputs.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clear accumulated outputs and memory at the end of validation epoch."""
        # Clear any accumulated outputs to prevent memory leaks
        if hasattr(pl_module, '_val_outputs'):
            pl_module._val_outputs.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MetricsAccumulator:
    """
    Memory-efficient metrics accumulator that avoids storing all batch outputs.
    
    This replaces the problematic pattern of accumulating all training outputs
    in memory, which was causing memory issues in the original implementation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.metrics_sum = {}
        self.count = 0
    
    def update(self, metrics: Dict[str, torch.Tensor]) -> None:
        """Update accumulated metrics with new batch metrics."""
        self.count += 1
        
        for key, value in metrics.items():
            if key not in self.metrics_sum:
                self.metrics_sum[key] = 0.0
            
            # Detach and convert to float to save memory
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            
            self.metrics_sum[key] += value
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        if self.count == 0:
            return {}
        
        return {key: val / self.count for key, val in self.metrics_sum.items()}
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest computed metrics."""
        return self.compute()

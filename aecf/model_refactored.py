"""
Core AECF model implementation.

This module contains the main AECF model with clean architecture and
proper separation of concerns.
"""

from typing import Dict, Optional, Tuple
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .config import AECFConfig
from .components import (
    EncoderFactory, AdaptiveGate, CurriculumMasker, OutputHeadFactory,
    validate_feature_dict, normalize_features
)
from .losses import AECFLoss
from .metrics import AECFMetrics


class AECFCore(nn.Module):
    """
    Core AECF model implementing adaptive early cross-modal fusion.
    
    This module contains the core fusion logic separated from the
    PyTorch Lightning training logic for better modularity.
    """
    
    def __init__(self, config: AECFConfig):
        super().__init__()
        self.config = config
        self.modalities = config.modalities
        
        # Create modality encoders
        self.encoders = nn.ModuleDict()
        for modality in self.modalities:
            # For pre-extracted features, use identity encoder
            encoder = EncoderFactory.create_encoder(
                modality=modality,
                input_dim=config.feat_dim,
                output_dim=config.feat_dim,
                encoder_type="identity"
            )
            self.encoders[modality] = encoder
        
        # Create adaptive gate
        if not config.gate_disabled:
            total_dim = len(self.modalities) * config.feat_dim
            self.gate = AdaptiveGate(
                input_dim=total_dim,
                num_modalities=len(self.modalities),
                hidden_dims=config.gate_hidden_dim,
                use_softmax=config.gate_use_softmax,
                learnable_temp=config.gate_learnable_temp,
                init_temp=config.gate_init_temp,
                min_temp=config.gate_min_temp
            )
        else:
            self.gate = None
        
        # Create curriculum masker
        if config.curriculum_mask:
            self.masker = CurriculumMasker(
                strategy=config.masking_strategy,
                prob_missing=config.masking_prob,
                tau=config.masking_tau
            )
        else:
            self.masker = None
        
        # Create output head
        self.output_head = OutputHeadFactory.create_head(
            task_type=config.task_type,
            feat_dim=config.feat_dim,
            output_dim=config.num_classes
        )
    
    def _validate_inputs(self, features: Dict[str, torch.Tensor]) -> None:
        """Validate input features."""
        validate_feature_dict(
            features=features,
            required_modalities=self.modalities,
            expected_feat_dim=self.config.feat_dim
        )
    
    def _normalize_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize features if enabled."""
        if self.config.feature_normalization:
            return normalize_features(features)
        return features
    
    def _encode_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality-specific encoders."""
        encoded = {}
        for modality in self.modalities:
            if modality in features:
                encoded[modality] = self.encoders[modality](features[modality])
        return encoded
    
    def _compute_fusion(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute adaptive fusion of modality features."""
        # Stack features for concatenation
        feature_list = [features[modality] for modality in self.modalities]
        batch_size = feature_list[0].size(0)
        
        if self.gate is None:
            # Simple average fusion when gate is disabled
            fused = torch.stack(feature_list, dim=0).mean(dim=0)
            weights = None
        else:
            # Adaptive gating
            concatenated = torch.cat(feature_list, dim=-1)
            weights, entropy = self.gate(concatenated)
            
            # Apply curriculum masking if enabled
            if self.masker is not None and self.training:
                mask = self.masker(weights)
                masked_weights = weights * mask
                # Renormalize to sum to 1
                masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-8)
                weights = masked_weights
            
            # Weighted fusion
            fused = torch.zeros_like(feature_list[0])
            for i, modality in enumerate(self.modalities):
                fused += weights[:, i:i+1] * features[modality]
        
        return fused, weights
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the core model.
        
        Args:
            features: Dictionary of modality features
            
        Returns:
            Tuple of (output_logits, modality_weights)
        """
        # Validate inputs
        self._validate_inputs(features)
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Apply encoders
        features = self._encode_features(features)
        
        # Compute fusion
        fused_features, weights = self._compute_fusion(features)
        
        # Apply output head
        output = self.output_head(fused_features)
        
        return output, weights


class AECF_CLIP(pl.LightningModule):
    """
    Production-ready AECF-CLIP model with PyTorch Lightning.
    
    This is the main model class that integrates all components
    with proper training, validation, and logging.
    """
    
    def __init__(self, config: AECFConfig):
        super().__init__()
        
        # Validate and store configuration
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Setup logging
        self.logger_obj = self._setup_logging()
        
        # Create core model
        self.model = AECFCore(config)
        
        # Create loss function
        self.loss_fn = AECFLoss(config)
        
        # Memory-efficient metrics accumulation
        from .training import MetricsAccumulator
        self._train_metrics = MetricsAccumulator()
        self._val_metrics = MetricsAccumulator()
        
        self.logger_obj.info(f"Initialized AECF-CLIP model with config: {config}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup model-specific logging."""
        logger = logging.getLogger(f"AECF_CLIP_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model."""
        return self.model(features)
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for training and validation."""
        # Extract features and targets
        features = {m: batch[m] for m in self.config.modalities if m in batch}
        targets = batch["label"]
        
        # Forward pass
        logits, weights = self(features)
        
        # Compute loss
        loss, loss_components = self.loss_fn(
            logits, targets, weights, 
            epoch=self.current_epoch,
            max_epochs=self.trainer.max_epochs if self.trainer else 100
        )
        
        # Compute metrics
        if self.config.task_type == "classification":
            metrics = AECFMetrics.compute_classification_metrics(logits, targets)
        else:
            # Placeholder metrics for other tasks
            metrics = {
                "accuracy": torch.tensor(0.0, device=logits.device),
                "loss": loss.detach()
            }
        
        # Combine all outputs
        outputs = {
            **loss_components,
            **metrics,
            "loss": loss
        }
        
        # Add modality weights if available
        if weights is not None:
            for i, modality in enumerate(self.config.modalities):
                outputs[f"weight_{modality}"] = weights[:, i].mean().detach()
            
            # Add entropy
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
            outputs["entropy"] = entropy.mean().detach()
        
        # Log metrics
        log_dict = {f"{stage}_{k}": v for k, v in outputs.items() if k != "loss"}
        self.log_dict(
            log_dict,
            prog_bar=(stage == "val"),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with memory-efficient metrics accumulation."""
        outputs = self._shared_step(batch, "train")
        
        # Use memory-efficient accumulation instead of storing all outputs
        metrics_to_accumulate = {k: v for k, v in outputs.items() if k != "loss"}
        self._train_metrics.update(metrics_to_accumulate)
        
        return outputs["loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with memory-efficient metrics accumulation."""
        outputs = self._shared_step(batch, "val")
        
        # Use memory-efficient accumulation instead of storing all outputs
        metrics_to_accumulate = {k: v for k, v in outputs.items() if k != "loss"}
        self._val_metrics.update(metrics_to_accumulate)
        
        return outputs["loss"]
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        outputs = self._shared_step(batch, "test")
        return outputs["loss"]
    
    def on_train_epoch_start(self) -> None:
        """Called at the start of training epoch."""
        self._train_metrics.reset()
    
    def on_validation_epoch_start(self) -> None:
        """Called at the start of validation epoch."""
        self._val_metrics.reset()
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log accumulated metrics
        avg_metrics = self._train_metrics.compute()
        if avg_metrics:
            self.log_dict(
                {f"train_epoch_{k}": v for k, v in avg_metrics.items()},
                on_epoch=True,
                prog_bar=False
            )
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Log accumulated metrics
        avg_metrics = self._val_metrics.compute()
        if avg_metrics:
            self.log_dict(
                {f"val_epoch_{k}": v for k, v in avg_metrics.items()},
                on_epoch=True,
                prog_bar=True
            )
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Separate parameter groups for different learning rates
        gate_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if "gate" in name:
                gate_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {
                "params": gate_params,
                "lr": self.config.gate_learning_rate,
                "weight_decay": 0.0
            },
            {
                "params": other_params,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay
            }
        ]
        
        optimizer = torch.optim.AdamW(param_groups)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information for debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "config": self.config.__dict__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "modalities": self.config.modalities,
            "task_type": self.config.task_type,
            "gate_disabled": self.config.gate_disabled
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_aecf_model(
    modalities: list[str] = None,
    task_type: str = "classification",
    num_classes: int = 80,
    **kwargs
) -> AECF_CLIP:
    """
    Convenience function to create AECF model with default configuration.
    
    Args:
        modalities: List of modalities to use
        task_type: Type of task (classification, regression, embedding)
        num_classes: Number of output classes/dimensions
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured AECF model
    """
    if modalities is None:
        modalities = ["image", "text"]
    
    config = AECFConfig(
        modalities=modalities,
        task_type=task_type,
        num_classes=num_classes,
        **kwargs
    )
    
    return AECF_CLIP(config)


def validate_model_inputs(
    features: Dict[str, torch.Tensor],
    required_modalities: list[str],
    feat_dim: int
) -> None:
    """Validate model inputs."""
    validate_feature_dict(features, required_modalities, feat_dim)


# ============================================================================
# Model Testing Utilities
# ============================================================================

def test_model_forward(model: AECF_CLIP, batch_size: int = 4) -> Dict[str, any]:
    """Test model forward pass with dummy data."""
    model.eval()
    
    # Create dummy data
    dummy_features = {}
    for modality in model.config.modalities:
        dummy_features[modality] = torch.randn(batch_size, model.config.feat_dim)
    
    try:
        with torch.no_grad():
            logits, weights = model(dummy_features)
        
        # Validate outputs
        expected_logits_shape = (batch_size, model.config.num_classes)
        if logits.shape != expected_logits_shape:
            return {"success": False, "error": f"Logits shape mismatch: {logits.shape} vs {expected_logits_shape}"}
        
        if weights is not None:
            expected_weights_shape = (batch_size, len(model.config.modalities))
            if weights.shape != expected_weights_shape:
                return {"success": False, "error": f"Weights shape mismatch: {weights.shape} vs {expected_weights_shape}"}
            
            # Check if weights sum to 1
            weight_sums = weights.sum(dim=1)
            if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6):
                return {"success": False, "error": "Weights don't sum to 1"}
        
        return {
            "success": True,
            "logits_shape": list(logits.shape),
            "weights_shape": list(weights.shape) if weights is not None else None,
            "logits_range": [logits.min().item(), logits.max().item()],
            "weights_range": [weights.min().item(), weights.max().item()] if weights is not None else None
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

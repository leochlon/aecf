"""
AECF-CLIP Model - Production-Ready Implementation

This module implements the Adaptive Early Cross-modal Fusion (AECF) model
with clean architecture, comprehensive validation, and maintainable design.

Key improvements over the original:
- Modular component design with clear separation of concerns
- Comprehensive input validation and error handling
- Type hints and detailed documentation
- Configurable components with validation
- Better testing and debugging support
- Production-ready error handling and logging
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Protocol
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class AECFConfig:
    """
    Structured configuration for AECF model with validation.
    
    This replaces the dict-based configuration with a type-safe,
    validated configuration object.
    """
    # Core model parameters
    modalities: List[str] = field(default_factory=lambda: ["image", "text"])
    feat_dim: int = 512
    num_classes: int = 80
    task_type: str = "classification"  # classification, regression, embedding
    
    # Gating network parameters
    gate_hidden_dim: int = 2048
    gate_use_softmax: bool = True
    gate_learnable_temp: bool = True
    gate_init_temp: float = 2.0
    gate_min_temp: float = 1.5
    gate_disabled: bool = False
    
    # Training parameters
    learning_rate: float = 1e-4
    gate_learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 30
    
    # Regularization parameters
    entropy_reg: bool = True
    entropy_free_epochs: int = 0
    entropy_warmup_epochs: int = 5
    entropy_max_lambda: float = 0.1
    
    # Curriculum masking parameters
    curriculum_mask: bool = True
    masking_strategy: str = "none"  # none, random, weighted_random, entropy_min, entropy_max
    masking_prob: float = 0.0
    masking_tau: float = 0.3
    
    # Consistency loss parameters
    consistency_coeff: float = 0.0
    consistency_ramp_epochs: int = 0
    
    # Loss parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    ece_penalty_coeff: float = 0.5
    l2_logits_coeff: float = 1e-4
    
    # Data parameters
    feature_normalization: bool = True
    label_frequencies: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate modalities
        if not self.modalities:
            raise ValueError("At least one modality must be specified")
        
        valid_modalities = {"image", "text", "audio"}
        for modality in self.modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality: {modality}. Valid: {valid_modalities}")
        
        # Validate dimensions
        if self.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {self.feat_dim}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        # Validate task type
        valid_tasks = {"classification", "regression", "embedding"}
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {self.task_type}. Valid: {valid_tasks}")
        
        # Validate learning rates
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.gate_learning_rate <= 0:
            raise ValueError(f"gate_learning_rate must be positive, got {self.gate_learning_rate}")
        
        # Validate masking strategy
        valid_strategies = {"none", "random", "weighted_random", "entropy_min", "entropy_max"}
        if self.masking_strategy not in valid_strategies:
            raise ValueError(f"Invalid masking_strategy: {self.masking_strategy}. Valid: {valid_strategies}")
        
        # Validate probability ranges
        if not 0 <= self.masking_prob <= 1:
            raise ValueError(f"masking_prob must be in [0,1], got {self.masking_prob}")
        if not 0 <= self.focal_alpha <= 1:
            raise ValueError(f"focal_alpha must be in [0,1], got {self.focal_alpha}")
        if not 0 <= self.label_smoothing <= 1:
            raise ValueError(f"label_smoothing must be in [0,1], got {self.label_smoothing}")
    
    def to_model_config(self) -> Dict[str, Any]:
        """Convert to model-compatible configuration dict."""
        return {
            "modalities": self.modalities,
            "feat_dim": self.feat_dim,
            "num_classes": self.num_classes,
            "task_type": self.task_type,
            "gate_hidden": self.gate_hidden_dim,
            "gate_disabled": self.gate_disabled,
            "lr": self.learning_rate,
            "gate_lr": self.gate_learning_rate,
            "wd": self.weight_decay,
            "epochs": self.epochs,
            "feature_norm": self.feature_normalization,
        }


# ============================================================================
# Component Protocols and Interfaces
# ============================================================================

class ModalityEncoder(Protocol):
    """Protocol for modality encoders."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode modality-specific features."""
        ...

class OutputAdapter(Protocol):
    """Protocol for output adapters."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply task-specific output transformation."""
        ...


# ============================================================================
# Modality Encoders
# ============================================================================

class LinearModalityEncoder(nn.Module):
    """
    Simple linear encoder for pre-extracted features.
    
    This is suitable for CLIP features or other pre-extracted representations.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Encoded features of shape [batch_size, output_dim]
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input [batch, features], got shape {x.shape}")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {x.size(1)}")
        
        return self.encoder(x)


class IdentityEncoder(nn.Module):
    """Identity encoder that passes features through unchanged."""
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.feat_dim:
            raise ValueError(f"Expected feat_dim {self.feat_dim}, got {x.size(-1)}")
        return x


class EncoderFactory:
    """Factory for creating modality encoders."""
    
    @staticmethod
    def create_encoder(
        modality: str,
        input_dim: int,
        output_dim: int,
        encoder_type: str = "linear"
    ) -> nn.Module:
        """
        Create a modality encoder.
        
        Args:
            modality: Modality name (image, text, audio)
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            encoder_type: Type of encoder (linear, identity)
            
        Returns:
            Encoder module
        """
        if encoder_type == "linear":
            return LinearModalityEncoder(input_dim, output_dim)
        elif encoder_type == "identity":
            if input_dim != output_dim:
                raise ValueError("Identity encoder requires input_dim == output_dim")
            return IdentityEncoder(output_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================================================================
# Adaptive Gating Network
# ============================================================================

class AdaptiveGate(nn.Module):
    """
    Adaptive gating mechanism for multi-modal fusion.
    
    Computes attention weights for different modalities based on
    concatenated features, with learnable temperature scaling.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_modalities: int,
        hidden_dims: Union[int, List[int]] = 2048,
        use_softmax: bool = True,
        learnable_temp: bool = True,
        init_temp: float = 2.0,
        min_temp: float = 1.5,
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.use_softmax = use_softmax
        self.learnable_temp = learnable_temp
        self.min_temp = min_temp
        self.eps = eps
        
        # Build MLP
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_modalities))
        self.gate_network = nn.Sequential(*layers)
        
        # Temperature parameter
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))
        else:
            self.register_buffer("temperature", torch.tensor(init_temp))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating weights and entropy.
        
        Args:
            x: Concatenated features [batch_size, input_dim]
            
        Returns:
            weights: Gating weights [batch_size, num_modalities]
            entropy: Per-sample entropy [batch_size]
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got shape {x.shape}")
        
        # Compute logits
        logits = self.gate_network(x)  # [batch_size, num_modalities]
        
        # Get temperature
        if self.learnable_temp:
            # Clamp temperature to minimum value
            with torch.no_grad():
                self.log_temperature.clamp_(min=math.log(self.min_temp))
            temperature = self.log_temperature.exp()
        else:
            temperature = self.temperature
        
        # Apply temperature scaling and activation
        scaled_logits = logits / temperature
        
        if self.use_softmax:
            weights = F.softmax(scaled_logits, dim=-1)
        else:
            weights = torch.sigmoid(scaled_logits)
            # Normalize to sum to 1 for consistency
            weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        
        # Compute Shannon entropy
        entropy = -(weights * (weights + self.eps).log()).sum(dim=-1)
        
        return weights, entropy


# ============================================================================
# Curriculum Masking
# ============================================================================

class CurriculumMasker(nn.Module):
    """
    Curriculum masking for progressive multi-modal training.
    
    Implements various masking strategies to encourage robust
    multi-modal representations during training.
    """
    
    def __init__(
        self,
        strategy: str = "none",
        prob_missing: float = 0.0,
        tau: float = 0.3
    ):
        super().__init__()
        
        valid_strategies = {"none", "random", "weighted_random", "entropy_min", "entropy_max"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Valid: {valid_strategies}")
        
        self.strategy = strategy
        self.register_buffer("prob_missing", torch.tensor(prob_missing, dtype=torch.float32))
        self.tau = tau
    
    @torch.no_grad()
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Generate masking pattern based on gating weights.
        
        Args:
            weights: Gating weights [batch_size, num_modalities]
            
        Returns:
            mask: Binary mask [batch_size, num_modalities]
        """
        if not self.training or self.strategy == "none":
            return torch.ones_like(weights)
        
        batch_size, num_modalities = weights.shape
        device = weights.device
        
        if self.strategy == "random":
            return (torch.rand_like(weights) > self.prob_missing).float()
        
        elif self.strategy == "weighted_random":
            # Higher weights → lower drop probability
            keep_prob = 1.0 - self.prob_missing * weights
            return torch.bernoulli(keep_prob)
        
        elif self.strategy in ["entropy_min", "entropy_max"]:
            # Compute entropy for each sample
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1, keepdim=True)
            
            if self.strategy == "entropy_min":
                # Keep low-entropy samples more often
                keep_prob = torch.sigmoid(-(entropy - self.tau) / self.tau)
            else:  # entropy_max
                # Keep high-entropy samples more often
                keep_prob = torch.sigmoid((entropy - self.tau) / self.tau)
            
            keep_prob = keep_prob.expand_as(weights)
            mask = torch.bernoulli(keep_prob)
            
            # Ensure at least one modality is kept per sample
            zero_rows = (mask.sum(dim=-1) == 0)
            if zero_rows.any():
                # Keep the highest weighted modality for zero rows
                best_modality = weights[zero_rows].argmax(dim=-1)
                mask[zero_rows] = 0
                mask[zero_rows, best_modality] = 1
            
            return mask
        
        return torch.ones_like(weights)


# ============================================================================
# Output Adapters
# ============================================================================

class ClassificationHead(nn.Module):
    """Output head for multi-label classification."""
    
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class RegressionHead(nn.Module):
    """Output head for regression tasks."""
    
    def __init__(self, feat_dim: int, output_dim: int):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class EmbeddingHead(nn.Module):
    """Output head for embedding tasks (identity)."""
    
    def __init__(self, feat_dim: int, output_dim: int):
        super().__init__()
        if feat_dim != output_dim:
            self.projection = nn.Linear(feat_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class OutputHeadFactory:
    """Factory for creating task-specific output heads."""
    
    @staticmethod
    def create_head(task_type: str, feat_dim: int, output_dim: int) -> nn.Module:
        """Create output head for the specified task."""
        if task_type == "classification":
            return ClassificationHead(feat_dim, output_dim)
        elif task_type == "regression":
            return RegressionHead(feat_dim, output_dim)
        elif task_type == "embedding":
            return EmbeddingHead(feat_dim, output_dim)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


# ============================================================================
# Core AECF Model
# ============================================================================

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
        
        # Create curriculum masker
        if config.curriculum_mask:
            self.masker = CurriculumMasker(
                strategy=config.masking_strategy,
                prob_missing=config.masking_prob,
                tau=config.masking_tau
            )
        
        # Create output head
        self.output_head = OutputHeadFactory.create_head(
            task_type=config.task_type,
            feat_dim=config.feat_dim,
            output_dim=config.num_classes
        )
    
    def _validate_inputs(self, features: Dict[str, torch.Tensor]) -> None:
        """Validate input features."""
        # Check all required modalities are present
        missing = [m for m in self.modalities if m not in features]
        if missing:
            raise ValueError(f"Missing modalities: {missing}")
        
        # Check feature shapes and types
        for modality, feat in features.items():
            if modality not in self.modalities:
                continue  # Skip extra modalities
            
            if not isinstance(feat, torch.Tensor):
                raise TypeError(f"{modality} features must be torch.Tensor, got {type(feat)}")
            
            if feat.dim() != 2:
                raise ValueError(f"{modality} features must be 2D [batch, feat], got shape {feat.shape}")
            
            if feat.size(1) != self.config.feat_dim:
                raise ValueError(
                    f"{modality} features must have dim {self.config.feat_dim}, "
                    f"got {feat.size(1)}"
                )
    
    def _normalize_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize features if enabled."""
        if not self.config.feature_normalization:
            return features
        
        normalized = {}
        for modality, feat in features.items():
            if modality in self.modalities:
                normalized[modality] = F.normalize(feat.float(), dim=-1)
            
        return normalized
    
    def _encode_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality-specific encoders."""
        encoded = {}
        for modality in self.modalities:
            encoded[modality] = self.encoders[modality](features[modality])
        return encoded
    
    def _compute_fusion(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute adaptive fusion of modality features."""
        if self.config.gate_disabled:
            # Simple average fusion
            feat_stack = torch.stack([features[m] for m in self.modalities], dim=1)
            fused = feat_stack.mean(dim=1)
            return fused, None
        
        # Concatenate features for gating network
        concat_feat = torch.cat([features[m] for m in self.modalities], dim=-1)
        
        # Compute gating weights
        weights, entropy = self.gate(concat_feat)
        
        # Apply curriculum masking if enabled
        if self.config.curriculum_mask and hasattr(self, 'masker'):
            mask = self.masker(weights)
            # Apply mask to features
            masked_features = {}
            for i, modality in enumerate(self.modalities):
                masked_features[modality] = features[modality] * mask[:, i:i+1]
        else:
            masked_features = features
            mask = None
        
        # Weighted fusion
        fused = torch.zeros_like(list(masked_features.values())[0])
        for i, modality in enumerate(self.modalities):
            fused += weights[:, i:i+1] * masked_features[modality]
        
        return fused, weights
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through AECF model.
        
        Args:
            features: Dict mapping modality names to feature tensors
                     Each tensor should be [batch_size, feat_dim]
        
        Returns:
            output: Task-specific output [batch_size, output_dim]
            weights: Modality weights [batch_size, num_modalities] or None
        """
        # Validate inputs
        self._validate_inputs(features)
        
        # Normalize features
        normalized = self._normalize_features(features)
        
        # Apply encoders
        encoded = self._encode_features(normalized)
        
        # Compute adaptive fusion
        fused, weights = self._compute_fusion(encoded)
        
        # Apply output head
        output = self.output_head(fused)
        
        return output, weights


# ============================================================================
# Loss Functions
# ============================================================================

class AECFLoss(nn.Module):
    """
    Comprehensive loss function for AECF model.
    
    Combines task-specific loss with entropy regularization
    and optional consistency loss.
    """
    
    def __init__(self, config: AECFConfig):
        super().__init__()
        self.config = config
        
        # Setup positive weights for classification if provided
        if config.task_type == "classification" and config.label_frequencies is not None:
            pos_weight = self._compute_pos_weights(config.label_frequencies)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
    
    def _compute_pos_weights(self, label_freq: torch.Tensor) -> torch.Tensor:
        """Compute positive weights from label frequencies."""
        pos_weight = ((1 - label_freq) / label_freq).clamp(max=10.0)
        return pos_weight
    
    def _focal_bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.05
    ) -> torch.Tensor:
        """Focal binary cross-entropy loss with label smoothing."""
        # Apply label smoothing
        targets_smooth = targets.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Focal loss terms
        pt_pos = probs
        pt_neg = 1 - probs
        
        # Positive weight handling
        pos_weight = self.pos_weight if self.pos_weight is not None else 1.0
        
        # Focal BCE loss
        loss = (
            -alpha * (pt_neg ** gamma) * pos_weight * targets_smooth * torch.log(pt_pos + 1e-8)
            - (1 - alpha) * (pt_pos ** gamma) * (1 - targets_smooth) * torch.log(pt_neg + 1e-8)
        )
        
        return loss.mean()
    
    def _compute_entropy_loss(self, weights: torch.Tensor, epoch: int, max_epochs: int) -> torch.Tensor:
        """Compute dynamic entropy regularization loss."""
        if not self.config.entropy_reg or weights is None:
            return torch.tensor(0.0, device=weights.device if weights is not None else 'cpu')
        
        # Compute dynamic lambda
        if epoch < self.config.entropy_free_epochs:
            lam = 0.0
        else:
            # Warmup phase
            warmup_progress = min(1.0, (epoch - self.config.entropy_free_epochs) / 
                                 max(1, self.config.entropy_warmup_epochs))
            lam = self.config.entropy_max_lambda * warmup_progress
            
            # Cosine decay in final 10%
            decay_start = int(0.9 * max_epochs)
            if epoch >= decay_start:
                decay_progress = (epoch - decay_start) / max(1, max_epochs - decay_start)
                lam *= 0.5 * (1 + math.cos(math.pi * decay_progress))
        
        # Compute entropy
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
        
        # Return negative entropy (encourage diversity)
        return -lam * entropy.mean()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 100
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and component losses.
        
        Args:
            logits: Model predictions
            targets: Ground truth targets
            weights: Modality weights (optional)
            epoch: Current training epoch
            max_epochs: Total training epochs
            
        Returns:
            total_loss: Combined loss
            loss_components: Dict of individual loss components
        """
        # Task-specific loss
        if self.config.task_type == "classification":
            task_loss = self._focal_bce_loss(
                logits, targets,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing
            )
            
            # Add ECE penalty for calibration
            probs = torch.sigmoid(logits)
            ece_loss = self.config.ece_penalty_coeff * self._ece_loss(probs, targets)
            
            # Add L2 regularization on logits
            l2_loss = self.config.l2_logits_coeff * logits.pow(2).mean()
            
        elif self.config.task_type == "regression":
            task_loss = F.mse_loss(logits, targets)
            ece_loss = torch.tensor(0.0, device=logits.device)
            l2_loss = torch.tensor(0.0, device=logits.device)
            
        else:  # multi-class classification
            task_loss = F.cross_entropy(logits, targets)
            ece_loss = torch.tensor(0.0, device=logits.device)
            l2_loss = torch.tensor(0.0, device=logits.device)
        
        # Entropy regularization
        entropy_loss = self._compute_entropy_loss(weights, epoch, max_epochs)
        
        # Total loss
        total_loss = task_loss + entropy_loss + ece_loss + l2_loss
        
        # Loss components for logging
        loss_components = {
            "task_loss": task_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "ece_loss": ece_loss.detach(),
            "l2_loss": l2_loss.detach(),
            "total_loss": total_loss.detach()
        }
        
        return total_loss, loss_components
    
    def _ece_loss(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
        """Expected Calibration Error for probability calibration."""
        conf = probs.flatten()
        correct = targets.bool().flatten().float()
        bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
        
        ece = torch.tensor(0.0, device=probs.device)
        for i in range(n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if mask.any():
                bin_conf = conf[mask].mean()
                bin_acc = correct[mask].mean()
                bin_size = mask.float().mean()
                ece += torch.abs(bin_conf - bin_acc) * bin_size
        
        return ece


# ============================================================================
# Metrics
# ============================================================================

class AECFMetrics:
    """Metrics computation for AECF model."""
    
    @staticmethod
    def compute_classification_metrics(
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute classification metrics."""
        probs = torch.sigmoid(logits)
        
        # Top-1 recall
        top1_preds = probs.topk(1, dim=-1).indices
        top1_recall = (targets.gather(1, top1_preds) > 0).float().mean()
        
        # Mean Average Precision
        map_score = AECFMetrics._batch_map(probs, targets)
        
        # Expected Calibration Error
        ece = AECFMetrics._ece(probs, targets)
        
        return {
            "top1_recall": top1_recall,
            "map": map_score,
            "ece": ece
        }
    
    @staticmethod
    def _batch_map(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute batch-wise mean average precision."""
        batch_size, num_classes = probs.shape
        
        # Get top predictions
        top1_preds = probs.argmax(dim=1)
        
        # Check if predictions are correct
        correct = targets.gather(1, top1_preds.unsqueeze(1)).squeeze(1).float()
        
        # Per-class precision
        pred_counts = torch.bincount(top1_preds, minlength=num_classes).float()
        correct_counts = torch.bincount(top1_preds, weights=correct, minlength=num_classes)
        
        # Avoid division by zero
        mask = pred_counts > 0
        precision = torch.zeros_like(pred_counts)
        precision[mask] = correct_counts[mask] / pred_counts[mask]
        
        # Return mean precision for predicted classes
        if mask.any():
            return precision[mask].mean()
        else:
            return torch.tensor(0.0, device=probs.device)
    
    @staticmethod
    def _ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
        """Expected Calibration Error."""
        conf = probs.flatten()
        correct = targets.bool().flatten().float()
        bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
        
        ece = torch.tensor(0.0, device=probs.device)
        for i in range(n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if mask.any():
                bin_conf = conf[mask].mean()
                bin_acc = correct[mask].mean()
                bin_weight = mask.float().mean()
                ece += torch.abs(bin_conf - bin_acc) * bin_weight
        
        return ece


# ============================================================================
# Main AECF Model (PyTorch Lightning)
# ============================================================================

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
        
        # Storage for batch outputs
        self._train_outputs = []
        self._val_outputs = []
        
        self.logger_obj.info(f"Initialized AECF-CLIP model with config: {config}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup model-specific logging."""
        logger = logging.getLogger(f"AECF_CLIP_{id(self)}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(levelname)s] %(name)s: %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
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
        """Training step."""
        outputs = self._shared_step(batch, "train")
        self._train_outputs.append(outputs)
        return outputs["loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self._shared_step(batch, "val")
        self._val_outputs.append(outputs)
        return outputs["loss"]
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        if self._train_outputs:
            # Log epoch summary
            avg_metrics = self._compute_epoch_metrics(self._train_outputs, "train")
            self.logger_obj.info(f"Train Epoch {self.current_epoch}: {avg_metrics}")
            self._train_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self._val_outputs:
            # Log epoch summary
            avg_metrics = self._compute_epoch_metrics(self._val_outputs, "val")
            self.logger_obj.info(f"Val Epoch {self.current_epoch}: {avg_metrics}")
            self._val_outputs.clear()
    
    def _compute_epoch_metrics(self, outputs: List[Dict], stage: str) -> Dict[str, float]:
        """Compute average metrics over epoch."""
        if not outputs:
            return {}
        
        metrics = {}
        for key in outputs[0].keys():
            if key != "loss":  # Skip loss as it's logged separately
                values = [o[key] for o in outputs if key in o]
                if values:
                    metrics[f"{stage}_{key}"] = torch.stack(values).mean().item()
        
        return metrics
    
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
    
    def get_model_info(self) -> Dict[str, Any]:
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
    modalities: List[str] = None,
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
    required_modalities: List[str],
    feat_dim: int
) -> None:
    """
    Validate model inputs before forward pass.
    
    Args:
        features: Input features dict
        required_modalities: Required modality names
        feat_dim: Expected feature dimension
    
    Raises:
        ValueError: If validation fails
    """
    # Check required modalities
    missing = [m for m in required_modalities if m not in features]
    if missing:
        raise ValueError(f"Missing required modalities: {missing}")
    
    # Check feature shapes and types
    for modality, feat in features.items():
        if modality not in required_modalities:
            continue
        
        if not isinstance(feat, torch.Tensor):
            raise ValueError(f"{modality} must be torch.Tensor, got {type(feat)}")
        
        if feat.dim() != 2:
            raise ValueError(f"{modality} must be 2D [batch, feat], got {feat.shape}")
        
        if feat.size(1) != feat_dim:
            raise ValueError(f"{modality} must have {feat_dim} features, got {feat.size(1)}")


# ============================================================================
# Model Testing Utilities
# ============================================================================

def test_model_forward(model: AECF_CLIP, batch_size: int = 4) -> Dict[str, Any]:
    """
    Test model forward pass with dummy data.
    
    Args:
        model: AECF model to test
        batch_size: Batch size for dummy data
    
    Returns:
        Test results including shapes and statistics
    """
    device = next(model.parameters()).device
    feat_dim = model.config.feat_dim
    
    # Create dummy features
    features = {}
    for modality in model.config.modalities:
        features[modality] = torch.randn(batch_size, feat_dim, device=device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, weights = model(features)
    
    results = {
        "input_shapes": {m: f.shape for m, f in features.items()},
        "output_shape": logits.shape,
        "weights_shape": weights.shape if weights is not None else None,
        "output_range": [logits.min().item(), logits.max().item()],
        "weights_sum": weights.sum(dim=-1).mean().item() if weights is not None else None,
        "success": True
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    config = AECFConfig(
        modalities=["image", "text"],
        task_type="classification",
        num_classes=80,
        feat_dim=512
    )
    
    model = AECF_CLIP(config)
    print(f"Created AECF model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_results = test_model_forward(model)
    print(f"Forward pass test: {test_results}")

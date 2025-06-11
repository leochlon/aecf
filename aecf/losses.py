"""
Loss functions for AECF model.

This module contains all loss computation logic with proper separation
from the main model class.
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AECFConfig


class AECFLoss(nn.Module):
    """
    Comprehensive loss computation for AECF model.
    
    Combines classification/regression loss with regularization terms
    including entropy regularization and consistency loss.
    """
    
    def __init__(self, config: AECFConfig):
        super().__init__()
        self.config = config
        
        # Store frequently used config values
        self.task_type = config.task_type
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.label_smoothing = config.label_smoothing
        self.entropy_max_lambda = config.entropy_max_lambda if hasattr(config, 'entropy_max_lambda') else config.entropy_max_coeff
        self.entropy_free_epochs = config.entropy_free_epochs
        self.entropy_warmup_epochs = config.entropy_warmup_epochs
        self.consistency_coeff = config.consistency_coeff
        self.consistency_ramp_epochs = config.consistency_ramp_epochs
        self.ece_penalty_coeff = config.ece_penalty_coeff
        self.l2_logits_coeff = config.l2_logits_coeff
        self.entropy_reg = config.entropy_reg if hasattr(config, 'entropy_reg') else True
        
        # Initialize positive weights for imbalanced classification
        if config.label_frequencies is not None and config.task_type == "classification":
            label_freq = torch.tensor(config.label_frequencies, dtype=torch.float32)
            pos_weight = torch.clamp(1.0 / (label_freq + 1e-8), min=0.1, max=10.0)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
    
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
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth targets [batch_size, num_classes] or [batch_size]
            weights: Modality weights [batch_size, num_modalities]
            epoch: Current training epoch
            max_epochs: Total training epochs
            
        Returns:
            Total loss and dictionary of loss components
        """
        loss_components = {}
        
        # Primary task loss
        if self.task_type == "classification":
            primary_loss = self._compute_classification_loss(logits, targets)
        elif self.task_type == "regression":
            primary_loss = self._compute_regression_loss(logits, targets)
        else:  # embedding
            primary_loss = torch.tensor(0.0, device=logits.device)
        
        loss_components["primary_loss"] = primary_loss
        total_loss = primary_loss
        
        # Entropy regularization
        if weights is not None and self.config.entropy_reg:
            entropy_loss = self._compute_entropy_loss(weights, epoch, max_epochs)
            loss_components["entropy_loss"] = entropy_loss
            total_loss = total_loss + entropy_loss
        
        # ECE penalty for classification
        if self.task_type == "classification" and self.ece_penalty_coeff > 0:
            ece_loss = self._compute_ece_penalty(logits, targets)
            loss_components["ece_loss"] = ece_loss
            total_loss = total_loss + ece_loss
        
        # L2 regularization on logits
        if self.l2_logits_coeff > 0:
            l2_loss = self.l2_logits_coeff * logits.pow(2).mean()
            loss_components["l2_loss"] = l2_loss
            total_loss = total_loss + l2_loss
        
        # Consistency loss (if needed - placeholder for future implementation)
        if self.consistency_coeff > 0:
            consistency_loss = self._compute_consistency_loss(logits, targets, epoch, max_epochs)
            loss_components["consistency_loss"] = consistency_loss
            total_loss = total_loss + consistency_loss
        
        loss_components["total_loss"] = total_loss
        return total_loss, loss_components
    
    def _compute_classification_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute classification loss with focal loss and label smoothing."""
        if targets.dtype == torch.long:
            # Multi-class classification
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        else:
            # Multi-label classification with focal BCE
            return self._focal_bce_loss(logits, targets)
    
    def _compute_regression_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute regression loss."""
        return F.mse_loss(logits, targets)
    
    def _focal_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal binary cross-entropy loss for multi-label classification.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss
        if self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Apply focal weighting
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
        
        focal_loss = focal_weight * bce_loss
        
        # Label smoothing for multi-label
        if self.label_smoothing > 0:
            smooth_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            smooth_loss = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction='none')
            focal_loss = (1 - self.label_smoothing) * focal_loss + self.label_smoothing * smooth_loss
        
        return focal_loss.mean()
    
    def _compute_entropy_loss(
        self, 
        weights: torch.Tensor, 
        epoch: int, 
        max_epochs: int
    ) -> torch.Tensor:
        """Compute entropy regularization loss."""
        # Calculate current lambda
        lambda_val = self._get_entropy_lambda(epoch, max_epochs)
        
        # Compute entropy: H = -sum(w * log(w))
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
        
        # Negative entropy loss (encourage diversity)
        entropy_loss = -lambda_val * entropy.mean()
        
        return entropy_loss
    
    def _get_entropy_lambda(self, epoch: int, max_epochs: int) -> float:
        """Get dynamic entropy regularization coefficient."""
        if epoch < self.entropy_free_epochs:
            return 0.0
        
        if epoch < self.entropy_free_epochs + self.entropy_warmup_epochs:
            # Linear warmup
            warmup_progress = (epoch - self.entropy_free_epochs) / self.entropy_warmup_epochs
            return self.entropy_max_lambda * warmup_progress
        
        # Cosine annealing after warmup
        remaining_epochs = max_epochs - self.entropy_free_epochs - self.entropy_warmup_epochs
        if remaining_epochs > 0:
            progress = (epoch - self.entropy_free_epochs - self.entropy_warmup_epochs) / remaining_epochs
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.entropy_max_lambda * cosine_factor
        
        return self.entropy_max_lambda
    
    def _compute_ece_penalty(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Expected Calibration Error penalty."""
        if targets.dtype == torch.long:
            # Multi-class ECE
            probs = F.softmax(logits, dim=-1)
            ece = self._expected_calibration_error(probs, targets)
        else:
            # Multi-label ECE (simplified)
            probs = torch.sigmoid(logits)
            ece = self._binary_ece(probs, targets)
        
        return self.ece_penalty_coeff * ece
    
    def _expected_calibration_error(
        self, 
        probs: torch.Tensor, 
        targets: torch.Tensor, 
        n_bins: int = 15
    ) -> torch.Tensor:
        """Compute Expected Calibration Error for multi-class classification."""
        device = probs.device
        
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(targets)
        
        ece = torch.zeros(1, device=device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _binary_ece(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Simplified ECE for multi-label classification."""
        # Average over all labels
        abs_diff = torch.abs(probs - targets)
        return abs_diff.mean()
    
    def _compute_consistency_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        epoch: int, 
        max_epochs: int
    ) -> torch.Tensor:
        """Placeholder for consistency loss implementation."""
        # This would require additional forward passes with different augmentations
        # For now, return zero
        return torch.tensor(0.0, device=logits.device)

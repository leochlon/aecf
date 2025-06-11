"""
Metrics computation for AECF model.

This module provides comprehensive metrics calculation for different task types.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


class AECFMetrics:
    """Metrics computation for AECF model."""
    
    @staticmethod
    def compute_classification_metrics(
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification metrics.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth targets [batch_size, num_classes] or [batch_size]
            
        Returns:
            Dictionary of metrics
        """
        if targets.dtype == torch.long:
            # Multi-class classification
            return AECFMetrics._compute_multiclass_metrics(logits, targets)
        else:
            # Multi-label classification
            return AECFMetrics._compute_multilabel_metrics(logits, targets)
    
    @staticmethod
    def _compute_multiclass_metrics(
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for multi-class classification."""
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        # Accuracy
        accuracy = (predictions == targets).float().mean()
        
        # Top-5 accuracy (if applicable)
        if logits.size(1) >= 5:
            top5_pred = logits.topk(5, dim=-1).indices
            top5_accuracy = (top5_pred == targets.unsqueeze(1)).any(dim=1).float().mean()
        else:
            top5_accuracy = accuracy
        
        # Confidence
        confidence = probs.max(dim=-1).values.mean()
        
        return {
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "confidence": confidence,
        }
    
    @staticmethod
    def _compute_multilabel_metrics(
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for multi-label classification."""
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        
        # Per-class metrics
        precision_per_class = AECFMetrics._precision_per_class(predictions, targets)
        recall_per_class = AECFMetrics._recall_per_class(predictions, targets)
        
        # Mean Average Precision (mAP)
        map_score = AECFMetrics._mean_average_precision(probs, targets)
        
        # Exact match ratio
        exact_match = (predictions == targets).all(dim=1).float().mean()
        
        # Hamming loss
        hamming_loss = (predictions != targets).float().mean()
        
        # Top-1 accuracy (max probability class)
        top1_pred = probs.argmax(dim=-1)
        top1_targets = targets.argmax(dim=-1)
        top1_accuracy = (top1_pred == top1_targets).float().mean()
        
        return {
            "map": map_score,
            "exact_match": exact_match,
            "hamming_loss": hamming_loss,
            "accuracy": top1_accuracy,  # For compatibility
            "precision": precision_per_class.mean(),
            "recall": recall_per_class.mean(),
        }
    
    @staticmethod
    def compute_regression_metrics(
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute regression metrics."""
        mse = F.mse_loss(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        
        # R-squared
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": torch.sqrt(mse),
            "r2": r2,
        }
    
    @staticmethod
    def compute_modality_metrics(weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute metrics related to modality weights.
        
        Args:
            weights: Modality weights [batch_size, num_modalities]
            
        Returns:
            Dictionary of modality-related metrics
        """
        # Entropy of weights (diversity measure)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        
        # Weight statistics
        weight_mean = weights.mean(dim=0)
        weight_std = weights.std(dim=0)
        
        # Effective number of modalities (inverse participation ratio)
        effective_modalities = 1.0 / (weights ** 2).sum(dim=-1).mean()
        
        metrics = {
            "entropy": entropy,
            "effective_modalities": effective_modalities,
        }
        
        # Add per-modality weights
        for i in range(weights.size(1)):
            metrics[f"weight_mod_{i}"] = weight_mean[i]
            metrics[f"weight_std_mod_{i}"] = weight_std[i]
        
        return metrics
    
    @staticmethod
    def _precision_per_class(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute precision for each class."""
        tp = (predictions * targets).sum(dim=0)
        fp = (predictions * (1 - targets)).sum(dim=0)
        precision = tp / (tp + fp + 1e-8)
        return precision
    
    @staticmethod
    def _recall_per_class(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute recall for each class."""
        tp = (predictions * targets).sum(dim=0)
        fn = ((1 - predictions) * targets).sum(dim=0)
        recall = tp / (tp + fn + 1e-8)
        return recall
    
    @staticmethod
    def _mean_average_precision(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute mean Average Precision (mAP) for multi-label classification.
        
        This is a simplified implementation. For production, consider using
        torchmetrics.AveragePrecision for more robust computation.
        """
        # Sort by probabilities (descending)
        sorted_probs, sorted_indices = torch.sort(probs, dim=0, descending=True)
        sorted_targets = targets.gather(0, sorted_indices)
        
        # Compute precision-recall curves and APs for each class
        aps = []
        for class_idx in range(probs.size(1)):
            class_targets = sorted_targets[:, class_idx]
            class_probs = sorted_probs[:, class_idx]
            
            # Skip if no positive examples
            if class_targets.sum() == 0:
                aps.append(torch.tensor(0.0, device=probs.device))
                continue
            
            # Compute cumulative precision and recall
            tp_cumsum = class_targets.cumsum(0).float()
            fp_cumsum = (1 - class_targets).cumsum(0).float()
            
            recall = tp_cumsum / (class_targets.sum() + 1e-8)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            
            # Compute AP using trapezoidal rule approximation
            recall_diff = torch.cat([recall[:1], recall[1:] - recall[:-1]])
            ap = (recall_diff * precision).sum()
            aps.append(ap)
        
        return torch.stack(aps).mean()
    
    @staticmethod
    def compute_calibration_metrics(
        probs: torch.Tensor, 
        targets: torch.Tensor, 
        n_bins: int = 15
    ) -> Dict[str, torch.Tensor]:
        """Compute calibration metrics."""
        if targets.dtype == torch.long:
            # Multi-class calibration
            ece = AECFMetrics._expected_calibration_error(probs, targets, n_bins)
            mce = AECFMetrics._maximum_calibration_error(probs, targets, n_bins)
        else:
            # Multi-label calibration (simplified)
            ece = (probs - targets).abs().mean()
            mce = (probs - targets).abs().max()
        
        return {
            "ece": ece,
            "mce": mce,
        }
    
    @staticmethod
    def _expected_calibration_error(
        probs: torch.Tensor, 
        targets: torch.Tensor, 
        n_bins: int = 15
    ) -> torch.Tensor:
        """Compute Expected Calibration Error."""
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
    
    @staticmethod
    def _maximum_calibration_error(
        probs: torch.Tensor, 
        targets: torch.Tensor, 
        n_bins: int = 15
    ) -> torch.Tensor:
        """Compute Maximum Calibration Error."""
        device = probs.device
        
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(targets)
        
        mce = torch.zeros(1, device=device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            
            if in_bin.any():
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = torch.max(mce, bin_error)
        
        return mce

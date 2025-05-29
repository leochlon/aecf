"""AECF-CLIP model implementation with all components."""

import math
from typing import Dict, Any, List, Tuple, Sequence, Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging

# ============================================================================
# Encoders for different modalities
# ============================================================================

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512, **kwargs):
        super().__init__()
        self.proj = nn.Linear(kwargs.get('input_dim', output_dim), output_dim)
    def forward(self, x):
        return self.proj(x)

class TextEncoder(nn.Module):
    def __init__(self, output_dim=512, **kwargs):
        super().__init__()
        self.proj = nn.Linear(kwargs.get('input_dim', output_dim), output_dim)
    def forward(self, x):
        return self.proj(x)

class AudioEncoder(nn.Module):
    def __init__(self, output_dim=512, **kwargs):
        super().__init__()
        self.proj = nn.Linear(kwargs.get('input_dim', output_dim), output_dim)
    def forward(self, x):
        return self.proj(x)

class ModularEncoderFactory:
    """Factory for creating encoders for different modality types"""
    
    @staticmethod
    def get_encoder(modality_type, output_dim=512, **kwargs):
        if modality_type == "image":
            return ImageEncoder(output_dim, **kwargs)
        elif modality_type == "text":
            return TextEncoder(output_dim, **kwargs)
        elif modality_type == "audio":
            return AudioEncoder(output_dim, **kwargs)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")

# ============================================================================
# Output adapters for different tasks
# ============================================================================

class OutputAdapterRegistry:
    """Registry for output adapters by task type."""
    _registry = {}

    @classmethod
    def register(cls, task_type):
        def decorator(adapter_cls):
            cls._registry[task_type] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def get_adapter(cls, task_type, feat_dim, output_dim):
        if task_type not in cls._registry:
            raise ValueError(f"No adapter registered for task: {task_type}")
        return cls._registry[task_type](feat_dim, output_dim)

# Example adapters
@OutputAdapterRegistry.register("classification")
class ClassificationAdapter(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.head = nn.Linear(feat_dim, output_dim)
    def forward(self, x):
        return self.head(x)

@OutputAdapterRegistry.register("regression")
class RegressionAdapter(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.head(x)

@OutputAdapterRegistry.register("embedding")
class EmbeddingAdapter(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.head = nn.Identity()
    def forward(self, x):
        return self.head(x)

# ============================================================================
# Gating Network for modality fusion
# ============================================================================

class GatingNet(nn.Module):
    """
    Adaptive gating mechanism for multi-modality fusion with learnable temperature.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    n_modalities : int, optional
        Number of modalities to gate (e.g., 2 for image and text), by default 2.
    hidden_dims : Sequence[int] | int, optional
        Hidden layer dimensions for the MLP. Can be a sequence of ints for multiple 
        layers or a single int for one hidden layer, by default 2048.
    use_softmax : bool, optional
        Whether to use softmax (True) or sigmoid (False) for gating, by default True.
    init_T : float, optional
        Initial temperature for softmax/sigmoid scaling, by default 2.0.
    learnable_T : bool, optional
        Whether temperature should be a learnable parameter, by default True.
    min_T : float, optional
        Minimum temperature value (for learnable temperature), by default 1.5.
    eps : float, optional
        Small constant to avoid log(0) in entropy computation, by default 1e-8.
    """
    def __init__(self, 
                 input_dim: int,
                 n_modalities: int = 2,
                 hidden_dims: Sequence[int] | int = 2048,
                 use_softmax: bool = True, 
                 init_T: float = 2.0,
                 learnable_T: bool = True,
                 min_T: float = 1.5,
                 eps: float = 1e-8) -> None:
        super().__init__()
        
        # Convert single hidden dimension to a sequence
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_modalities))
        self.net = nn.Sequential(*layers)
        
        # Temperature handling
        self.use_softmax = use_softmax
        self.learnable_T = learnable_T
        self.min_T = min_T
        self.eps = eps
        
        if learnable_T:
            self.log_T = nn.Parameter(torch.tensor(math.log(init_T)))
        else:
            self.register_buffer("temperature", torch.tensor(init_T))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating probabilities and Shannon entropy.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (B, input_dim).

        Returns
        -------
        probs : torch.Tensor
            Gating probabilities of shape (B, n_modalities).
        entropy : torch.Tensor
            Per-sample Shannon entropy of shape (B,).
        """
        logits = self.net(x)  # Shape: (B, n_modalities)
        
        # Get temperature
        if self.learnable_T:
            with torch.no_grad():
                self.log_T.clamp_(min=math.log(self.min_T))
            T = self.log_T.exp()
        else:
            T = self.temperature
        
        # Apply activation
        if self.use_softmax:
            probs = F.softmax(logits / T, dim=-1)
        else:
            probs = torch.sigmoid(logits / T)
        
        # Calculate entropy
        entropy = -(probs * (probs + self.eps).log()).sum(dim=-1)  # Shape: (B,)
        
        return probs, entropy

# ============================================================================
# Masking utilities
# ============================================================================

class CurriculumMasker(nn.Module):
    """
    Stochastic masking of modality-specific weights **w_raw ∈ ℝ^{B×M}**.

    strategy ∈ {
        "none"            : keep both modalities,
        "random"          : Bernoulli(p_missing) independent of w_raw,
        "weighted_random" : Bernoulli weighted by softmax(w_raw),
        "entropy_min"     : keep samples with low entropy more often,
        "entropy_max"     : keep samples with HIGH entropy more often,
    }

    Returned mask has the same shape as w_raw and is ∈ {0,1}.
    """
    def __init__(self,
                 strategy   : str   = "none",
                 p_missing  : float = 0.0,
                 tau        : float = 0.3):
        super().__init__()
        assert strategy in {"none", "random", "weighted_random",
                            "entropy_min", "entropy_max"}
        self.strategy = strategy
        self.register_buffer("p_missing",
                             torch.tensor(float(p_missing), dtype=torch.float32))
        self.tau = tau

    @torch.no_grad()
    def forward(self, w_raw: torch.Tensor) -> torch.Tensor:          # (B,M)
        if (not self.training) or self.strategy == "none":
            return torch.ones_like(w_raw)

        B, M = w_raw.shape
        device = w_raw.device

        if self.strategy == "random":
            return (torch.rand(B, M, device=device) > self.p_missing).float()

        if self.strategy == "weighted_random":
            probs = w_raw.softmax(dim=-1)                # each row sums to 1
            keep  = 1.0 - self.p_missing * probs         # higher w ⇒ less drop
            return torch.bernoulli(keep)

        # entropy-based strategies ------------------------------------------
        with torch.no_grad():
            w_norm = w_raw.softmax(dim=-1)
            ent    = -(w_norm * (w_norm + 1e-8).log()).sum(dim=-1, keepdim=True)

        if self.strategy == "entropy_min":       # prefer keeping low-entropy
            # Original method
            keep_prob = torch.sigmoid(-(ent - self.tau) / self.tau)
            keep_prob = keep_prob.expand_as(w_raw)   # (B,1) → (B,M)
            mask = torch.bernoulli(keep_prob)
            
            # Ensure at least one modality is kept
            if (mask.sum(dim=-1) == 0).any():
                # Alternative implementation: drop lowest weight modality if entropy is low
                min_m = F.one_hot(w_norm.argmin(-1), M).float()
                alt_keep = 1.0 - min_m * (w_norm.min(-1, keepdim=True).values < self.tau)
                # Use alternative mask for rows where all modalities were dropped
                zero_rows = (mask.sum(dim=-1) == 0)
                mask[zero_rows] = alt_keep[zero_rows]
            
            return mask
            
        if self.strategy == "entropy_max":       # prefer keeping high-entropy
            # Original method
            keep_prob = torch.sigmoid((ent - self.tau) / self.tau)
            keep_prob = keep_prob.expand_as(w_raw)   # (B,1) → (B,M)
            mask = torch.bernoulli(keep_prob)
            
            # Ensure at least one modality is kept
            if (mask.sum(dim=-1) == 0).any():
                # Alternative implementation: drop highest weight modality if entropy is high
                max_m = F.one_hot(w_norm.argmax(-1), M).float()
                alt_keep = 1.0 - max_m * (w_norm.max(-1, keepdim=True).values > 1 - self.tau)
                # Use alternative mask for rows where all modalities were dropped
                zero_rows = (mask.sum(dim=-1) == 0)
                mask[zero_rows] = alt_keep[zero_rows]
            
            return mask
        
        # Should never reach here
        return torch.ones_like(w_raw)

def apply_adaptive_mask(
        batch: dict,
        epoch: int,
        warmup_epochs: int = 15,
        pi_max: float = 0.30,
) -> dict:
    """
    Implements Eq. (4) in the paper: π_t = π_max * min(1, epoch / warmup).

    Parameters
    ----------
    batch : dict with keys "image", "text" (frozen CLIP feats) and "label".
    epoch : current training epoch (0-based).
    warmup_epochs : how many epochs to ramp π.
    pi_max : the final drop probability after warm-up.

    Returns
    -------
    Same `batch` dict but with
        • zeroed-out image/text features,
        • an extra key "mask" ∈{0,1}^{B×2} for debugging/logging.
    """
    # --- 1. compute current π_t -------------------------------------------
    pi_t = pi_max * min(1.0, epoch / float(warmup_epochs))

    # --- 2. draw Bernoulli masks (B,2) ------------------------------------
    B = batch["image"].size(0)
    keep = (torch.rand(B, 2, device=batch["image"].device) > pi_t).float()

    # --- 3. apply masks ----------------------------------------------------
    batch = batch.copy()  # Don't modify original batch
    batch["image"] = batch["image"] * keep[:, 0:1]  # broad-/broadcast to feat dim
    batch["text"] = batch["text"] * keep[:, 1:2]
    batch["mask"] = keep  # (B,2)

    return batch

# ============================================================================
# Utility functions for the model
# ============================================================================

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def validate_modalities(features: dict, required_modalities: list = None):
    if required_modalities is not None:
        missing = [m for m in required_modalities if m not in features]
        if missing:
            raise ValueError(f"Missing required modalities: {missing}")
    if not features:
        raise ValueError("No modalities provided in input features.")
    return True

# ============================================================================
# Main model implementation
# ============================================================================

class AECF_CLIP(pl.LightningModule):
    """
    Adaptive Entropy-Gated Contrastive Fusion on frozen CLIP features.
    Implements curriculum masking, entropy regularisation, and consistency loss.
    """

    # -----------------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)           # so PL stores cfg in checkpoint
        self.cfg = cfg
        self._custom_logger = get_logger("AECF_CLIP")

        # Setup modalities and encoders
        self.modalities = cfg.get("modalities", ["image", "text"])
        self.encoders = nn.ModuleDict()
        feat_dims = {}
        for modality in self.modalities:
            encoder_cfg = cfg.get(f"{modality}_encoder_cfg", {})
            encoder = ModularEncoderFactory.get_encoder(modality, **encoder_cfg)
            self.encoders[modality] = encoder
            feat_dims[modality] = encoder_cfg.get("output_dim", cfg.get("feat_dim", 512))
        self.feat_dims = feat_dims

        # Setup gating and output adapter
        self.gate = GatingNet(
            sum(feat_dims.values()), 
            n_modalities=len(self.modalities),
            hidden_dims=cfg.get("gate_hidden", 2048)
        )
        
        self.output_adapter = OutputAdapterRegistry.get_adapter(
            cfg.get("task_type", "classification"),
            cfg.get("feat_dim", 512),
            cfg.get("num_classes", 80)
        )

        # Setup masking
        self.masker = CurriculumMasker(
            cfg.get("masking_mode", "none"),
            cfg.get("p_missing", 0.0),
            cfg.get("tau", 0.3)
        )

        # ---------- curriculum & loss hyper-params ----------
        self.entropy_free   = cfg.get("entropy_free", 0)
        self.entropy_warmup = cfg.get("entropy_warmup", 5)
        self.entropy_max    = cfg.get("entropy_max", 0.1)
        self.cec_coef       = cfg.get("cec_coef", 0.0)
        self.cec_ramp_ep    = cfg.get("cec_ramp_epochs", 0)

        # ---------- optimisation ----------
        self.lr           = cfg.get("lr", 1e-4)
        self.gate_lr      = cfg.get("gate_lr", 1e-3)   
        self.weight_decay = cfg.get("wd", 1e-2)

        # ---------- logging helpers ----------
        self._train_outs: List[Dict[str, torch.Tensor]] = []
        self._val_outs  : List[Dict[str, torch.Tensor]] = []
        self.last_train_metrics: Dict[str, float] = {}
        self.last_val_metrics:   Dict[str, float] = {}
        
        # ---------- label frequency handling for classification ----------
        if cfg.get("task_type", "classification") == "classification":
            if "label_freq" in cfg:
                # This should be moved to the output adapter
                self.register_buffer("pos_weight", self._compute_pos_weights(cfg["label_freq"]))
            else:
                self._custom_logger.warning("No label_freq provided – using uniform weights")
                self.register_buffer("pos_weight", 
                                    torch.ones(cfg.get("num_classes", 80), 
                                              device=self.device))

    def _compute_pos_weights(self, label_freq):
        """Compute positive weights from label frequency for BCE loss."""
        pos_w = ((1 - label_freq) / label_freq).clamp(max=10.0)
        return pos_w

    # ---------------------------------------------------------------- utility
    @staticmethod
    def _ece(probs, labels, n_bins=15):
        """Compute Expected Calibration Error."""
        conf = probs.flatten()
        corr = labels.bool().flatten().float()
        bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)

        err = torch.zeros((), device=probs.device)
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.any():
                err += (conf[m].mean() - corr[m].mean()).abs() * m.float().mean()

        return err

    def _lambda_dyn(self) -> float:
        """
        Compute the dynamic lambda coefficient for entropy regularization.
        
        Returns
        -------
        float
            The current lambda value based on training progress
        """
        if not hasattr(self, 'trainer') or self.trainer is None:
            return 0.0

        epoch_now  = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs

        if epoch_now < self.entropy_free:
            return 0.0

        # ---- warm-up to λ_max ----
        prog = min(1.0, (epoch_now - self.entropy_free) /
                          max(1, self.entropy_warmup))
        lam = self.entropy_max * prog     # flat after warm-up

        # ---- cosine decay in last 10 % ----
        decay_start = int(0.9 * max_epochs)
        if epoch_now >= decay_start:
            t = (epoch_now - decay_start) / max(1, max_epochs - decay_start)
            lam *= 0.5 * (1 + math.cos(math.pi * t))   # λ_max → 0.0

        return lam

    # ---------------------------------------------------------------- forward
    def forward(self, features: Dict[str, torch.Tensor]):
        """
        Forward pass through the model.
        
        Parameters
        ----------
        features : Dict[str, torch.Tensor]
            Dictionary of features for each modality
            
        Returns
        -------
        tuple
            (output logits, modality weights)
        """
        validate_modalities(features, self.modalities)
        
        # Process each modality through its encoder
        normalized_features = {}
        for modality in self.modalities:
            feat = features[modality]
            if self.cfg.get("feature_norm", True):
                normalized_features[modality] = F.normalize(feat.float(), dim=-1)
            else:
                normalized_features[modality] = feat.float()
        
        # Concatenate for gating network
        concat_features = torch.cat([normalized_features[m] for m in self.modalities], dim=-1)
        
        # Get weights from gating network
        weights, entropy = self.gate(concat_features)
        
        # Apply masking if enabled
        if self.training and self.cfg.get("masking_mode", "none") != "none":
            mask = self.masker(weights)
            for i, modality in enumerate(self.modalities):
                normalized_features[modality] = normalized_features[modality] * mask[:, i:i+1]
        
        # Weighted fusion of features
        fused = torch.zeros_like(list(normalized_features.values())[0])
        for i, modality in enumerate(self.modalities):
            fused += weights[:, i:i+1] * normalized_features[modality]
        
        # Apply output adapter
        output = self.output_adapter(fused)
        
        self._custom_logger.debug(f"Forward pass: modalities={self.modalities}, weights={weights.mean(dim=0).tolist()}")
        return output, weights

    # ---------------------------------------------------------------- shared
    def _shared_step(self,
                     batch: Dict[str, torch.Tensor],
                     stage: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Shared step for training and validation.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of data
        stage : str
            'train' or 'val'
            
        Returns
        -------
        tuple
            (loss, metrics dict)
        """
        # Validate input batch has required modalities
        if not all(m in batch for m in self.modalities):
            missing = [m for m in self.modalities if m not in batch]
            raise ValueError(f"Missing modalities in batch: {missing}")
        
        # Get features and labels
        features = {m: batch[m] for m in self.modalities}
        labels = batch["label"]
        
        # Forward pass
        logits, weights = self(features)
        
        # Handle different task types
        task_type = self.cfg.get("task_type", "classification")
        
        if task_type == "classification":
            # Binary classification loss
            loss_ce = self._focal_bce_loss(
                logits, labels,
                alpha=0.25, gamma=2.0, smooth=0.05,
                pos_weight=getattr(self, "pos_weight", None)
            )
            
            # Calculate metrics
            probs = torch.sigmoid(logits)
            top1 = probs.topk(1, dim=-1).indices
            rec1 = (labels.gather(1, top1) > 0).float().mean()
            pos_e = self._ece(probs, labels)
            ap = self._batch_map1(probs, labels)
            
        elif task_type == "regression":
            # MSE loss for regression
            loss_ce = F.mse_loss(logits, labels)
            
            # Placeholder metrics for regression
            probs = logits
            rec1 = torch.tensor(0.0, device=self.device)
            pos_e = torch.tensor(0.0, device=self.device)
            ap = torch.tensor(0.0, device=self.device)
            
        else:
            # Default to cross-entropy for multi-class
            loss_ce = F.cross_entropy(logits, labels)
            
            probs = F.softmax(logits, dim=-1)
            rec1 = (logits.argmax(dim=1) == labels).float().mean()
            pos_e = self._ece(probs, labels)
            ap = torch.tensor(0.0, device=self.device)

        # ---------- λ(x) regulariser ----------------------------------------
        lam = self._lambda_dyn()                     # scalar float
        gate_H = -(weights * (weights + 1e-8).log()).sum(dim=-1) # (B,)
        loss_ent = -lam * gate_H.mean()

        if torch.rand(1).item() < 0.01:
            weight_str = " ".join([f"w_{m}={weights[:,i].mean():.3f}" 
                                 for i, m in enumerate(self.modalities)])
            self._custom_logger.info(f"[dbg] ep={self.current_epoch:02d} "
                  f"λ={lam:.3f}  H={gate_H.mean():.4f}  {weight_str}")

        # ------------------- consistency loss (CEC) -------------
        loss_cec = torch.tensor(0., device=self.device)
        if self.cec_coef > 0 and task_type == "classification":
            with torch.no_grad():
                p_t = torch.sigmoid(logits)
                
            # Apply CEC to each modality
            loss_cec_raw = 0
            for i, modality in enumerate(self.modalities):
                feat = F.normalize(features[modality].float(), dim=-1)
                modality_logits = self.output_adapter(feat)
                loss_cec_raw += F.mse_loss(torch.sigmoid(modality_logits), p_t)
            
            ramp = min(1.0, self.current_epoch / self.cec_ramp_ep) if self.cec_ramp_ep > 0 else 1.0
            loss_cec = self.cec_coef * ramp * loss_cec_raw

        # ------------------- total loss -------------------------
        loss = loss_ce + loss_ent + loss_cec
        
        # Add ECE penalty if doing classification
        if task_type == "classification":
            ece_pen = self._ece(probs, labels)
            loss_ece = 0.5 * ece_pen
            loss += loss_ece
            
            # Small L2 on logits to prevent explosion
            loss += 1e-4 * logits.pow(2).mean()

        # ------------------- metrics ----------------------------
        out = {
            "loss": loss.detach(),
            "rec1": rec1.detach(),
            "ece": pos_e.detach(),
            "gate_H": gate_H.mean().detach(),
            "lambda_": torch.tensor(lam, device=self.device),
            "mAP": ap.detach(),
        }
        
        # Add per-modality weights to metrics
        for i, modality in enumerate(self.modalities):
            out[f"w_{modality}"] = weights[:, i].mean().detach()

        # Log metrics
        self.log_dict({f"{stage}_{k}": v for k, v in out.items()},
                      prog_bar=(stage == "val"),
                      on_step=False, on_epoch=True)

        return loss, out

    # ------------------- Utility methods moved from utils.py ----------------
    def _batch_map1(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Mini-batch mAP@1 (mean precision-at-1 across classes).
        """
        B, C = logits.shape
        top1 = logits.argmax(dim=1)                 # (B,)

        # 1 if the predicted class is actually present in the GT for that sample
        correct = labels.gather(1, top1.unsqueeze(1)).squeeze(1).float()  # (B,)

        # per-class precision
        preds_per_class   = torch.bincount(top1, minlength=C).float()     # (C,)
        correct_per_class = torch.bincount(top1, weights=correct, minlength=C)

        # avoid div-by-zero; mask classes never predicted in this batch
        mask = preds_per_class > 0
        precision_per_class = torch.zeros_like(preds_per_class)
        precision_per_class[mask] = correct_per_class[mask] / preds_per_class[mask]

        if mask.any():
            return precision_per_class[mask].mean()
        else:                          # rare corner-case: batch too small
            return torch.tensor(0.0, device=logits.device)

    def _focal_bce_loss(self, logits, targets, *, alpha=0.25, gamma=2.0, smooth=0.05, pos_weight=None):
        """
        Focal BCE loss with label smoothing.
        """
        targets = targets.float() * (1 - smooth) + 0.5 * smooth
        p  = torch.sigmoid(logits)
        pt_pos, pt_neg = p, 1 - p
        w_pos = pos_weight if pos_weight is not None else 1.0
        loss = (
            -alpha     * (pt_neg ** gamma) * w_pos * targets     * torch.log(pt_pos + 1e-8)
            -(1-alpha) * (pt_pos ** gamma)        * (1-targets) * torch.log(pt_neg + 1e-8)
        )
        return loss.mean()

    # ---------------------------------------------------------------- hooks
    def training_step(self, batch, *_):
        loss, out = self._shared_step(batch, "train")
        self._train_outs.append(out)
        return loss

    def validation_step(self, batch, *_):
        loss, out = self._shared_step(batch, "val")
        self._val_outs.append(out)
        return loss

    def on_train_epoch_start(self):
        self._train_outs.clear()

    def on_validation_epoch_start(self):
        self._val_outs.clear()

    # ───────────────── train epoch end ─────────────────
    def on_train_epoch_end(self):
        if not self._train_outs:
            return
        metrics = {k: torch.stack([x[k] for x in self._train_outs if k in x]).mean().item()
                   for k in self._train_outs[0].keys()}
        self.last_train_metrics = metrics
        self._train_outs.clear()

    # ──────────────── validation epoch end ─────────────
    def on_validation_epoch_end(self):
        if not self._val_outs:
            return
        metrics = {k: torch.stack([x[k] for x in self._val_outs if k in x]).mean().item()
                   for k in self._val_outs[0].keys()}
        self.last_val_metrics = metrics
        self._val_outs.clear()

    def configure_optimizers(self):
        """Configure optimizers with separate learning rates for gate and other parameters."""
        param_groups = [
            {"params": self.gate.parameters(), "lr": self.gate_lr, "weight_decay": 0.0},
            {"params": [p for n, p in self.named_parameters() 
                         if not n.startswith("gate.")], 
             "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        
        opt = torch.optim.AdamW(param_groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.get("epochs", 30))
        return {"optimizer": opt, "lr_scheduler": sched}

    def configure_output_head(self, task_type=None, output_dim=None):
        """
        Reconfigure the output adapter for a different task.
        
        Parameters
        ----------
        task_type : str, optional
            Type of task (classification, regression, etc.)
        output_dim : int, optional
            Output dimension
        """
        if task_type is None:
            task_type = self.cfg.get("task_type", "classification")
        if output_dim is None:
            output_dim = self.cfg.get("num_classes", 80)
            
        self.output_adapter = OutputAdapterRegistry.get_adapter(
            task_type,
            self.cfg.get("feat_dim", 512),
            output_dim
        )
        self._custom_logger.info(f"Configured output head: {task_type}, output_dim={output_dim}")

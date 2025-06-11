"""
AECF Core Components - Production Ready

This module implements focused, reusable nn.Module components that address 
PyTorch's concerns about monolithic design. Each component has a single 
responsibility and can be used independently.

Key improvements:
- Input validation with clear error messages
- Type hints throughout
- Comprehensive docstrings
- Protocol-based design for extensibility
- Memory-efficient implementations
- Proper error handling
"""

from __future__ import annotations
import math
import logging
from typing import Dict, Optional, List, Tuple, Protocol, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Component Protocols for Type Safety
# ============================================================================

class ModalityEncoder(Protocol):
    """Protocol defining interface for modality encoders."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode modality-specific features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_dim]
            
        Returns
        -------
        torch.Tensor
            Encoded features [batch_size, output_dim]
        """
        ...


class OutputAdapter(Protocol):
    """Protocol defining interface for output adapters."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt fused features for specific task.
        
        Parameters
        ----------
        x : torch.Tensor
            Fused features [batch_size, feat_dim]
            
        Returns
        -------
        torch.Tensor
            Task-specific outputs [batch_size, output_dim]
        """
        ...


# ============================================================================
# Input Validation Utilities
# ============================================================================

def validate_tensor_input(
    tensor: torch.Tensor,
    name: str,
    expected_dims: int,
    expected_shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[torch.dtype] = None
) -> None:
    """
    Validate tensor input with clear error messages.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate
    name : str
        Name of tensor for error messages
    expected_dims : int
        Expected number of dimensions
    expected_shape : Tuple[int, ...], optional
        Expected shape (use -1 for any size)
    dtype : torch.dtype, optional
        Expected data type
        
    Raises
    ------
    TypeError
        If input is not a tensor
    ValueError
        If tensor properties don't match expectations
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
    
    if tensor.dim() != expected_dims:
        raise ValueError(f"{name} must be {expected_dims}D, got {tensor.dim()}D with shape {tensor.shape}")
    
    if expected_shape is not None:
        for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(f"{name} dimension {i} must be {expected}, got {actual}")
    
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def validate_feature_dict(
    features: Dict[str, torch.Tensor],
    required_modalities: List[str],
    feat_dim: int
) -> None:
    """
    Validate feature dictionary input.
    
    Parameters
    ----------
    features : Dict[str, torch.Tensor]
        Feature dictionary
    required_modalities : List[str]
        Required modality keys
    feat_dim : int
        Expected feature dimension
        
    Raises
    ------
    ValueError
        If features are invalid
    """
    # Check required modalities
    missing = [m for m in required_modalities if m not in features]
    if missing:
        raise ValueError(f"Missing required modalities: {missing}")
    
    # Check each feature tensor
    batch_sizes = []
    for modality, feat in features.items():
        if modality in required_modalities:
            validate_tensor_input(feat, f"features['{modality}']", 2)
            if feat.size(1) != feat_dim:
                raise ValueError(f"features['{modality}'] must have {feat_dim} features, got {feat.size(1)}")
            batch_sizes.append(feat.size(0))
    
    # Check consistent batch sizes
    if len(set(batch_sizes)) > 1:
        raise ValueError(f"Inconsistent batch sizes across modalities: {dict(zip(required_modalities, batch_sizes))}")


def normalize_features(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize features to unit length.
    
    Parameters
    ----------
    features : Dict[str, torch.Tensor]
        Feature dictionary
        
    Returns
    -------
    Dict[str, torch.Tensor]
        Normalized features
    """
    normalized = {}
    for modality, feat in features.items():
        normalized[modality] = F.normalize(feat.float(), dim=-1)
    return normalized


# ============================================================================
# Focused Modality Encoders
# ============================================================================

class IdentityEncoder(nn.Module):
    """
    Identity encoder for pre-extracted features.
    
    Use this when features are already in the correct format.
    Memory efficient - no extra parameters.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize identity encoder.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        output_dim : int
            Output feature dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through identity encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_dim]
            
        Returns
        -------
        torch.Tensor
            Output features [batch_size, output_dim]
        """
        validate_tensor_input(x, "input", 2, (-1, self.input_dim))
        
        if self.proj is not None:
            return self.proj(x)
        return x


class LinearEncoder(nn.Module):
    """
    Simple linear encoder with optional activation and dropout.
    
    Efficient single-layer transformation with regularization.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_activation: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize linear encoder.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        output_dim : int
            Output feature dimension
        use_activation : bool
            Whether to use ReLU activation
        dropout : float
            Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.proj = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU() if use_activation else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_dim]
            
        Returns
        -------
        torch.Tensor
            Encoded features [batch_size, output_dim]
        """
        validate_tensor_input(x, "input", 2, (-1, self.input_dim))
        
        x = self.proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder.
    
    For more complex feature transformations that require multiple layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.1
    ):
        """
        Initialize MLP encoder.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        output_dim : int
            Output feature dimension
        hidden_dims : List[int]
            Hidden layer dimensions
        activation : str
            Activation function name
        dropout : float
            Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Get activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_dim]
            
        Returns
        -------
        torch.Tensor
            Encoded features [batch_size, output_dim]
        """
        validate_tensor_input(x, "input", 2, (-1, self.input_dim))
        return self.net(x)


class EncoderFactory:
    """
    Factory for creating modality encoders.
    
    Centralizes encoder creation with validation and consistent interfaces.
    """
    
    AVAILABLE_ENCODERS = {
        "identity": IdentityEncoder,
        "linear": LinearEncoder,
        "mlp": MLPEncoder
    }
    
    @classmethod
    def create_encoder(
        cls,
        modality: str,
        input_dim: int,
        output_dim: int,
        encoder_type: str = "identity",
        **kwargs
    ) -> ModalityEncoder:
        """
        Create encoder for specific modality.
        
        Parameters
        ----------
        modality : str
            Modality name (for logging)
        input_dim : int
            Input feature dimension
        output_dim : int
            Output feature dimension
        encoder_type : str
            Type of encoder to create
        **kwargs
            Additional encoder-specific parameters
            
        Returns
        -------
        ModalityEncoder
            Created encoder
            
        Raises
        ------
        ValueError
            If encoder type is unknown
        """
        if encoder_type not in cls.AVAILABLE_ENCODERS:
            raise ValueError(f"Unknown encoder type: {encoder_type}. "
                           f"Available: {list(cls.AVAILABLE_ENCODERS.keys())}")
        
        encoder_class = cls.AVAILABLE_ENCODERS[encoder_type]
        encoder = encoder_class(input_dim, output_dim, **kwargs)
        
        logger.debug(f"Created {encoder_type} encoder for {modality}: "
                    f"{input_dim} -> {output_dim}")
        
        return encoder


# ============================================================================
# Adaptive Gating Network
# ============================================================================

class AdaptiveGate(nn.Module):
    """
    Adaptive gating mechanism for multi-modal fusion.
    
    This is a focused component that learns to weight different modalities
    based on their concatenated features. Includes proper temperature scaling
    and entropy computation for regularization.
    
    Key features:
    - Input validation with clear error messages
    - Temperature clamping for numerical stability
    - Both softmax and sigmoid gating options
    - Entropy computation for regularization
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
        """
        Initialize adaptive gate.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension (concatenated modalities)
        num_modalities : int
            Number of modalities to gate
        hidden_dims : Union[int, List[int]]
            Hidden layer dimensions
        use_softmax : bool
            Whether to use softmax (True) or sigmoid (False)
        learnable_temp : bool
            Whether temperature should be learnable
        init_temp : float
            Initial temperature value
        min_temp : float
            Minimum temperature value (for numerical stability)
        eps : float
            Small constant for numerical stability
        """
        super().__init__()
        
        # Validate inputs
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_modalities <= 0:
            raise ValueError(f"num_modalities must be positive, got {num_modalities}")
        if init_temp <= 0:
            raise ValueError(f"init_temp must be positive, got {init_temp}")
        if min_temp <= 0:
            raise ValueError(f"min_temp must be positive, got {min_temp}")
        if min_temp > init_temp:
            raise ValueError("min_temp cannot be greater than init_temp")
        
        self.input_dim = input_dim
        self.num_modalities = num_modalities
        self.use_softmax = use_softmax
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
            self.min_temp = min_temp
        else:
            self.register_buffer("temperature", torch.tensor(init_temp))
            self.min_temp = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating weights and entropy.
        
        Parameters
        ----------
        x : torch.Tensor
            Concatenated features [batch_size, input_dim]
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - weights: Modality weights [batch_size, num_modalities]
            - entropy: Shannon entropy per sample [batch_size]
        """
        validate_tensor_input(x, "input", 2, (-1, self.input_dim))
        
        # Compute gate logits
        logits = self.gate_network(x)
        
        # Get temperature
        if hasattr(self, 'log_temperature'):
            # Clamp temperature for numerical stability
            temp = torch.clamp(self.log_temperature.exp(), min=self.min_temp)
        else:
            temp = self.temperature
        
        # Apply temperature scaling
        scaled_logits = logits / temp
        
        # Apply activation
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
    Curriculum masking for progressive training difficulty.
    
    Implements various masking strategies to selectively drop modalities
    during training, helping the model learn robust representations.
    
    This is a focused component that can be used independently.
    """
    
    def __init__(
        self,
        strategy: str = "none",
        prob_missing: float = 0.0,
        tau: float = 0.3
    ):
        """
        Initialize curriculum masker.
        
        Parameters
        ----------
        strategy : str
            Masking strategy: "none", "random", "weighted_random", 
            "entropy_min", "entropy_max"
        prob_missing : float
            Base probability of masking modalities
        tau : float
            Temperature parameter for entropy-based strategies
        """
        super().__init__()
        
        valid_strategies = {"none", "random", "weighted_random", "entropy_min", "entropy_max"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Valid: {valid_strategies}")
        
        if not 0 <= prob_missing <= 1:
            raise ValueError(f"prob_missing must be in [0,1], got {prob_missing}")
        
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        
        self.strategy = strategy
        self.register_buffer("prob_missing", torch.tensor(prob_missing))
        self.tau = tau
    
    @torch.no_grad()
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Generate curriculum mask.
        
        Parameters
        ----------
        weights : torch.Tensor
            Modality weights [batch_size, num_modalities]
            
        Returns
        -------
        torch.Tensor
            Binary mask [batch_size, num_modalities]
        """
        validate_tensor_input(weights, "weights", 2)
        
        if not self.training or self.strategy == "none":
            return torch.ones_like(weights)
        
        batch_size, num_modalities = weights.shape
        device = weights.device
        
        if self.strategy == "random":
            return (torch.rand_like(weights) > self.prob_missing).float()
        
        elif self.strategy == "weighted_random":
            # Higher weights -> less likely to be dropped
            probs = F.softmax(weights, dim=-1)
            keep_prob = 1.0 - self.prob_missing * probs
            return torch.bernoulli(keep_prob)
        
        else:  # entropy-based strategies
            # Compute entropy per sample
            normalized_weights = F.softmax(weights, dim=-1)
            entropy = -(normalized_weights * (normalized_weights + 1e-8).log()).sum(dim=-1, keepdim=True)
            
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


# ============================================================================
# Output Adapters
# ============================================================================

class ClassificationHead(nn.Module):
    """
    Output head for multi-label classification.
    
    Focused component for classification tasks with proper initialization
    and optional class balancing.
    """
    
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        use_bias: bool = True,
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Initialize classification head.
        
        Parameters
        ----------
        feat_dim : int
            Input feature dimension
        num_classes : int
            Number of output classes
        use_bias : bool
            Whether to use bias in output layer
        pos_weight : torch.Tensor, optional
            Positive class weights for balanced training
        """
        super().__init__()
        
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {feat_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(feat_dim, num_classes, bias=use_bias)
        
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.classifier.weight)
        if use_bias:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, feat_dim]
            
        Returns
        -------
        torch.Tensor
            Classification logits [batch_size, num_classes]
        """
        validate_tensor_input(x, "input", 2, (-1, self.feat_dim))
        return self.classifier(x)


class RegressionHead(nn.Module):
    """
    Output head for regression tasks.
    
    Simple linear layer with optional activation for regression.
    """
    
    def __init__(
        self,
        feat_dim: int,
        output_dim: int,
        activation: Optional[str] = None
    ):
        """
        Initialize regression head.
        
        Parameters
        ----------
        feat_dim : int
            Input feature dimension
        output_dim : int
            Output dimension
        activation : str, optional
            Output activation function
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        
        self.regressor = nn.Linear(feat_dim, output_dim)
        
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regression head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, feat_dim]
            
        Returns
        -------
        torch.Tensor
            Regression outputs [batch_size, output_dim]
        """
        validate_tensor_input(x, "input", 2, (-1, self.feat_dim))
        return self.activation(self.regressor(x))


class EmbeddingHead(nn.Module):
    """
    Output head for embedding tasks.
    
    Identity transformation with optional L2 normalization.
    """
    
    def __init__(self, feat_dim: int, normalize: bool = True):
        """
        Initialize embedding head.
        
        Parameters
        ----------
        feat_dim : int
            Feature dimension
        normalize : bool
            Whether to L2 normalize outputs
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.normalize = normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, feat_dim]
            
        Returns
        -------
        torch.Tensor
            Embedding outputs [batch_size, feat_dim]
        """
        validate_tensor_input(x, "input", 2, (-1, self.feat_dim))
        
        if self.normalize:
            return F.normalize(x, dim=-1)
        return x


class OutputHeadFactory:
    """
    Factory for creating task-specific output heads.
    
    Centralizes output head creation with validation.
    """
    
    AVAILABLE_HEADS = {
        "classification": ClassificationHead,
        "regression": RegressionHead,
        "embedding": EmbeddingHead
    }
    
    @classmethod
    def create_head(
        cls,
        task_type: str,
        feat_dim: int,
        output_dim: int,
        **kwargs
    ) -> OutputAdapter:
        """
        Create output head for specific task.
        
        Parameters
        ----------
        task_type : str
            Type of task
        feat_dim : int
            Input feature dimension
        output_dim : int
            Output dimension
        **kwargs
            Additional head-specific parameters
            
        Returns
        -------
        OutputAdapter
            Created output head
        """
        if task_type not in cls.AVAILABLE_HEADS:
            raise ValueError(f"Unknown task type: {task_type}. "
                           f"Available: {list(cls.AVAILABLE_HEADS.keys())}")
        
        head_class = cls.AVAILABLE_HEADS[task_type]
        
        if task_type == "classification":
            head = head_class(feat_dim, output_dim, **kwargs)
        elif task_type == "regression":
            head = head_class(feat_dim, output_dim, **kwargs)
        else:  # embedding
            head = head_class(feat_dim, **kwargs)
        
        logger.debug(f"Created {task_type} head: {feat_dim} -> {output_dim}")
        return head


class CurriculumMasker(nn.Module):
    """
    Curriculum masking for progressive learning.
    
    Implements various masking strategies to progressively increase
    the difficulty of multi-modal fusion during training.
    """
    
    def __init__(
        self,
        strategy: str = "none",
        prob_missing: float = 0.0,
        tau: float = 0.3
    ):
        super().__init__()
        self.strategy = strategy
        self.prob_missing = prob_missing
        self.tau = tau
        
        valid_strategies = {"none", "random", "weighted_random", "entropy_min", "entropy_max"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid masking strategy: {strategy}")
    
    def forward(
        self,
        weights: torch.Tensor,
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply curriculum masking based on weights.
        
        Args:
            weights: Modality weights [batch_size, num_modalities]
            epoch: Current training epoch
            max_epochs: Total training epochs
            
        Returns:
            Masking tensor [batch_size, num_modalities]
        """
        if self.strategy == "none" or not self.training:
            return torch.ones_like(weights)
        
        batch_size, num_modalities = weights.shape
        
        if self.strategy == "random":
            # Random masking with fixed probability
            mask = torch.rand_like(weights) > self.prob_missing
            
        elif self.strategy == "weighted_random":
            # Random masking with weights-based probability
            prob = self.prob_missing * (1 - weights)  # Higher prob for lower weights
            mask = torch.rand_like(weights) > prob
            
        elif self.strategy == "entropy_min":
            # Mask modality with minimum entropy contribution
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1, keepdim=True)
            probs = F.softmax(-weights / self.tau, dim=-1)  # Lower weights = higher prob
            mask = torch.rand_like(weights) > (self.prob_missing * probs)
            
        elif self.strategy == "entropy_max":
            # Mask modality with maximum entropy contribution
            probs = F.softmax(weights / self.tau, dim=-1)  # Higher weights = higher prob
            mask = torch.rand_like(weights) > (self.prob_missing * probs)
            
        else:
            mask = torch.ones_like(weights)
        
        # Ensure at least one modality remains unmasked
        all_masked = ~mask.any(dim=-1, keepdim=True)
        if all_masked.any():
            # Randomly unmask one modality for fully masked samples
            random_idx = torch.randint(0, num_modalities, (batch_size, 1), device=weights.device)
            mask.scatter_(1, random_idx, True)
        
        return mask.float()


# ============================================================================
# Output Heads
# ============================================================================

class ClassificationHead(nn.Module):
    """Classification output head."""
    
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RegressionHead(nn.Module):
    """Regression output head."""
    
    def __init__(self, feat_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class EmbeddingHead(nn.Module):
    """Embedding output head (identity)."""
    
    def __init__(self, feat_dim: int, output_dim: int):
        super().__init__()
        if feat_dim != output_dim:
            self.head = nn.Linear(feat_dim, output_dim)
        else:
            self.head = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class OutputHeadFactory:
    """Factory for creating output heads."""
    
    @staticmethod
    def create_head(task_type: str, feat_dim: int, output_dim: int) -> OutputAdapter:
        """Create output head for specific task."""
        if task_type == "classification":
            return ClassificationHead(feat_dim, output_dim)
        elif task_type == "regression":
            return RegressionHead(feat_dim, output_dim)
        elif task_type == "embedding":
            return EmbeddingHead(feat_dim, output_dim)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


# ============================================================================
# Input Validation Utilities
# ============================================================================

def validate_feature_dict(
    features: Dict[str, torch.Tensor],
    required_modalities: List[str],
    expected_feat_dim: int
) -> None:
    """Validate input feature dictionary."""
    # Check required modalities
    missing = [m for m in required_modalities if m not in features]
    if missing:
        raise ValueError(f"Missing required modalities: {missing}")
    
    # Check tensor properties
    for modality, tensor in features.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor for {modality}, got {type(tensor)}")
        
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor for {modality}, got {tensor.dim()}D")
        
        if tensor.size(1) != expected_feat_dim:
            raise ValueError(f"Feature dim mismatch for {modality}: expected {expected_feat_dim}, got {tensor.size(1)}")


def normalize_features(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize features if needed."""
    normalized = {}
    for modality, tensor in features.items():
        # L2 normalization
        normalized[modality] = F.normalize(tensor, dim=-1, p=2)
    return normalized

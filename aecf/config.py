"""
AECF Configuration Management with Validation

Production-ready configuration system for AECF models with comprehensive
validation, type safety, and clear error messages.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


@dataclass
class AECFConfig:
    """
    Structured configuration for AECF model with comprehensive validation.
    
    This replaces the dict-based configuration with a type-safe,
    validated configuration object that prevents runtime errors.
    """
    
    # Core model parameters
    modalities: List[str] = field(default_factory=lambda: ["image", "text"])
    feat_dim: int = 512
    num_classes: int = 80
    task_type: str = "classification"
    
    # Gating network parameters
    gate_hidden_dims: List[int] = field(default_factory=lambda: [2048])
    gate_hidden_dim: int = 2048  # For backward compatibility
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
    gradient_clip_val: float = 1.0
    mixed_precision: bool = True
    val_check_interval: float = 1.0
    
    # Checkpointing and logging
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Additional trainer kwargs
    trainer_kwargs: dict = field(default_factory=dict)
    
    # Regularization parameters
    entropy_free_epochs: int = 0
    entropy_warmup_epochs: int = 5
    entropy_max_coeff: float = 0.1
    ece_penalty_coeff: float = 0.5
    l2_logits_coeff: float = 1e-4
    
    # Curriculum masking parameters
    curriculum_mask: bool = True
    masking_strategy: str = "entropy_min"
    masking_prob: float = 0.3
    masking_tau: float = 0.4
    
    # Consistency loss parameters
    consistency_loss_coeff: float = 0.1
    consistency_ramp_epochs: int = 5
    
    # Feature processing
    feature_normalization: bool = True
    
    # Loss function parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    entropy_reg: bool = True
    entropy_max_lambda: float = 0.1
    consistency_coeff: float = 0.1
    consistency_ramp_epochs: int = 5
    label_frequencies: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_core_params()
        self._validate_gating_params()
        self._validate_training_params()
        self._validate_regularization_params()
        self._validate_masking_params()
        self._validate_loss_params()
    
    def _validate_core_params(self):
        """Validate core model parameters."""
        # Validate modalities
        valid_modalities = {"image", "text", "audio"}
        if not self.modalities:
            raise ValueError("At least one modality must be specified")
        
        invalid_modalities = set(self.modalities) - valid_modalities
        if invalid_modalities:
            raise ValueError(f"Invalid modalities: {invalid_modalities}. "
                           f"Valid options: {valid_modalities}")
        
        # Validate dimensions
        if self.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {self.feat_dim}")
        
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        # Validate task type
        valid_task_types = {"classification", "regression", "embedding"}
        if self.task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type: {self.task_type}. "
                           f"Valid options: {valid_task_types}")
    
    def _validate_gating_params(self):
        """Validate gating network parameters."""
        if not self.gate_hidden_dims:
            raise ValueError("gate_hidden_dims cannot be empty")
        
        if any(dim <= 0 for dim in self.gate_hidden_dims):
            raise ValueError("All gate_hidden_dims must be positive")
        
        if self.gate_init_temp <= 0:
            raise ValueError(f"gate_init_temp must be positive, got {self.gate_init_temp}")
        
        if self.gate_min_temp <= 0:
            raise ValueError(f"gate_min_temp must be positive, got {self.gate_min_temp}")
        
        if self.gate_min_temp > self.gate_init_temp:
            raise ValueError("gate_min_temp cannot be greater than gate_init_temp")
    
    def _validate_training_params(self):
        """Validate training parameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.gate_learning_rate <= 0:
            raise ValueError(f"gate_learning_rate must be positive, got {self.gate_learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        
        if self.gradient_clip_val <= 0:
            raise ValueError(f"gradient_clip_val must be positive, got {self.gradient_clip_val}")
        
        if not 0 < self.val_check_interval <= 1:
            raise ValueError(f"val_check_interval must be in (0, 1], got {self.val_check_interval}")
        
        if self.early_stopping_patience < 0:
            raise ValueError(f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}")
        
        if self.early_stopping_min_delta < 0:
            raise ValueError(f"early_stopping_min_delta must be non-negative, got {self.early_stopping_min_delta}")
    
    def _validate_regularization_params(self):
        """Validate regularization parameters."""
        if self.entropy_free_epochs < 0:
            raise ValueError(f"entropy_free_epochs must be non-negative, got {self.entropy_free_epochs}")
        
        if self.entropy_warmup_epochs < 0:
            raise ValueError(f"entropy_warmup_epochs must be non-negative, got {self.entropy_warmup_epochs}")
        
        if self.entropy_max_coeff < 0:
            raise ValueError(f"entropy_max_coeff must be non-negative, got {self.entropy_max_coeff}")
        
        if self.ece_penalty_coeff < 0:
            raise ValueError(f"ece_penalty_coeff must be non-negative, got {self.ece_penalty_coeff}")
        
        if self.l2_logits_coeff < 0:
            raise ValueError(f"l2_logits_coeff must be non-negative, got {self.l2_logits_coeff}")
    
    def _validate_masking_params(self):
        """Validate curriculum masking parameters."""
        valid_strategies = {"none", "random", "weighted_random", "entropy_min", "entropy_max"}
        if self.masking_strategy not in valid_strategies:
            raise ValueError(f"Invalid masking_strategy: {self.masking_strategy}. "
                           f"Valid options: {valid_strategies}")
        
        if not 0 <= self.masking_prob <= 1:
            raise ValueError(f"masking_prob must be in [0, 1], got {self.masking_prob}")
        
        if self.masking_tau <= 0:
            raise ValueError(f"masking_tau must be positive, got {self.masking_tau}")
        
        if self.consistency_loss_coeff < 0:
            raise ValueError(f"consistency_loss_coeff must be non-negative, got {self.consistency_loss_coeff}")
        
        if self.consistency_ramp_epochs < 0:
            raise ValueError(f"consistency_ramp_epochs must be non-negative, got {self.consistency_ramp_epochs}")
    
    def _validate_loss_params(self):
        """Validate loss function parameters."""
        if self.focal_alpha < 0:
            raise ValueError(f"focal_alpha must be non-negative, got {self.focal_alpha}")
        
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be non-negative, got {self.focal_gamma}")
        
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        
        if self.consistency_coeff < 0:
            raise ValueError(f"consistency_coeff must be non-negative, got {self.consistency_coeff}")
        
        if self.consistency_ramp_epochs < 0:
            raise ValueError(f"consistency_ramp_epochs must be non-negative, got {self.consistency_ramp_epochs}")
        
        if self.label_frequencies is not None:
            if not all(f >= 0 for f in self.label_frequencies):
                raise ValueError("All label frequencies must be non-negative")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AECFConfig':
        """
        Create AECFConfig from dictionary with validation.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration parameters
            
        Returns
        -------
        AECFConfig
            Validated configuration object
            
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Legacy key mapping
        legacy_mapping = {
            'lr': 'learning_rate',
            'gate_lr': 'gate_learning_rate',
            'wd': 'weight_decay',
            'gate_hidden': 'gate_hidden_dims',
            'entropy_free': 'entropy_free_epochs',
            'entropy_warmup': 'entropy_warmup_epochs',
            'entropy_max': 'entropy_max_coeff',
            'masking_mode': 'masking_strategy',
            'p_missing': 'masking_prob',
            'tau': 'masking_tau',
            'cec_coef': 'consistency_loss_coeff',
            'cec_ramp_epochs': 'consistency_ramp_epochs',
            'feature_norm': 'feature_normalization'
        }
        
        # Convert legacy keys
        converted_dict = {}
        for key, value in config_dict.items():
            new_key = legacy_mapping.get(key, key)
            converted_dict[new_key] = value
        
        # Handle special conversions
        if 'gate_hidden_dims' in converted_dict:
            gate_hidden = converted_dict['gate_hidden_dims']
            if isinstance(gate_hidden, int):
                converted_dict['gate_hidden_dims'] = [gate_hidden]
        
        # Filter known parameters
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in converted_dict.items() if k in valid_fields}
        
        # Log any ignored parameters
        ignored = set(config_dict.keys()) - set([legacy_mapping.get(k, k) for k in config_dict.keys()])
        if ignored:
            logging.warning(f"Ignoring unknown configuration parameters: {ignored}")
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary
        """
        return asdict(self)
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format for backward compatibility.
        
        Returns
        -------
        Dict[str, Any]
            Configuration in legacy format
        """
        legacy_mapping = {
            # Core parameters
            "modalities": self.modalities,
            "feat_dim": self.feat_dim,
            "num_classes": self.num_classes,
            "task_type": self.task_type,
            
            # Gating parameters
            "gate_hidden": self.gate_hidden_dims[0] if len(self.gate_hidden_dims) == 1 else self.gate_hidden_dims,
            "gate_use_softmax": self.gate_use_softmax,
            "gate_learnable_temp": self.gate_learnable_temp,
            "gate_init_temp": self.gate_init_temp,
            "gate_min_temp": self.gate_min_temp,
            
            # Training parameters
            "lr": self.learning_rate,
            "gate_lr": self.gate_learning_rate,
            "wd": self.weight_decay,
            "epochs": self.epochs,
            
            # Regularization
            "entropy_free": self.entropy_free_epochs,
            "entropy_warmup": self.entropy_warmup_epochs,
            "entropy_max": self.entropy_max_coeff,
            
            # Masking
            "masking_mode": self.masking_strategy,
            "p_missing": self.masking_prob,
            "tau": self.masking_tau,
            
            # Consistency loss
            "cec_coef": self.consistency_loss_coeff,
            "cec_ramp_epochs": self.consistency_ramp_epochs,
            
            # Feature processing
            "feature_norm": self.feature_normalization,
        }
        
        return legacy_mapping
    
    def update(self, **kwargs) -> 'AECFConfig':
        """
        Create a new configuration with updated parameters.
        
        Parameters
        ----------
        **kwargs
            Parameters to update
            
        Returns
        -------
        AECFConfig
            New configuration with updated parameters
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["AECFConfig:"]
        lines.append(f"  Modalities: {self.modalities}")
        lines.append(f"  Task: {self.task_type} ({self.num_classes} classes)")
        lines.append(f"  Features: {self.feat_dim}D")
        lines.append(f"  Gating: {'disabled' if self.gate_disabled else 'enabled'}")
        lines.append(f"  Masking: {self.masking_strategy}")
        lines.append(f"  Learning rates: {self.learning_rate} (main), {self.gate_learning_rate} (gate)")
        return "\n".join(lines)


def create_default_config(
    task_type: str = "classification",
    num_classes: int = 80,
    modalities: Optional[List[str]] = None
) -> AECFConfig:
    """
    Create default configuration for common use cases.
    
    Parameters
    ----------
    task_type : str
        Type of task
    num_classes : int
        Number of output classes
    modalities : List[str], optional
        List of modalities to use
        
    Returns
    -------
    AECFConfig
        Default configuration
    """
    if modalities is None:
        modalities = ["image", "text"]
    
    return AECFConfig(
        task_type=task_type,
        num_classes=num_classes,
        modalities=modalities
    )


def validate_config_compatibility(config: AECFConfig, data_config: Dict[str, Any]) -> None:
    """
    Validate that model configuration is compatible with data configuration.
    
    Parameters
    ----------
    config : AECFConfig
        Model configuration
    data_config : Dict[str, Any]
        Data configuration
        
    Raises
    ------
    ValueError
        If configurations are incompatible
    """
    # Check modalities match
    data_modalities = set(data_config.get("modalities", []))
    model_modalities = set(config.modalities)
    
    if not model_modalities.issubset(data_modalities):
        missing = model_modalities - data_modalities
        raise ValueError(f"Model requires modalities not in data: {missing}")
    
    # Check feature dimensions
    data_feat_dim = data_config.get("feat_dim", 512)
    if config.feat_dim != data_feat_dim:
        raise ValueError(f"Feature dimension mismatch: model={config.feat_dim}, data={data_feat_dim}")
    
    # Check number of classes for classification
    if config.task_type == "classification":
        data_num_classes = data_config.get("num_classes", 80)
        if config.num_classes != data_num_classes:
            raise ValueError(f"Number of classes mismatch: model={config.num_classes}, data={data_num_classes}")


# Legacy support functions
def validate_config(config: AECFConfig) -> None:
    """Legacy function - validation now happens in __post_init__."""
    pass  # Validation is now automatic


def create_config_from_dict(config_dict: Dict[str, Any]) -> AECFConfig:
    """Legacy function - use AECFConfig.from_dict() instead."""
    return AECFConfig.from_dict(config_dict)


def config_to_legacy_dict(config: AECFConfig) -> Dict[str, Any]:
    """Legacy function - use config.to_legacy_dict() instead."""
    return config.to_legacy_dict()

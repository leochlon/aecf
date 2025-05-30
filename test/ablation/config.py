"""
Ablation configuration dataclass and validation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


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
    
    # Training hyperparameters - optimized for A100 GPU utilization
    batch_size: int = 256  # Increased for A100 40GB GPU
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

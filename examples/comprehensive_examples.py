"""
AECF Usage Examples

This file demonstrates how to use the new modular AECF architecture
for different tasks and configurations.
"""

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import logging

# Import the refactored AECF components
from aecf import (
    AECF_CLIP, AECFConfig, AECFTrainer, create_aecf_model, create_default_config,
    validate_model_inputs, test_model_forward
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_usage_example():
    """Basic usage example with default configuration."""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Create model with default configuration
    model = create_aecf_model(
        modalities=["image", "text"],
        task_type="classification",
        num_classes=80
    )
    
    print(f"Created model: {model.get_model_info()}")
    
    # Test forward pass
    test_results = test_model_forward(model, batch_size=8)
    print(f"Forward pass test: {test_results}")
    
    return model


def custom_configuration_example():
    """Example with custom configuration."""
    print("=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    # Create custom configuration
    config = AECFConfig(
        modalities=["image", "text", "audio"],
        task_type="regression",
        num_classes=1,
        feat_dim=768,
        gate_hidden_dims=[1024, 512],
        learning_rate=5e-4,
        gate_learning_rate=1e-2,
        entropy_max_coeff=0.2,
        masking_strategy="entropy_min",
        masking_prob=0.4,
        epochs=50
    )
    
    print(f"Custom config:\n{config}")
    
    # Create model with custom config
    model = AECF_CLIP(config)
    
    # Test model
    test_results = test_model_forward(model, batch_size=4)
    print(f"Custom model test: {test_results}")
    
    return model, config


def training_example():
    """Complete training example."""
    print("=" * 60)
    print("TRAINING EXAMPLE")
    print("=" * 60)
    
    # Create configuration for training
    config = create_default_config(
        task_type="classification",
        num_classes=80,
        modalities=["image", "text"]
    )
    
    # Update training-specific settings
    config = config.update(
        epochs=5,  # Short for example
        learning_rate=1e-4,
        gradient_clip_val=0.5,
        mixed_precision=True,
        early_stopping_patience=3
    )
    
    print(f"Training config:\n{config}")
    
    # Create trainer
    trainer = AECFTrainer(
        config=config,
        logger_type="tensorboard",
        logger_config={"version": "example_run"}
    )
    
    # Create dummy dataset
    dummy_dataset = DummyMultiModalDataset(
        num_samples=100,
        feat_dim=config.feat_dim,
        num_classes=config.num_classes,
        modalities=config.modalities
    )
    
    train_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=False)
    
    print("Created data loaders")
    
    # Note: In practice, you would call trainer.fit(train_loader, val_loader)
    # For this example, we'll just setup the trainer
    trainer.setup_trainer()
    print("Training setup complete")
    
    return trainer


def legacy_compatibility_example():
    """Example showing legacy compatibility."""
    print("=" * 60)
    print("LEGACY COMPATIBILITY EXAMPLE")
    print("=" * 60)
    
    # Legacy dict-based configuration
    legacy_config = {
        "modalities": ["image", "text"],
        "feat_dim": 512,
        "num_classes": 80,
        "task_type": "classification",
        "lr": 1e-4,
        "gate_lr": 1e-3,
        "wd": 1e-2,
        "gate_hidden": 2048,
        "entropy_free": 0,
        "entropy_warmup": 5,
        "entropy_max": 0.1,
        "masking_mode": "entropy_min",
        "p_missing": 0.3,
        "tau": 0.4
    }
    
    # Convert to new configuration format
    new_config = AECFConfig.from_dict(legacy_config)
    print(f"Converted config:\n{new_config}")
    
    # Create model with converted config
    model = AECF_CLIP(new_config)
    
    # Convert back to legacy format if needed
    legacy_dict = new_config.to_legacy_dict()
    print(f"Back to legacy format: {list(legacy_dict.keys())}")
    
    return model, new_config


def component_validation_example():
    """Example showing input validation and error handling."""
    print("=" * 60)
    print("COMPONENT VALIDATION EXAMPLE")
    print("=" * 60)
    
    config = create_default_config()
    model = AECF_CLIP(config)
    
    # Valid input
    valid_features = {
        "image": torch.randn(4, 512),
        "text": torch.randn(4, 512)
    }
    
    try:
        validate_model_inputs(valid_features, config.modalities, config.feat_dim)
        print("✓ Valid input passed validation")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid input - wrong dimension
    invalid_features = {
        "image": torch.randn(4, 256),  # Wrong dimension
        "text": torch.randn(4, 512)
    }
    
    try:
        validate_model_inputs(invalid_features, config.modalities, config.feat_dim)
        print("✗ Invalid input should have failed validation")
    except Exception as e:
        print(f"✓ Correctly caught validation error: {e}")
    
    # Invalid input - missing modality
    missing_modality = {
        "image": torch.randn(4, 512)
        # Missing "text" modality
    }
    
    try:
        validate_model_inputs(missing_modality, config.modalities, config.feat_dim)
        print("✗ Missing modality should have failed validation")
    except Exception as e:
        print(f"✓ Correctly caught missing modality: {e}")


def memory_efficient_example():
    """Example showing memory-efficient training."""
    print("=" * 60)
    print("MEMORY EFFICIENT EXAMPLE")
    print("=" * 60)
    
    # Configuration for memory efficiency
    config = AECFConfig(
        modalities=["image", "text"],
        task_type="classification",
        num_classes=80,
        mixed_precision=True,  # Use mixed precision
        gradient_clip_val=1.0,  # Prevent gradient explosion
        # Memory-efficient settings would be set in the trainer
    )
    
    model = AECF_CLIP(config)
    
    # The MetricsAccumulator is used internally to prevent memory issues
    from aecf import MetricsAccumulator
    
    metrics_acc = MetricsAccumulator()
    
    # Simulate batch processing
    for i in range(5):
        dummy_metrics = {
            "loss": torch.tensor(0.5 + i * 0.1),
            "accuracy": torch.tensor(0.8 - i * 0.05),
            "entropy": torch.tensor(1.2 + i * 0.1)
        }
        metrics_acc.update(dummy_metrics)
    
    final_metrics = metrics_acc.compute()
    print(f"Accumulated metrics: {final_metrics}")
    
    return model, metrics_acc


class DummyMultiModalDataset(data.Dataset):
    """Dummy dataset for examples and testing."""
    
    def __init__(self, num_samples=100, feat_dim=512, num_classes=80, modalities=None):
        self.num_samples = num_samples
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.modalities = modalities or ["image", "text"]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = {}
        
        # Generate random features for each modality
        for modality in self.modalities:
            sample[modality] = torch.randn(self.feat_dim)
        
        # Generate random multi-label target
        sample["label"] = torch.randint(0, 2, (self.num_classes,)).float()
        
        return sample


def run_all_examples():
    """Run all examples."""
    print("Running AECF Usage Examples")
    print("=" * 80)
    
    try:
        # Basic usage
        basic_model = basic_usage_example()
        
        # Custom configuration
        custom_model, custom_config = custom_configuration_example()
        
        # Training setup
        trainer = training_example()
        
        # Legacy compatibility
        legacy_model, legacy_config = legacy_compatibility_example()
        
        # Validation
        component_validation_example()
        
        # Memory efficiency
        memory_model, metrics_acc = memory_efficient_example()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return {
            "basic_model": basic_model,
            "custom_model": custom_model,
            "trainer": trainer,
            "legacy_model": legacy_model,
            "memory_model": memory_model
        }
        
    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_all_examples()
    
    if results:
        print("\nExample results:")
        for name, obj in results.items():
            if hasattr(obj, 'get_model_info'):
                info = obj.get_model_info()
                print(f"  {name}: {info['total_parameters']} parameters")
            else:
                print(f"  {name}: {type(obj).__name__}")

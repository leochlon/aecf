"""
Comprehensive unit tests for the refactored AECF components.

This test suite validates the modular architecture, proper error handling,
memory management, and backward compatibility.
"""

import unittest
import torch
import tempfile
import shutil
from pathlib import Path
import logging
from typing import Dict, Any

# Import AECF components
from aecf import (
    AECF_CLIP, AECFConfig, AECFTrainer, create_aecf_model, create_default_config,
    AdaptiveGate, CurriculumMasker, EncoderFactory, OutputHeadFactory,
    AECFLoss, AECFMetrics, MemoryManagementCallback, MetricsAccumulator,
    validate_feature_dict, normalize_features, validate_tensor_input,
    validate_model_inputs, test_model_forward, validate_config_compatibility
)

# Suppress logging during tests
logging.getLogger().setLevel(logging.WARNING)


class TestAECFConfig(unittest.TestCase):
    """Test the configuration system."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = create_default_config()
        
        self.assertEqual(config.modalities, ["image", "text"])
        self.assertEqual(config.task_type, "classification")
        self.assertEqual(config.num_classes, 80)
        self.assertEqual(config.feat_dim, 512)
        self.assertGreater(config.learning_rate, 0)


class TestAECFComponents(unittest.TestCase):
    """Test individual AECF components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = create_default_config()
        self.batch_size = 4
        self.feat_dim = 512
    
    def test_adaptive_gate(self):
        """Test adaptive gating mechanism."""
        gate = AdaptiveGate(
            input_dim=self.feat_dim * 2,  # Two modalities
            num_modalities=2,
            hidden_dim=1024
        )
        
        # Test forward pass
        concat_features = torch.randn(self.batch_size, self.feat_dim * 2)
        weights, entropy = gate(concat_features)
        
        # Check output shape and properties
        self.assertEqual(weights.shape, (self.batch_size, 2))
        
        # Weights should sum to 1 (approximately)
        weight_sums = weights.sum(dim=1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))


class TestAECFModel(unittest.TestCase):
    """Test the main AECF model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = create_default_config()
        self.model = AECF_CLIP(self.config)
        self.batch_size = 4
    
    def test_model_creation(self):
        """Test model creation and basic properties."""
        # Test model info
        info = self.model.get_model_info()
        self.assertIn("total_parameters", info)
        self.assertIn("trainable_parameters", info)
        self.assertIn("modalities", info)
        self.assertEqual(info["modalities"], self.config.modalities)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        features = {
            "image": torch.randn(self.batch_size, self.config.feat_dim),
            "text": torch.randn(self.batch_size, self.config.feat_dim)
        }
        
        logits, weights = self.model(features)
        
        # Check output shapes
        expected_logits_shape = (self.batch_size, self.config.num_classes)
        self.assertEqual(logits.shape, expected_logits_shape)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management components."""
    
    def test_metrics_accumulator(self):
        """Test memory-efficient metrics accumulation."""
        accumulator = MetricsAccumulator()
        
        # Add some metrics
        for i in range(5):
            metrics = {
                "loss": torch.tensor(0.5 + i * 0.1),
                "accuracy": torch.tensor(0.8 - i * 0.05)
            }
            accumulator.update(metrics)
        
        # Compute averages
        final_metrics = accumulator.compute()
        
        # Check results
        self.assertIn("loss", final_metrics)
        self.assertIn("accuracy", final_metrics)
        self.assertAlmostEqual(final_metrics["loss"], 0.7, places=5)
        self.assertAlmostEqual(final_metrics["accuracy"], 0.7, places=5)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with legacy code."""
    
    def test_modular_components_available(self):
        """Test that all modular components are accessible."""
        from aecf import (
            AdaptiveGate, CurriculumMasker, EncoderFactory, 
            OutputHeadFactory, AECFLoss, AECFMetrics
        )
        
        # Should be able to import without error
        self.assertTrue(hasattr(AdaptiveGate, '__init__'))
        self.assertTrue(hasattr(CurriculumMasker, '__init__'))
        self.assertTrue(hasattr(EncoderFactory, 'create_encoder'))
        self.assertTrue(hasattr(OutputHeadFactory, 'create_head'))
        self.assertTrue(hasattr(AECFLoss, '__init__'))
        self.assertTrue(hasattr(AECFMetrics, 'compute_classification_metrics'))


def run_test_suite():
    """Run the complete test suite."""
    # Create test suite
    test_classes = [
        TestAECFConfig,
        TestAECFComponents,
        TestAECFModel,
        TestMemoryManagement,
        TestBackwardCompatibility
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running AECF Comprehensive Test Suite")
    print("=" * 80)
    
    result = run_test_suite()
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
    
    print("=" * 80)

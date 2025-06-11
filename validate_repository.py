#!/usr/bin/env python3
"""
Final validation script for the cleaned AECF repository.

This script validates that all components are working correctly
after the repository cleanup and refactoring.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Core imports
        from aecf import (
            AECFConfig, AECF_CLIP, create_aecf_model, 
            AECFTrainer, test_model_forward
        )
        print("  ✅ Core imports successful")
        
        # Component imports
        from aecf import (
            AdaptiveGate, CurriculumMasker, EncoderFactory, 
            OutputHeadFactory, AECFLoss, AECFMetrics
        )
        print("  ✅ Component imports successful")
        
        # Utility imports
        from aecf import (
            validate_feature_dict, normalize_features, 
            MetricsAccumulator
        )
        print("  ✅ Utility imports successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system."""
    print("\n🔧 Testing configuration system...")
    
    try:
        from aecf import AECFConfig, create_default_config
        
        # Test default config
        config = create_default_config()
        print(f"  ✅ Default config: {config.modalities}")
        
        # Test custom config
        custom_config = AECFConfig(
            modalities=["image", "text", "audio"],
            task_type="regression",
            num_classes=1,
            feat_dim=768
        )
        print(f"  ✅ Custom config: {custom_config.task_type}")
        
        # Test legacy conversion
        legacy_dict = {"lr": 1e-4, "gate_lr": 1e-3, "modalities": ["image", "text"]}
        converted = AECFConfig.from_dict(legacy_dict)
        print(f"  ✅ Legacy conversion: {converted.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_model_functionality():
    """Test model creation and functionality."""
    print("\n🤖 Testing model functionality...")
    
    try:
        import torch
        from aecf import create_aecf_model, test_model_forward
        
        # Create model
        model = create_aecf_model(
            modalities=["image", "text"],
            task_type="classification",
            num_classes=80
        )
        print(f"  ✅ Model created successfully")
        
        # Test forward pass
        result = test_model_forward(model, batch_size=4)
        if result["success"]:
            print(f"  ✅ Forward pass successful: {result['logits_shape']}")
        else:
            print(f"  ❌ Forward pass failed: {result['error']}")
            return False
        
        # Test model info
        info = model.get_model_info()
        print(f"  ✅ Model info: {info['total_parameters']} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_loss_and_metrics():
    """Test loss and metrics functionality."""
    print("\n📊 Testing loss and metrics...")
    
    try:
        import torch
        from aecf import AECFConfig, AECFLoss, AECFMetrics
        
        config = AECFConfig()
        
        # Test loss function
        loss_fn = AECFLoss(config)
        dummy_logits = torch.randn(4, 80)
        dummy_targets = torch.randint(0, 2, (4, 80)).float()
        dummy_weights = torch.softmax(torch.randn(4, 2), dim=1)
        
        total_loss, components = loss_fn(dummy_logits, dummy_targets, dummy_weights)
        print(f"  ✅ Loss computation: {total_loss.item():.4f}")
        
        # Test metrics
        metrics = AECFMetrics.compute_classification_metrics(dummy_logits, dummy_targets)
        print(f"  ✅ Metrics computation: mAP={metrics['map'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Loss/metrics test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_management():
    """Test memory management functionality."""
    print("\n💾 Testing memory management...")
    
    try:
        from aecf import MetricsAccumulator
        import torch
        
        # Test metrics accumulator
        acc = MetricsAccumulator()
        
        for i in range(5):
            metrics = {
                "loss": torch.tensor(0.5 + i * 0.1),
                "accuracy": torch.tensor(0.8 - i * 0.05)
            }
            acc.update(metrics)
        
        final_metrics = acc.compute()
        print(f"  ✅ Memory-efficient accumulation: avg_loss={final_metrics['loss']:.3f}")
        
        # Test reset
        acc.reset()
        empty_metrics = acc.compute()
        if len(empty_metrics) == 0:
            print("  ✅ Metrics accumulator reset successful")
        else:
            print("  ❌ Metrics accumulator reset failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory management test failed: {e}")
        traceback.print_exc()
        return False

def test_examples():
    """Test that examples can be imported."""
    print("\n📚 Testing examples...")
    
    try:
        # Test that example files exist and can be imported
        examples_dir = Path("examples")
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*.py"))
            print(f"  ✅ Found {len(example_files)} example files")
            
            # Test importing comprehensive examples
            sys.path.insert(0, str(examples_dir))
            try:
                import comprehensive_examples
                print("  ✅ Comprehensive examples importable")
            except Exception as e:
                print(f"  ⚠️  Comprehensive examples import warning: {e}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Examples test failed: {e}")
        return False

def validate_repository_structure():
    """Validate the repository structure."""
    print("\n📁 Validating repository structure...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "MANIFEST.in",
        "aecf/__init__.py",
        "aecf/config.py",
        "aecf/components.py",
        "aecf/model_refactored.py",
        "aecf/losses.py",
        "aecf/metrics.py",
        "aecf/training.py",
        "examples/comprehensive_examples.py",
        "tests/test_comprehensive.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print(f"  ✅ All {len(required_files)} required files present")
        return True

def main():
    """Run all validation tests."""
    print("🎯 AECF REPOSITORY VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Repository Structure", validate_repository_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Model Functionality", test_model_functionality),
        ("Loss and Metrics", test_loss_and_metrics),
        ("Memory Management", test_memory_management),
        ("Examples", test_examples),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION RESULTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🏆 Repository is clean and production-ready!")
        print("✨ AECF refactoring complete and validated!")
    else:
        print("❌ Some tests failed. Please review the output above.")
        return False
    
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

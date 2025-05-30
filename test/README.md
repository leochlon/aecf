# AECF Ablation Suite - Modularized Structure

This directory contains the refactored and modularized version of the AECF ablation study suite. The original 1595-line `cocoAblation.py` file has been broken down into logical, maintainable modules while preserving all original functionality.

## 🏗️ Architecture Overview

The codebase is organized into the following modules:

### 📁 Core Modules

- **`ablation/`** - Core ablation experiment functionality
  - `config.py` - AblationConfig dataclass with experiment parameters
  - `experiment.py` - Single experiment execution logic
  - `suite.py` - Main orchestrator for running multiple ablations
  - `convenience.py` - Helper functions for common use cases

- **`data/`** - Data management and validation
  - `manager.py` - COCODataManager for COCO dataset handling
  - `manifest.py` - DatasetManifest and integrity checking
  - `legacy.py` - Backward compatibility (deprecated DataManager)

- **`analysis/`** - Results analysis and reporting
  - `analyzer.py` - ResultsAnalyzer for post-experiment analysis

- **`ui/`** - User interface components
  - `progress.py` - CleanProgressBar for training progress

- **`utils/`** - Utilities and setup
  - `setup.py` - Environment setup, GPU optimizations, logging
  - `constants.py` - Default configurations and constants

### 🚀 Main Entry Point

- **`cocoAblation.py`** - Main command-line interface (replaces original file)

## 📋 Usage

The modularized version maintains the exact same command-line interface as the original:

```bash
# Show help
python cocoAblation.py --help

# Run all ablations (default)
python cocoAblation.py

# Run specific ablations
python cocoAblation.py --ablations full no_gate

# Quick mode (80 epochs instead of 200)
python cocoAblation.py --quick

# Custom output directory
python cocoAblation.py --output-dir ./my_results

# Parallel execution (experimental)
python cocoAblation.py --parallel
```

### Available Ablations

- `full` - Complete AECF model (baseline)
- `no_gate` - Without gating mechanism
- `no_entropy` - Without entropy regularization
- `no_curmask` - Without curriculum masking
- `img_only` - Image modality only
- `txt_only` - Text modality only

## 🧩 Module Details

### AblationConfig

```python
from ablation.config import AblationConfig

# Create a custom configuration
config = AblationConfig(
    name="custom_experiment",
    gate_disabled=True,
    max_epochs=80,
    batch_size=256
)
```

### AblationSuite

```python
from ablation.suite import AblationSuite
from pathlib import Path

# Create and run suite
suite = AblationSuite(
    data_root=Path("/path/to/coco2014"),
    output_dir=Path("./results")
)

# Run specific ablations
results = suite.run_ablations(ablation_names=["full", "no_gate"])

# Run with custom configurations
custom_configs = {
    "custom": AblationConfig(name="custom", max_epochs=80)
}
results = suite.run_ablations(custom_configs=custom_configs)
```

### Convenience Functions

```python
from ablation.convenience import run_quick_ablation, run_full_ablation_suite

# Quick ablation (80 epochs)
results = run_quick_ablation(["full", "no_gate"])

# Full suite (all ablations)
results = run_full_ablation_suite()
```

## 🔧 Environment Setup

The suite automatically handles:
- GPU optimizations (TF32, mixed precision)
- Logging suppression for cleaner output
- Memory optimization for A100 GPUs
- CUDA availability detection

## 📊 Data Management

The modularized data manager provides:
- Automatic COCO dataset validation
- Cache management and integrity checking
- Fallback to dummy data for testing
- Comprehensive error handling

## 🧪 Testing and Validation

All functionality has been preserved from the original implementation:

```bash
# Test all imports
python -c "from ablation.suite import AblationSuite; print('✅ All imports work')"

# Test suite creation
python -c "from ablation.suite import AblationSuite; s = AblationSuite(); print('✅ Suite created')"

# Test CLI
python cocoAblation.py --help
```

## 📈 Benefits of Modularization

1. **Maintainability** - Code is organized into logical, single-responsibility modules
2. **Testability** - Individual components can be tested in isolation
3. **Reusability** - Components can be imported and used in other projects
4. **Extensibility** - New ablations and features can be added easily
5. **Documentation** - Clear module boundaries make the codebase self-documenting

## 🔄 Migration from Original

If you were using the original `cocoAblation.py`:

1. **No changes needed** - The command-line interface is identical
2. **Import changes** - If importing functions, update imports:
   ```python
   # Old
   from cocoAblation import run_quick_ablation
   
   # New
   from ablation.convenience import run_quick_ablation
   ```

## 🐛 Troubleshooting

Common issues and solutions:

1. **Import errors** - Ensure you're running from the `test/` directory
2. **CUDA warnings** - Normal on CPU-only systems
3. **Data not found** - Suite will create dummy cache for testing

## 📚 Original Functionality Preserved

All original features are preserved:
- ✅ All 6 standard ablations
- ✅ Quick mode (80 epochs)
- ✅ Parallel execution support
- ✅ Comprehensive logging
- ✅ Results analysis and reporting
- ✅ Data validation and integrity checking
- ✅ GPU optimization and memory management
- ✅ Colab/Jupyter notebook compatibility

## 🎯 Next Steps

The modularized codebase is ready for:
- Adding new ablation configurations
- Implementing new analysis methods
- Integration with different datasets
- Extension to other model architectures
- Deployment in production environments

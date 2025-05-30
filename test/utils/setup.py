"""
Environment setup and configuration for COCO Ablation Suite.
"""
import os
import logging
import warnings
import torch

def setup_environment():
    """Setup environment variables and optimizations."""
    # Optimize GPU utilization settings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async CUDA operations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

    # Suppress PyTorch Lightning verbose logging
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*The number of training batches.*")
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")

    # Configure logging levels to reduce verbosity
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    # Suppress all AECF_CLIP logs (comprehensive suppression)
    aecf_logger = logging.getLogger("AECF_CLIP")
    aecf_logger.setLevel(logging.WARNING)
    aecf_logger.propagate = False
    for handler in list(aecf_logger.handlers):
        aecf_logger.removeHandler(handler)

    # Optionally suppress all root logs below WARNING
    logging.getLogger().setLevel(logging.WARNING)

    # Set tensor core precision for better performance
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training


def setup_logging(output_dir, name="ablation_suite"):
    """Setup logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.FileHandler(output_dir / f"{name}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def silent_get_logger(name: str):
    """Override the get_logger function in the model module to respect our suppression."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)  # Always suppress INFO and DEBUG
    logger.propagate = False
    # Remove all handlers to prevent any output
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    return logger


def patch_aecf_logging():
    """Patch the get_logger function in the model module to enforce our logging suppression."""
    try:
        import aecf.model
        aecf.model.get_logger = silent_get_logger
    except ImportError:
        pass  # Module not available, skip patching

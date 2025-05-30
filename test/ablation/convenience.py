"""
Convenience functions for common ablation use cases.
"""
import pandas as pd
from .suite import AblationSuite


def run_quick_ablation(ablations: list = ["full", "no_gate"]) -> pd.DataFrame:
    """Run a quick ablation with 80 epochs for comprehensive testing."""
    suite = AblationSuite()
    
    # Create quick configs optimized for A100
    quick_configs = {}
    for name in ablations:
        config = suite.STANDARD_ABLATIONS[name]
        config.max_epochs = 80  # Changed from 5 to 80 epochs
        config.batch_size = 256  # Larger batches for A100
        config.patience = 15  # Increased patience for longer training
        quick_configs[name] = config
    
    return suite.run_ablations(custom_configs=quick_configs)


def run_full_ablation_suite() -> pd.DataFrame:
    """Run the complete ablation suite."""
    suite = AblationSuite()
    return suite.run_ablations()

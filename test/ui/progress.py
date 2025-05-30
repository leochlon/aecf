"""
Custom progress bar for clean training output.
"""
import torch
from pytorch_lightning.callbacks import TQDMProgressBar


class CleanProgressBar(TQDMProgressBar):
    """Custom progress bar showing only epoch-level progress with key metrics."""
    
    def __init__(self):
        super().__init__()
        self.max_epochs = None
        # Initialize progress bar references to prevent errors
        self._train_progress_bar = None
        self._val_progress_bar = None
        self._test_progress_bar = None
        self._predict_progress_bar = None
    
    def init_train_tqdm(self):
        """Initialize epoch-level progress bar instead of batch-level."""
        # Store max_epochs for later use
        if hasattr(self.trainer, 'max_epochs'):
            self.max_epochs = self.trainer.max_epochs
            
        # Create a simple print-based progress tracker instead of tqdm
        print(f"Starting training for {self.max_epochs} epochs...")
        
        # Set the internal reference to None to prevent errors
        self._train_progress_bar = None
        return None  # Don't return batch-level bar
    
    def init_validation_tqdm(self):
        """Disable validation progress bar."""
        self._val_progress_bar = None
        return None
    
    def init_predict_tqdm(self):
        """Disable prediction progress bar.""" 
        self._predict_progress_bar = None
        return None
    
    def init_test_tqdm(self):
        """Disable test progress bar."""
        self._test_progress_bar = None
        return None
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Print epoch progress at start of each epoch."""
        current_epoch = trainer.current_epoch + 1
        print(f"\rEpoch {current_epoch}/{self.max_epochs}", end="", flush=True)
    
    def on_validation_end(self, trainer, pl_module):
        """Print metrics after validation."""
        if trainer.logged_metrics:
            # Extract key metrics
            metrics = []
            for key, value in trainer.logged_metrics.items():
                if any(metric in key.lower() for metric in ['val_loss', 'val_ece', 'val_map', 'h']):
                    if isinstance(value, torch.Tensor):
                        metrics.append(f"{key}={value.item():.4f}")
                    else:
                        val = value if isinstance(value, (int, float)) else str(value)
                        metrics.append(f"{key}={val:.4f}" if isinstance(val, (int, float)) else f"{key}={val}")
            
            # Print the epoch info with metrics on same line
            current_epoch = trainer.current_epoch + 1
            if metrics:
                metrics_str = ", ".join(metrics)
                print(f"\rEpoch {current_epoch}/{self.max_epochs} - {metrics_str}")
            else:
                print(f"\rEpoch {current_epoch}/{self.max_epochs} - Complete")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Epoch end - metrics already printed in on_validation_end."""
        pass
    
    def on_train_end(self, trainer, pl_module):
        """Training complete."""
        print("\nTraining completed.")
    
    def get_metrics(self, trainer, pl_module):
        """Override get_metrics to prevent reference errors."""
        return {}
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Override to prevent batch-level progress."""
        pass
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Override to prevent validation batch progress."""
        pass
    
    # Override properties to handle internal references
    @property
    def train_progress_bar(self):
        """Return None to prevent access errors."""
        return None
    
    @property
    def val_progress_bar(self):
        """Return None to prevent access errors."""
        return None
    
    @property
    def test_progress_bar(self):
        """Return None to prevent access errors."""
        return None
    
    @property
    def predict_progress_bar(self):
        """Return None to prevent access errors."""
        return None

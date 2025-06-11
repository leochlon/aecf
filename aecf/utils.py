"""
Lightweight metrics tracking utility for AECF.
"""

from typing import Dict, Any, Optional
import torch


class QuickMeter:
    """Lightweight metrics tracker with minimal overhead."""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update metrics with new values."""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] = (self.metrics[key] * self.counts[key] + value) / (self.counts[key] + 1)
            self.counts[key] += 1
    
    def get(self, key: str, default: float = 0.0) -> float:
        """Get metric value."""
        return self.metrics.get(key, default)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
    
    def summary(self) -> Dict[str, float]:
        """Get all metrics."""
        return self.metrics.copy()

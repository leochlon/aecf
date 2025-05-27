"""
Lightweight logging utility for metrics.
"""
import csv
from pathlib import Path

class QuickMeter:
    """Ultra-light accumulator for metrics with dynamic keys."""

    def __init__(self):
        self.rows = []

    def add(self, epoch: int, split: str, metrics: dict):
        """Store one row worth of metrics."""
        row = {"epoch": epoch, "split": split}
        # Cast tensors to Python scalars
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v
        self.rows.append(row)

    def save(self, fname: str | Path):
        """Dump all accumulated rows to a CSV."""
        if not self.rows:
            print("[QuickMeter] nothing to save")
            return

        # Full union of keys across all rows
        all_keys = set().union(*[r.keys() for r in self.rows])
        fieldnames = ["epoch", "split", *sorted(all_keys - {"epoch", "split"})]

        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        print(f"[QuickMeter] wrote {len(self.rows)} rows to {fname}")

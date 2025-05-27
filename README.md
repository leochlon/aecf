# AECF – Adaptive Ensemble CLIP Fusion

Official code for **"Adaptive Masking and Ensemble Fusion for Vision‑Language Recognition"** (NeurIPS 2025).

## Quick Start

```bash
# Clone repo
git clone https://github.com/your‑org/aecf-neurips-25.git
cd aecf-neurips-25

# Install dependencies
pip install -r requirements.txt

# Prepare COCO dataset
./scripts/fetch_coco.sh ~/coco2014

# Run training
python -m aecf.train --root ~/coco2014 --epochs 30 --gpus 1
```

## Repository Structure

```
aecf-neurips-25/
├── aecf/                     # Core AECF package
│   ├── __init__.py           # Package exports
│   ├── model.py              # AECF_CLIP model
│   ├── gating.py             # Gating network
│   ├── masking.py            # Masking strategies
│   ├── datasets.py           # Dataset handling (COCO, AVMNIST)
│   ├── encoders.py           # Modality encoders
│   ├── output_adapters.py    # Task-specific outputs
│   ├── utils.py              # Utility functions
│   └── train.py              # Training entry point
├── scripts/                  # Helper scripts
│   ├── fetch_coco.sh         # COCO dataset setup
│   ├── coco_ablation_suite.py# Ablation experiments
│   └── avmnist.py            # AVMNIST experiments
├── utils/                    # Utility modules
│   └── quick_meter.py        # Metric logging
└── requirements.txt          # Dependencies
```

## Available Tasks

- **COCO Multi-label Classification**: Run with `python -m aecf.train`
- **AVMNIST**: Run with `python scripts/avmnist.py`
- **Ablation Studies**: Run with `python scripts/coco_ablation_suite.py`

## Citation

```bibtex
@inproceedings{aecf2025,
  title={Adaptive Masking and Ensemble Fusion for Vision-Language Recognition},
  author={Author, A. and Author, B.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

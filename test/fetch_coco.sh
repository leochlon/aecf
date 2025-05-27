#!/usr/bin/env bash
# Script to download and setup COCO-2014 dataset
# Usage: ./scripts/fetch_coco.sh /path/to/target
set -e

ROOT=${1:-~/coco2014}
mkdir -p "$ROOT"
mkdir -p "$ROOT/annotations"

echo "=== COCO-2014 Dataset Setup ==="
echo "Target directory: $ROOT"
echo ""
echo "Please download the following files from https://cocodataset.org/#download:"
echo "1. 2014 Train images [83K/13GB]"
echo "2. 2014 Val images [41K/6GB]"
echo "3. 2014 Train/Val annotations [241MB]"
echo ""
echo "Then extract them as follows:"
echo "- Train images → $ROOT/train2014/"
echo "- Val images → $ROOT/val2014/"
echo "- Annotations → $ROOT/annotations/"
echo ""
echo "When finished, your directory structure should look like:"
echo "$ROOT/"
echo "├── train2014/       # contains training images"
echo "├── val2014/         # contains validation images"
echo "└── annotations/     # contains annotation JSON files"
echo "    ├── captions_train2014.json"
echo "    ├── captions_val2014.json"
echo "    └── instances_*.json"
echo ""
echo "Then you can run training with: python -m aecf.train --root $ROOT"

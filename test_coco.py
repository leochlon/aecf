# -*- coding: utf-8 -*-
"""AECF vs Baseline Comparison - Clean Final Version"""

import os
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score

# ============================================================================
# Setup and Imports
# ============================================================================

def install_packages():
    """Install required packages."""
    packages = ["open-clip-torch", "pycocotools", "transformers", "scikit-learn"]
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages if needed
if not os.environ.get('DOCKER_CONTAINER'):
    install_packages()

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import AECF Components - handle different import paths
try:
    from datasets import CocoDataset, ClipFeatureDataset, check_existing_features, make_clip_loaders, verify_clip_features, simulate_missing_modalities
    from proper_aecf_core import MultimodalAttentionPool, CurriculumMasking
except ImportError:
    try:
        from aecf.datasets import CocoDataset, ClipFeatureDataset, check_existing_features, make_clip_loaders, verify_clip_features, simulate_missing_modalities
        from aecf.proper_aecf_core import MultimodalAttentionPool, CurriculumMasking
    except ImportError:
        print("❌ Could not import AECF components. Ensure they are in the path.")
        sys.exit(1)

print("✅ AECF components imported successfully")

# ============================================================================
# Data Setup
# ============================================================================

def setup_data(coco_root="/content/data/coco", batch_size=512):
    """Setup COCO dataset using existing CLIP features or extract if needed."""
    print("Setting up COCO dataset...")
    
    # First check for existing pre-extracted features
    train_file, val_file, test_file = check_existing_features("./")
    
    if train_file and val_file:
        print("🎯 Using existing CLIP features from current directory")
        
        # Verify the features look reasonable
        verify_clip_features(train_file)
        verify_clip_features(val_file)
        if test_file:
            verify_clip_features(test_file)
        
        # Create data loaders
        if test_file:
            train_loader, val_loader, test_loader = make_clip_loaders(
                train_file=train_file,
                val_file=val_file,
                test_file=test_file,
                batch_size=batch_size
            )
            return train_loader, val_loader, test_loader
        else:
            train_loader, val_loader = make_clip_loaders(
                train_file=train_file,
                val_file=val_file,
                batch_size=batch_size
            )
            return train_loader, val_loader
    
    # Fallback: Use standard pipeline with COCO download and feature extraction
    print("⚠️  No existing features found - using standard pipeline")
    
    # Import the full pipeline function
    try:
        from datasets import setup_aecf_data_pipeline
    except ImportError:
        try:
            from aecf.datasets import setup_aecf_data_pipeline
        except ImportError:
            print("❌ Could not import setup_aecf_data_pipeline")
            sys.exit(1)
    
    return setup_aecf_data_pipeline(coco_root, batch_size=batch_size)

# ============================================================================
# Evaluation Functions - mAP Based
# ============================================================================

def calculate_map_score(y_pred, y_true):
    """Calculate mAP score for multi-label classification."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Apply sigmoid to logits
    y_pred_prob = 1 / (1 + np.exp(-y_pred))
    
    try:
        # Only calculate for classes that appear in ground truth
        valid_classes = y_true.sum(axis=0) > 0
        if not valid_classes.any():
            return 0.0
        
        map_score = average_precision_score(
            y_true[:, valid_classes], 
            y_pred_prob[:, valid_classes], 
            average='macro'
        )
        return map_score
    except ValueError:
        return 0.0

def evaluate_model(model, val_loader, missing_ratio=0.0):
    """Evaluate model with mAP score."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {missing_ratio*100:.0f}%", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if missing_ratio > 0:
                batch = simulate_missing_modalities(batch, missing_ratio)
            
            if hasattr(model, 'aecf_fusion'):
                logits, _ = model(batch)
            else:
                logits = model(batch)
            
            all_preds.append(logits.cpu())
            all_labels.append(batch['label'].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return calculate_map_score(all_preds, all_labels)

# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=10, model_name="Model"):
    """Train model with mAP evaluation."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training {model_name}...")
    best_map = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if hasattr(model, 'aecf_fusion'):
                logits, fusion_info = model(batch)
                loss = criterion(logits, batch['label'])
                if torch.isfinite(fusion_info['entropy_loss']):
                    loss += fusion_info['entropy_loss']
            else:
                logits = model(batch)
                loss = criterion(logits, batch['label'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation with mAP
        val_map = evaluate_model(model, val_loader)
        
        print(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, val_mAP={val_map:.4f}")
        
        # Early stopping
        if val_map > best_map:
            best_map = val_map
        elif epoch > 3 and val_map < best_map * 0.95:
            print("Early stopping - mAP not improving")
            break

# ============================================================================
# Model Definitions
# ============================================================================

class BaselineModel(nn.Module):
    """Simple baseline: concatenation + MLP."""
    
    def __init__(self, image_dim=512, text_dim=512, num_classes=80):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, num_classes)
        )
    
    def forward(self, batch):
        img_feat = self.image_proj(batch['image'])
        txt_feat = self.text_proj(batch['text'])
        fused = self.fusion(torch.cat([img_feat, txt_feat], dim=-1))
        return self.classifier(fused)

class AECFModel(nn.Module):
    """Model with AECF fusion layer."""
    
    def __init__(self, image_dim=512, text_dim=512, num_classes=80):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        self.aecf_fusion = ModularAECFLayer(feature_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, num_classes)
        )
    
    def forward(self, batch):
        img_feat = self.image_proj(batch['image'])
        txt_feat = self.text_proj(batch['text'])
        fused, fusion_info = self.aecf_fusion(img_feat, txt_feat)
        return self.classifier(fused), fusion_info

class ModularAECFLayer(nn.Module):
    """Modular AECF layer with curriculum masking."""
    
    def __init__(self, feature_dim=256, masking_prob=0.2, entropy_weight=0.01):
        super().__init__()
        self.entropy_weight = entropy_weight
        
        self.curriculum_masking = CurriculumMasking(
            base_mask_prob=masking_prob, entropy_target=0.7, min_active=1
        )
        self.attention_pool = MultimodalAttentionPool(
            embed_dim=feature_dim, num_heads=1, dropout=0.1,
            curriculum_masking=self.curriculum_masking, batch_first=True
        )
        self.fusion_query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
    
    def forward(self, image_feat, text_feat):
        """Forward pass with curriculum masking."""
        batch_size = image_feat.size(0)
        
        # Stack modalities and apply attention pooling
        modalities = torch.stack([image_feat, text_feat], dim=1)
        query = self.fusion_query.expand(batch_size, -1, -1)
        
        fused, info = self.attention_pool(
            query=query, key=modalities, value=modalities, return_info=True
        )
        
        if fused.size(1) == 1:
            fused = fused.squeeze(1)
        
        # Process entropy loss
        entropy_loss = torch.tensor(0.0, device=image_feat.device)
        if 'entropy' in info and torch.isfinite(info['entropy']).all():
            entropy_loss = self.curriculum_masking.entropy_loss(info['entropy'])
            entropy_loss = self.entropy_weight * entropy_loss
            if not torch.isfinite(entropy_loss):
                entropy_loss = torch.tensor(0.0, device=image_feat.device)
        
        processed_info = {
            'entropy': info.get('entropy', torch.tensor(0.0, device=image_feat.device)).mean(),
            'masking_rate': info.get('mask_rate', torch.tensor(0.0, device=image_feat.device)).mean(),
            'entropy_loss': entropy_loss
        }
        
        return fused, processed_info

# ============================================================================
# Results and Evaluation
# ============================================================================

def evaluate_robustness(model, val_loader, missing_ratios, model_name):
    """Evaluate robustness across missing modality ratios."""
    print(f"\nEvaluating {model_name} robustness...")
    results = {}
    
    for ratio in missing_ratios:
        map_score = evaluate_model(model, val_loader, ratio)
        results[ratio] = map_score
        print(f"  {ratio*100:.0f}% missing: mAP={map_score:.4f}")
    
    return results

def plot_and_summarize_results(baseline_results, aecf_results, missing_ratios):
    """Plot results and print summary."""
    print("\n" + "="*60)
    print("🏆 RESULTS SUMMARY (mAP Scores)")
    print("="*60)
    print(f"{'Missing %':<12} {'Baseline':<12} {'AECF':<12} {'Improvement':<12}")
    print("-"*60)
    
    improvements = []
    for ratio in missing_ratios:
        baseline_map = baseline_results[ratio]
        aecf_map = aecf_results[ratio]
        improvement = (aecf_map - baseline_map) / baseline_map * 100 if baseline_map > 0 else 0
        improvements.append(improvement)
        
        print(f"{ratio*100:>6.0f}%{'':<6} {baseline_map:<12.4f} {aecf_map:<12.4f} {improvement:>+8.1f}%")
    
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    print(f"\nAverage AECF improvement: {avg_improvement:+.1f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    missing_pcts = [r*100 for r in missing_ratios]
    baseline_maps = [baseline_results[r] for r in missing_ratios]
    aecf_maps = [aecf_results[r] for r in missing_ratios]
    
    plt.plot(missing_pcts, baseline_maps, 'b-o', label='Baseline', linewidth=2, markersize=8)
    plt.plot(missing_pcts, aecf_maps, 'r-o', label='AECF', linewidth=2, markersize=8)
    plt.xlabel('Missing Modalities (%)')
    plt.ylabel('mAP Score')
    plt.title('Robustness to Missing Modalities (mAP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(baseline_maps), max(aecf_maps)) * 1.1)
    plt.show()
    
    return avg_improvement

def debug_predictions(model, val_loader):
    """Debug model predictions."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if hasattr(model, 'aecf_fusion'):
            logits, info = model(batch)
            print(f"AECF - Entropy: {info['entropy']:.4f}, Masking: {info['masking_rate']:.4f}")
        else:
            logits = model(batch)
        
        probs = torch.sigmoid(logits)
        batch_map = calculate_map_score(logits, batch['label'])
        
        print(f"Logits: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"Probs: [{probs.min():.3f}, {probs.max():.3f}] (avg: {probs.mean():.3f})")
        print(f"GT avg: {batch['label'].mean():.3f}, Batch mAP: {batch_map:.4f}")

# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("🚀 Starting AECF vs Baseline Experiment")
    
    # Setup data - will use existing .pt files if available
    data_result = setup_data(batch_size=512)
    
    # Handle different return formats (with or without test loader)
    if len(data_result) == 3:
        train_loader, val_loader, test_loader = data_result
        print("📊 Using train, validation, and test sets")
    else:
        train_loader, val_loader = data_result
        test_loader = None
        print("📊 Using train and validation sets")
    
    # Get dimensions
    sample_batch = next(iter(train_loader))
    img_dim = sample_batch['image'].size(-1)
    txt_dim = sample_batch['text'].size(-1)
    num_classes = sample_batch['label'].size(-1)
    
    print(f"Dimensions - Image: {img_dim}D, Text: {txt_dim}D, Classes: {num_classes}")
    
    # Create and train models
    baseline_model = BaselineModel(img_dim, txt_dim, num_classes)
    aecf_model = AECFModel(img_dim, txt_dim, num_classes)
    
    print("\n📚 Training Baseline...")
    train_model(baseline_model, train_loader, val_loader, epochs=10, model_name="Baseline")
    
    print("\n📚 Training AECF...")
    train_model(aecf_model, train_loader, val_loader, epochs=10, model_name="AECF")
    
    # Debug models
    print("\n🔍 Debugging models...")
    print("Baseline:")
    debug_predictions(baseline_model, val_loader)
    print("\nAECF:")
    debug_predictions(aecf_model, val_loader)
    
    # Evaluate robustness
    missing_ratios = [0.0, 0.2, 0.4, 0.6]
    
    baseline_results = evaluate_robustness(baseline_model, val_loader, missing_ratios, "Baseline")
    aecf_results = evaluate_robustness(aecf_model, val_loader, missing_ratios, "AECF")
    
    # Show results
    avg_improvement = plot_and_summarize_results(baseline_results, aecf_results, missing_ratios)
    
    # Save results
    os.makedirs('/app/results', exist_ok=True)
    
    results = {
        'baseline_results': baseline_results,
        'aecf_results': aecf_results,
        'average_improvement_percent': avg_improvement,
        'missing_ratios': missing_ratios,
        'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'data_source': 'existing_features' if check_existing_features("./")[0] else 'extracted_features'
    }
    
    import json
    with open('/app/results/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Benchmark completed successfully!")
    print("📊 Results saved to /app/results/benchmark_results.json")
    
    if test_loader:
        print("\n🧪 Additional test set evaluation available")
        test_baseline = evaluate_model(baseline_model, test_loader)
        test_aecf = evaluate_model(aecf_model, test_loader)
        print(f"Test set - Baseline mAP: {test_baseline:.4f}, AECF mAP: {test_aecf:.4f}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""AECF vs Baseline Comparison - Fixed Version"""

import os
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Enhanced Dataset with Feature Normalization
# ============================================================================

class NormalizedClipFeatureDataset(ClipFeatureDataset):
    """
    Dataset that normalizes CLIP features to unit norm.
    """
    
    def __init__(self, features_file, normalize_features=True):
        super().__init__(features_file)
        
        if normalize_features:
            print("🔧 Normalizing features to unit norm...")
            
            # Normalize image features
            img_norms = torch.norm(self.images, dim=1, keepdim=True)
            self.images = self.images / (img_norms + 1e-8)
            
            # Normalize text features  
            txt_norms = torch.norm(self.texts, dim=1, keepdim=True)
            self.texts = self.texts / (txt_norms + 1e-8)
            
            print(f"   Image features normalized: norm = {torch.norm(self.images, dim=1).mean():.3f}")
            print(f"   Text features normalized: norm = {torch.norm(self.texts, dim=1).mean():.3f}")

def make_normalized_clip_loaders(train_file, val_file, test_file=None, batch_size=512, num_workers=0):
    """Create loaders with normalized CLIP features."""
    
    train_dataset = NormalizedClipFeatureDataset(train_file, normalize_features=True)
    val_dataset = NormalizedClipFeatureDataset(val_file, normalize_features=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    if test_file:
        test_dataset = NormalizedClipFeatureDataset(test_file, normalize_features=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

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
        
        try:
            # Verify the features look reasonable
            train_valid = verify_clip_features(train_file)
            val_valid = verify_clip_features(val_file)
            
            if not train_valid or not val_valid:
                print("❌ Feature validation failed, falling back to standard pipeline")
                train_file, val_file, test_file = None, None, None
            else:
                test_valid = False
                if test_file:
                    test_valid = verify_clip_features(test_file)
                    if not test_valid:
                        print("⚠️  Test file validation failed, proceeding without test set")
                        test_file = None
                
                # Create data loaders with normalization
                try:
                    if test_file:
                        train_loader, val_loader, test_loader = make_normalized_clip_loaders(
                            train_file=train_file,
                            val_file=val_file,
                            test_file=test_file,
                            batch_size=batch_size
                        )
                        print("✅ Normalized data loaders created successfully (with test set)")
                        return train_loader, val_loader, test_loader
                    else:
                        train_loader, val_loader = make_normalized_clip_loaders(
                            train_file=train_file,
                            val_file=val_file,
                            batch_size=batch_size
                        )
                        print("✅ Normalized data loaders created successfully")
                        return train_loader, val_loader
                        
                except Exception as e:
                    print(f"❌ Error creating data loaders: {e}")
                    print("   Falling back to standard pipeline")
                    train_file, val_file, test_file = None, None, None
                    
        except Exception as e:
            print(f"❌ Error with existing features: {e}")
            print("   Falling back to standard pipeline")
            train_file, val_file, test_file = None, None, None
    
    # Fallback: Use standard pipeline
    if not train_file or not val_file:
        print("⚠️  Using standard pipeline (download COCO + extract features)")
        
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
# Improved Missing Modality Simulation
# ============================================================================

def simulate_missing_modalities_improved(batch, missing_prob=0.3):
    """
    Improved missing modality simulation.
    Sets missing modalities to zero vectors (not just zero scalars).
    """
    batch_size = batch['image'].size(0)
    feature_dim = batch['image'].size(1)
    
    # Create random masks
    img_missing = torch.rand(batch_size) < missing_prob
    txt_missing = torch.rand(batch_size) < missing_prob
    
    # Ensure at least one modality remains per sample
    both_missing = img_missing & txt_missing
    if both_missing.any():
        # Randomly keep one modality for samples with both missing
        keep_img = torch.rand(both_missing.sum()) > 0.5
        img_missing[both_missing] = ~keep_img
        txt_missing[both_missing] = keep_img
    
    # Apply masks by zeroing out features
    batch_copy = batch.copy()
    
    if img_missing.any():
        batch_copy['image'][img_missing] = 0.0
    if txt_missing.any():
        batch_copy['text'][txt_missing] = 0.0
    
    return batch_copy

def simulate_missing_images(batch, missing_prob=0.3):
    """Simulate missing images only."""
    batch_size = batch['image'].size(0)
    img_missing = torch.rand(batch_size) < missing_prob
    
    batch_copy = batch.copy()
    if img_missing.any():
        batch_copy['image'][img_missing] = 0.0
    
    return batch_copy

def simulate_missing_text(batch, missing_prob=0.3):
    """Simulate missing text only."""
    batch_size = batch['text'].size(0)
    txt_missing = torch.rand(batch_size) < missing_prob
    
    batch_copy = batch.copy()
    if txt_missing.any():
        batch_copy['text'][txt_missing] = 0.0
    
    return batch_copy

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

def evaluate_model(model, val_loader, missing_ratio=0.0, missing_type='both', debug_missing=False):
    """Evaluate model with mAP score.
    
    Args:
        missing_type: 'both', 'images', or 'text' - what to make missing
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Debug counters for missing data analysis
    total_samples = 0
    img_missing_count = 0
    txt_missing_count = 0
    attention_weights_sum = torch.zeros(2) if hasattr(model, 'fusion_type') and model.fusion_type == 'aecf' else None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Eval {missing_ratio*100:.0f}% {missing_type}", leave=False)):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Track missing data statistics
            if missing_ratio > 0:
                original_batch = {k: v.clone() for k, v in batch.items()}
                
                # Apply different missing data patterns
                if missing_type == 'images':
                    batch = simulate_missing_images(batch, missing_ratio)
                elif missing_type == 'text':
                    batch = simulate_missing_text(batch, missing_ratio)
                else:  # missing_type == 'both'
                    batch = simulate_missing_modalities_improved(batch, missing_ratio)
                
                if debug_missing and batch_idx == 0:  # Debug first batch
                    # Count missing modalities
                    img_missing = (batch['image'].sum(dim=1) == 0).sum().item()
                    txt_missing = (batch['text'].sum(dim=1) == 0).sum().item()
                    total_samples += batch['image'].size(0)
                    img_missing_count += img_missing
                    txt_missing_count += txt_missing
                    
                    print(f"\nDEBUG Missing Data Stats (ratio={missing_ratio}):")
                    print(f"  Batch size: {batch['image'].size(0)}")
                    print(f"  Image missing: {img_missing}/{batch['image'].size(0)} ({img_missing/batch['image'].size(0)*100:.1f}%)")
                    print(f"  Text missing: {txt_missing}/{batch['image'].size(0)} ({txt_missing/batch['image'].size(0)*100:.1f}%)")
                    
                    # Check feature norms
                    img_norms = torch.norm(batch['image'], dim=1)
                    txt_norms = torch.norm(batch['text'], dim=1)
                    print(f"  Image norms: min={img_norms.min():.4f}, max={img_norms.max():.4f}, mean={img_norms.mean():.4f}")
                    print(f"  Text norms: min={txt_norms.min():.4f}, max={txt_norms.max():.4f}, mean={txt_norms.mean():.4f}")
            
            if hasattr(model, 'fusion_type') and model.fusion_type == 'aecf':
                logits, info = model(batch)
                
                # Track attention patterns with missing data
                if debug_missing and missing_ratio > 0 and batch_idx == 0 and 'attention_weights' in info:
                    att_weights = info['attention_weights']
                    if att_weights.dim() >= 2:
                        # Handle the new shape after masking [batch, 1, 2]
                        if att_weights.dim() == 3:  # [batch, 1, 2]
                            avg_att = att_weights.mean(dim=0).squeeze()  # [2]
                        else:
                            avg_att = att_weights.mean(dim=0)
                        
                        if avg_att.numel() >= 2:
                            # Handle potential NaN from masked attention
                            img_att = avg_att[0].item() if not torch.isnan(avg_att[0]) else 0.0
                            txt_att = avg_att[1].item() if not torch.isnan(avg_att[1]) else 0.0
                            print(f"  AECF attention with missing data: img={img_att:.3f}, txt={txt_att:.3f}")
                            
                            # Check missing modality detection
                            if 'img_missing_count' in info and 'txt_missing_count' in info:
                                print(f"  Missing modalities detected: img={info['img_missing_count']}, txt={info['txt_missing_count']}")
                            
                            # Check if attention is still going to zero vectors (should be 0 now)
                            img_missing_mask = (batch['image'].sum(dim=1) == 0)
                            txt_missing_mask = (batch['text'].sum(dim=1) == 0)
                            
                            if img_missing_mask.any() and att_weights.dim() > 1:
                                if att_weights.dim() == 3:  # [batch, 1, 2]
                                    img_missing_att = att_weights[img_missing_mask, 0, 0].mean()
                                else:
                                    img_missing_att = att_weights[img_missing_mask, 0].mean()
                                att_val = img_missing_att.item() if not torch.isnan(img_missing_att) else 0.0
                                print(f"  Attention to missing images: {att_val:.3f} (should be ~0)")
                            if txt_missing_mask.any() and att_weights.dim() > 1:
                                if att_weights.dim() == 3:  # [batch, 1, 2]
                                    txt_missing_att = att_weights[txt_missing_mask, 0, 1].mean()
                                else:
                                    txt_missing_att = att_weights[txt_missing_mask, 1].mean()
                                att_val = txt_missing_att.item() if not torch.isnan(txt_missing_att) else 0.0
                                print(f"  Attention to missing text: {att_val:.3f} (should be ~0)")
                                
                            # Show attention distribution for samples with missing modalities
                            if img_missing_mask.any():
                                present_modalities = (~txt_missing_mask[img_missing_mask]).sum()
                                print(f"  Samples with missing images: attention should focus on text")
                            if txt_missing_mask.any():
                                present_modalities = (~img_missing_mask[txt_missing_mask]).sum()
                                print(f"  Samples with missing text: attention should focus on images")
            else:
                logits = model(batch)
            
            all_preds.append(logits.cpu())
            all_labels.append(batch['label'].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return calculate_map_score(all_preds, all_labels)

# ============================================================================
# Enhanced Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=15, model_name="Model"):
    """Enhanced training with better hyperparameters for AECF."""
    model = model.to(device)
    
    # Use same learning rate for both models
    lr = 1e-3
    weight_decay = 0.01
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training {model_name} (lr={lr}, wd={weight_decay})...")
    best_map = 0.0
    patience = 5
    no_improve = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        epoch_entropies = []
        epoch_mask_rates = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Check if this is an AECF model
            if hasattr(model, 'fusion_type') and model.fusion_type == 'aecf':
                logits, fusion_info = model(batch)
                loss = criterion(logits, batch['label'])
                
                # Track curriculum masking statistics
                if 'entropy' in fusion_info:
                    entropy_val = fusion_info['entropy']
                    if torch.is_tensor(entropy_val):
                        if entropy_val.numel() > 1:
                            entropy_val = entropy_val.mean()
                        epoch_entropies.append(entropy_val.item())
                
                if 'masking_rate' in fusion_info:
                    mask_rate = fusion_info['masking_rate']
                    if torch.is_tensor(mask_rate):
                        if mask_rate.numel() > 1:
                            mask_rate = mask_rate.mean()
                        epoch_mask_rates.append(mask_rate.item())
                
                # Proper entropy regularization
                if 'entropy_loss' in fusion_info and torch.isfinite(fusion_info['entropy_loss']):
                    entropy_reg = 0.01 * fusion_info['entropy_loss']  # Reduced weight
                    loss += entropy_reg
            else:
                logits = model(batch)
                loss = criterion(logits, batch['label'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation with mAP
        val_map = evaluate_model(model, val_loader)
        
        # Print training stats with curriculum masking info for AECF
        if hasattr(model, 'fusion_type') and model.fusion_type == 'aecf' and epoch_entropies:
            avg_entropy = np.mean(epoch_entropies) if epoch_entropies else 0.0
            avg_mask_rate = np.mean(epoch_mask_rates) if epoch_mask_rates else 0.0
            print(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, val_mAP={val_map:.4f}, "
                  f"entropy={avg_entropy:.3f}, mask_rate={avg_mask_rate:.3f}")
        else:
            print(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, val_mAP={val_map:.4f}")
        
        # Early stopping with patience
        if val_map > best_map:
            best_map = val_map
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience and epoch > 5:
            print(f"Early stopping - no improvement for {patience} epochs")
            break

# ============================================================================
# Unified Model Definition
# ============================================================================

class MultimodalClassifier(nn.Module):
    """Unified multimodal classifier that can use either baseline or AECF fusion."""
    
    def __init__(self, image_dim=512, text_dim=512, num_classes=80, fusion_type='baseline'):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Shared feature projections
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layers based on type
        if fusion_type == 'baseline':
            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif fusion_type == 'aecf':
            # Use proper AECF components
            self.curriculum_masking = CurriculumMasking(
                base_mask_prob=0.1,  # Conservative masking
                entropy_target=0.7,  # Target 70% of max entropy
                min_active=1
            )
            
            self.attention_pool = MultimodalAttentionPool(
                embed_dim=256,
                num_heads=8,
                dropout=0.1,
                curriculum_masking=self.curriculum_masking,
                batch_first=True
            )
            
            # Learnable fusion query
            self.fusion_query = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # Shared classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, batch):
        img_feat = self.image_proj(batch['image'])
        txt_feat = self.text_proj(batch['text'])
        
        if self.fusion_type == 'baseline':
            # Baseline: Simple concatenation fusion - vulnerable to missing data
            fused = self.fusion(torch.cat([img_feat, txt_feat], dim=-1))
            logits = self.classifier(fused)
            return logits
            
        elif self.fusion_type == 'aecf':
            # AECF: Attention-based fusion with curriculum masking
            batch_size = img_feat.size(0)
            
            # Detect missing modalities based on input (before projection)
            img_present = torch.norm(batch['image'], dim=1) > 1e-6  # [batch_size]
            txt_present = torch.norm(batch['text'], dim=1) > 1e-6   # [batch_size]
            
            # Stack modalities for attention (use zeros for missing)
            modalities = torch.stack([img_feat, txt_feat], dim=1)  # [batch, 2, 256]
            
            # Create attention mask (True = should be ignored in attention)
            key_padding_mask = torch.stack([~img_present, ~txt_present], dim=1)  # [batch, 2]
            
            # Create fusion query for each sample in batch
            query = self.fusion_query.expand(batch_size, -1, -1)  # [batch, 1, 256]
            
            # Apply multimodal attention with proper masking
            fused, info = self.attention_pool(
                query=query,
                key=modalities,
                value=modalities,
                key_padding_mask=key_padding_mask,  # Properly mask missing modalities
                return_info=True
            )
            
            # Extract the single fused representation
            fused = fused.squeeze(1)  # [batch, 256]
            
            # Classify
            logits = self.classifier(fused)
            
            # Process info for training
            fusion_info = {}
            if 'entropy' in info:
                fusion_info['entropy'] = info['entropy']
            if 'mask_rate' in info:
                fusion_info['masking_rate'] = info['mask_rate']
            if 'attention_weights' in info:
                fusion_info['attention_weights'] = info['attention_weights']
            
            # Add missing modality stats for debugging
            fusion_info['img_missing_count'] = (~img_present).sum().item()
            fusion_info['txt_missing_count'] = (~txt_present).sum().item()
            fusion_info['modality_mask'] = torch.stack([img_present, txt_present], dim=1)
                
            # Compute entropy loss if we have entropy info
            if 'entropy' in info:
                entropy_loss = self.curriculum_masking.entropy_loss(info['entropy'])
                fusion_info['entropy_loss'] = entropy_loss
            
            return logits, fusion_info

# ============================================================================
# Results and Evaluation (same as before)
# ============================================================================

def evaluate_robustness_comprehensive(model, val_loader, missing_ratios, model_name):
    """Evaluate robustness across missing modality ratios for different scenarios."""
    print(f"\nEvaluating {model_name} robustness...")
    results = {
        'both': {},
        'images': {},
        'text': {}
    }
    
    # Test complete data first
    map_score = evaluate_model(model, val_loader, 0.0, 'both')
    results['both'][0.0] = map_score
    results['images'][0.0] = map_score
    results['text'][0.0] = map_score
    print(f"  Complete data: mAP={map_score:.4f}")
    
    for ratio in missing_ratios:
        if ratio == 0.0:
            continue
            
        # Test missing images only
        debug_missing = hasattr(model, 'fusion_type') and model.fusion_type == 'aecf'
        map_score_img = evaluate_model(model, val_loader, ratio, 'images', debug_missing=debug_missing)
        results['images'][ratio] = map_score_img
        print(f"  {ratio*100:.0f}% images missing: mAP={map_score_img:.4f}")
        
        # Test missing text only
        map_score_txt = evaluate_model(model, val_loader, ratio, 'text')
        results['text'][ratio] = map_score_txt
        print(f"  {ratio*100:.0f}% text missing: mAP={map_score_txt:.4f}")
        
        # Test mixed missing (original behavior)
        map_score_both = evaluate_model(model, val_loader, ratio, 'both')
        results['both'][ratio] = map_score_both
        print(f"  {ratio*100:.0f}% both missing: mAP={map_score_both:.4f}")
    
    return results

def evaluate_robustness(model, val_loader, missing_ratios, model_name):
    """Legacy function - calls comprehensive evaluation but returns mixed results for compatibility."""
    comprehensive_results = evaluate_robustness_comprehensive(model, val_loader, missing_ratios, model_name)
    return comprehensive_results['both']  # Return mixed results for backward compatibility

def plot_and_summarize_results_comprehensive(baseline_results, aecf_results, missing_ratios):
    """Plot comprehensive results with separate scenarios."""
    print("\n" + "="*80)
    print("🏆 COMPREHENSIVE RESULTS SUMMARY (mAP Scores)")
    print("="*80)
    
    scenarios = ['images', 'text', 'both']
    scenario_names = ['Missing Images', 'Missing Text', 'Missing Both']
    
    for scenario, scenario_name in zip(scenarios, scenario_names):
        print(f"\n📊 {scenario_name}:")
        print(f"{'Missing %':<12} {'Baseline':<12} {'AECF':<12} {'Improvement':<12}")
        print("-"*60)
        
        improvements = []
        for ratio in missing_ratios:
            baseline_map = baseline_results[scenario][ratio]
            aecf_map = aecf_results[scenario][ratio]
            improvement = (aecf_map - baseline_map) / baseline_map * 100 if baseline_map > 0 else 0
            improvements.append(improvement)
            
            print(f"{ratio*100:>6.0f}%{'':<6} {baseline_map:<12.4f} {aecf_map:<12.4f} {improvement:>+8.1f}%")
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        print(f"Average AECF improvement: {avg_improvement:+.1f}%")
    
    return baseline_results, aecf_results

def plot_and_summarize_results(baseline_results, aecf_results, missing_ratios):
    """Legacy function - simplified summary for backward compatibility."""
    print("\n" + "="*60)
    print("🏆 RESULTS SUMMARY (mAP Scores) - Mixed Missing")
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
    
    return avg_improvement

def debug_predictions(model, val_loader):
    """Debug model predictions with detailed analysis."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Debug input features
        print(f"Input features - Image norm: {torch.norm(batch['image'], dim=1).mean():.4f}, Text norm: {torch.norm(batch['text'], dim=1).mean():.4f}")
        
        # Check if this is an AECF model
        if hasattr(model, 'fusion_type') and model.fusion_type == 'aecf':
            logits, info = model(batch)
            # Convert tensor values to scalars for printing
            entropy_val = info.get('entropy', 0.0)
            if torch.is_tensor(entropy_val):
                if entropy_val.numel() > 1:
                    entropy_val = entropy_val.mean().item()  # Take mean if it's a vector
                else:
                    entropy_val = entropy_val.item()
            
            masking_val = info.get('masking_rate', 0.0)
            if torch.is_tensor(masking_val):
                if masking_val.numel() > 1:
                    masking_val = masking_val.mean().item()  # Take mean if it's a vector
                else:
                    masking_val = masking_val.item()
            print(f"AECF - Entropy: {entropy_val:.4f}, Masking: {masking_val:.4f}")
            if 'attention_weights' in info:
                try:
                    att_weights = info['attention_weights']
                    print(f"Attention weights shape: {att_weights.shape}")
                    
                    # Handle different possible shapes
                    if att_weights.dim() >= 3:  # [batch, heads, seq_len] or similar 
                        att_weights = att_weights.mean(dim=1)  # Average over heads
                    if att_weights.dim() >= 2:  # [batch, seq_len]
                        att_weights = att_weights.mean(dim=0)  # Average over batch
                    
                    # Ensure we have at least 2 elements for image/text
                    if att_weights.numel() >= 2:
                        print(f"AECF - Attention weights: image={att_weights[0]:.3f}, text={att_weights[1]:.3f}")
                    else:
                        print(f"AECF - Attention weights: {att_weights}")
                except Exception as e:
                    print(f"AECF - Could not parse attention weights: {e}")
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
    print("🚀 Starting Fixed AECF vs Baseline Experiment")
    
    # Setup data - will use normalized existing .pt files if available
    data_result = setup_data(batch_size=512)
    
    # Handle different return formats
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
    
    # Debug feature statistics
    print(f"\n🔍 Feature Analysis:")
    img_batch = sample_batch['image']
    txt_batch = sample_batch['text']
    print(f"Image features - mean: {img_batch.mean():.4f}, std: {img_batch.std():.4f}, norm: {torch.norm(img_batch, dim=1).mean():.4f}")
    print(f"Text features - mean: {txt_batch.mean():.4f}, std: {txt_batch.std():.4f}, norm: {torch.norm(txt_batch, dim=1).mean():.4f}")
    print(f"Cross-modal similarity: {torch.nn.functional.cosine_similarity(img_batch[:100], txt_batch[:100]).mean():.4f}")
    
    # Create models with different fusion strategies
    baseline_model = MultimodalClassifier(img_dim, txt_dim, num_classes, fusion_type='baseline')
    aecf_model = MultimodalClassifier(img_dim, txt_dim, num_classes, fusion_type='aecf')
    
    print("\n📚 Training Baseline (Concatenation)...")
    train_model(baseline_model, train_loader, val_loader, epochs=30, model_name="Baseline")
    
    print("\n📚 Training AECF (Attention + Curriculum Masking)...")
    train_model(aecf_model, train_loader, val_loader, epochs=30, model_name="AECF")
    
    # Debug models
    print("\n🔍 Debugging models...")
    print("Baseline:")
    debug_predictions(baseline_model, val_loader)
    print("\nAECF:")
    debug_predictions(aecf_model, val_loader)
    
    # Evaluate robustness with comprehensive scenarios
    missing_ratios = [0.0, 0.2, 0.4, 0.6]
    
    baseline_results = evaluate_robustness_comprehensive(baseline_model, val_loader, missing_ratios, "Baseline")
    aecf_results = evaluate_robustness_comprehensive(aecf_model, val_loader, missing_ratios, "AECF")
    
    # Show comprehensive results
    plot_and_summarize_results_comprehensive(baseline_results, aecf_results, missing_ratios)
    
    # Also show legacy mixed results for compatibility
    avg_improvement = plot_and_summarize_results(baseline_results['both'], aecf_results['both'], missing_ratios)
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    
    results = {
        'baseline_results': baseline_results,
        'aecf_results': aecf_results,
        'average_improvement_percent': avg_improvement,
        'missing_ratios': missing_ratios,
        'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'data_source': 'existing_features_normalized'
    }
    
    import json
    with open('./results/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Fixed benchmark completed successfully!")
    print("📊 Results saved to ./results/benchmark_results.json")
    
    if test_loader:
        print("\n🧪 Additional test set evaluation")
        test_baseline = evaluate_model(baseline_model, test_loader)
        test_aecf = evaluate_model(aecf_model, test_loader)
        print(f"Test set - Baseline mAP: {test_baseline:.4f}, AECF mAP: {test_aecf:.4f}")
        improvement = (test_aecf - test_baseline) / test_baseline * 100 if test_baseline > 0 else 0
        print(f"Test improvement: {improvement:+.1f}%")

if __name__ == "__main__":
    main()
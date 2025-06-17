#!/usr/bin/env python3
"""
Parallel Training Experiment: Baseline vs AECF with Late Curriculum Activation
Enhanced with F1 scores and pathology-level analysis

Baseline Model: Simple concatenation fusion throughout
AECF Model: Attention pooling from start, curriculum masking turns on at epoch 40

Expected: Gate entropy near zero epochs 1-40, then jumps up when masking activates
Tracks per-pathology F1 scores to identify which pathologies benefit most from curriculum masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import AECF components
try:
    from aecf.AECFLayer import MultimodalAttentionPool, CurriculumMasking
except ImportError:
    from AECFLayer import MultimodalAttentionPool, CurriculumMasking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Common pathology names for X-ray datasets (adjust based on your actual dataset)
PATHOLOGY_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]

class BaselineModel(nn.Module):
    """Simple concatenation baseline model."""
    
    def __init__(self, image_dim=512, text_dim=512, num_classes=80, hidden_dim=256):
        super().__init__()
        self.name = "Concat_Baseline"
        self.hidden_dim = hidden_dim
        
        # Simple encoders
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Single modality projections for missing modality scenarios
        self.image_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Shared classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, image_features, text_features):
        # Encode
        img_encoded = self.image_encoder(image_features)
        txt_encoded = self.text_encoder(text_features)
        
        # Detect presence
        img_present = torch.norm(image_features, dim=1) > 1e-6
        txt_present = torch.norm(text_features, dim=1) > 1e-6
        
        both_present = img_present & txt_present
        only_img = img_present & ~txt_present
        only_txt = ~img_present & txt_present
        
        batch_size = image_features.size(0)
        fused_features = torch.zeros(batch_size, self.hidden_dim * 2, device=image_features.device)
        
        # Both modalities: concatenation
        if both_present.any():
            indices = torch.where(both_present)[0]
            concat_features = torch.cat([img_encoded[indices], txt_encoded[indices]], dim=-1)
            fused_features[indices] = concat_features
        
        # Single modalities
        if only_img.any():
            indices = torch.where(only_img)[0]
            fused_features[indices] = self.image_proj(img_encoded[indices])
        
        if only_txt.any():
            indices = torch.where(only_txt)[0]
            fused_features[indices] = self.text_proj(txt_encoded[indices])
        
        return self.classifier(fused_features)

class AECFModel(nn.Module):
    """AECF model with controllable curriculum masking."""
    
    def __init__(self, image_dim=512, text_dim=512, num_classes=80, hidden_dim=256):
        super().__init__()
        self.name = "AECF_Model"
        self.hidden_dim = hidden_dim
        self.curriculum_enabled = False  # Control curriculum masking
        self.missing_modality_training = False
        
        # Simple encoders (same architecture as baseline)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # AECF components
        self.curriculum_masking = CurriculumMasking(base_mask_prob=0.15)
        self.attention_pool = MultimodalAttentionPool(
            embed_dim=hidden_dim,
            num_heads=4,
            curriculum_masking=None,  # Start with no masking
            batch_first=True
        )
        self.fusion_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Single modality projections
        self.image_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Fusion projection
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Shared classifier (same architecture as baseline)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def _simulate_missing_modalities(self, image_features, text_features, missing_prob=0.3):
        """Randomly mask modalities during training."""
        if not (self.training and self.missing_modality_training):
            return image_features, text_features
        
        batch_size = image_features.size(0)
        mask_image = torch.rand(batch_size, device=image_features.device) < missing_prob
        mask_text = torch.rand(batch_size, device=text_features.device) < missing_prob
        
        # Ensure at least one modality remains
        both_masked = mask_image & mask_text
        if both_masked.any():
            keep_image = torch.rand(both_masked.sum(), device=image_features.device) > 0.5
            mask_image[both_masked] = ~keep_image
            mask_text[both_masked] = keep_image
        
        masked_image = image_features.clone()
        masked_text = text_features.clone()
        masked_image[mask_image] = 0
        masked_text[mask_text] = 0
        
        return masked_image, masked_text
    
    def toggle_curriculum(self, enabled):
        """Enable/disable curriculum masking."""
        self.curriculum_enabled = enabled
        if enabled:
            self.attention_pool.curriculum_masking = self.curriculum_masking
            print("🎯 Curriculum masking ENABLED")
        else:
            self.attention_pool.curriculum_masking = None
            print("📖 Curriculum masking DISABLED")
    
    def forward(self, image_features, text_features, return_info=False):
        batch_size = image_features.size(0)
        info = {}
        
        # Simulate missing modalities during training
        if self.training and self.missing_modality_training:
            image_features, text_features = self._simulate_missing_modalities(image_features, text_features)
        
        # Encode
        img_encoded = self.image_encoder(image_features)
        txt_encoded = self.text_encoder(text_features)
        
        # Detect presence
        img_present = torch.norm(image_features, dim=1) > 1e-6
        txt_present = torch.norm(text_features, dim=1) > 1e-6
        
        both_present = img_present & txt_present
        only_img = img_present & ~txt_present
        only_txt = ~img_present & txt_present
        
        fused_features = torch.zeros(batch_size, self.hidden_dim * 2, device=image_features.device)
        
        # Both modalities: use attention pooling
        if both_present.any():
            indices = torch.where(both_present)[0]
            modalities = torch.stack([img_encoded[indices], txt_encoded[indices]], dim=1)
            query = self.fusion_query.expand(len(indices), -1, -1)
            
            attn_output, attn_info = self.attention_pool(
                query=query, key=modalities, value=modalities, return_info=True
            )
            
            multimodal_features = self.fusion_proj(attn_output.squeeze(1))
            fused_features[indices] = multimodal_features
            
            if return_info:
                info.update(attn_info)
        
        # Single modalities
        if only_img.any():
            indices = torch.where(only_img)[0]
            fused_features[indices] = self.image_proj(img_encoded[indices])
        
        if only_txt.any():
            indices = torch.where(only_txt)[0]
            fused_features[indices] = self.text_proj(txt_encoded[indices])
        
        logits = self.classifier(fused_features)
        return (logits, info) if return_info else logits

def load_data():
    """Load X-ray CLIP features."""
    train_data = torch.load('xray_train_clip_feats.pt', map_location='cpu')
    val_data = torch.load('xray_validation_clip_feats.pt', map_location='cpu')
    
    train_dataset = TensorDataset(train_data['image'], train_data['text'], train_data['label'])
    val_dataset = TensorDataset(val_data['image'], val_data['text'], val_data['label'])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, train_data['image'].shape[1], train_data['text'].shape[1], train_data['label'].shape[1]

def mask_modality(images, texts, mask_type='none'):
    """Mask one modality."""
    if mask_type == 'images':
        return torch.zeros_like(images), texts
    elif mask_type == 'texts':
        return images, torch.zeros_like(texts)
    return images, texts

def calculate_metrics(y_pred, y_true, threshold=0.5):
    """Calculate mAP and per-label F1 scores."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_pred_prob = torch.sigmoid(torch.tensor(y_pred)).numpy()
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    
    # Calculate mAP
    try:
        valid_classes = y_true.sum(axis=0) > 0
        if not valid_classes.any():
            map_score = 0.0
        else:
            map_score = average_precision_score(y_true[:, valid_classes], y_pred_prob[:, valid_classes], average='macro')
    except ValueError:
        map_score = 0.0
    
    # Calculate per-label F1 scores
    f1_scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:  # Only calculate F1 if positive samples exist
            try:
                f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
                f1_scores.append(f1)
            except:
                f1_scores.append(0.0)
        else:
            f1_scores.append(0.0)
    
    # Calculate macro F1
    macro_f1 = np.mean([f1 for f1 in f1_scores if f1 > 0]) if any(f1 > 0 for f1 in f1_scores) else 0.0
    
    return map_score, macro_f1, np.array(f1_scores)

def evaluate_model(model, val_loader, mask_type='none'):
    """Evaluate model with masking, returning mAP, macro F1, and per-label F1."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, texts, labels in tqdm(val_loader, desc=f"Eval {mask_type}", leave=False):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            masked_images, masked_texts = mask_modality(images, texts, mask_type)
            logits = model(masked_images, masked_texts)
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    return calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))

def train_both_models(baseline_model, aecf_model, train_loader, val_loader, epochs=60, lr=1e-4):
    """Train both models in parallel with curriculum activation at epoch 40."""
    print(f"\n{'='*60}")
    print("PARALLEL TRAINING: BASELINE vs AECF")
    print(f"Epochs 1-40: AECF without curriculum masking")
    print(f"Epochs 41-60: AECF with curriculum masking")
    print('='*60)
    
    # Setup optimizers
    baseline_model = baseline_model.to(device)
    aecf_model = aecf_model.to(device)
    
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=0.01)
    aecf_optimizer = torch.optim.AdamW(aecf_model.parameters(), lr=lr, weight_decay=0.01)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Results tracking
    results = {
        'baseline': {
            'train_loss': [], 'val_full_map': [], 'val_full_f1': [], 'val_full_f1_per_label': [],
            'val_no_images_map': [], 'val_no_images_f1': [], 'val_no_images_f1_per_label': [],
            'val_no_texts_map': [], 'val_no_texts_f1': [], 'val_no_texts_f1_per_label': []
        },
        'aecf': {
            'train_loss': [], 'val_full_map': [], 'val_full_f1': [], 'val_full_f1_per_label': [],
            'val_no_images_map': [], 'val_no_images_f1': [], 'val_no_images_f1_per_label': [],
            'val_no_texts_map': [], 'val_no_texts_f1': [], 'val_no_texts_f1_per_label': [],
            'gate_entropy': [], 'mask_rate': []
        }
    }
    
    for epoch in range(epochs):
        # Toggle curriculum masking at epoch 40
        if epoch == 40:
            print(f"\n🎯 EPOCH {epoch+1}: ACTIVATING CURRICULUM MASKING!")
            aecf_model.toggle_curriculum(True)
            aecf_model.missing_modality_training = True
        
        # Training phase
        baseline_model.train()
        aecf_model.train()
        
        baseline_losses = []
        aecf_losses = []
        epoch_entropies = []
        epoch_mask_rates = []
        
        for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            # Train baseline model
            baseline_optimizer.zero_grad()
            baseline_logits = baseline_model(images, texts)
            baseline_loss = criterion(baseline_logits, labels)
            baseline_loss.backward()
            baseline_optimizer.step()
            baseline_losses.append(baseline_loss.item())
            
            # Train AECF model
            aecf_optimizer.zero_grad()
            aecf_logits, aecf_info = aecf_model(images, texts, return_info=True)
            aecf_loss = criterion(aecf_logits, labels)
            aecf_loss.backward()
            aecf_optimizer.step()
            aecf_losses.append(aecf_loss.item())
            
            # Collect AECF metrics
            if 'entropy' in aecf_info:
                epoch_entropies.append(aecf_info['entropy'].mean().item())
            if 'mask_rate' in aecf_info:
                epoch_mask_rates.append(aecf_info['mask_rate'].mean().item())
        
        # Validation
        baseline_val_full = evaluate_model(baseline_model, val_loader, 'none')
        baseline_val_no_images = evaluate_model(baseline_model, val_loader, 'images')
        baseline_val_no_texts = evaluate_model(baseline_model, val_loader, 'texts')
        
        aecf_val_full = evaluate_model(aecf_model, val_loader, 'none')
        aecf_val_no_images = evaluate_model(aecf_model, val_loader, 'images')
        aecf_val_no_texts = evaluate_model(aecf_model, val_loader, 'texts')
        
        # Store results
        results['baseline']['train_loss'].append(np.mean(baseline_losses))
        results['baseline']['val_full_map'].append(baseline_val_full[0])
        results['baseline']['val_full_f1'].append(baseline_val_full[1])
        results['baseline']['val_full_f1_per_label'].append(baseline_val_full[2])
        results['baseline']['val_no_images_map'].append(baseline_val_no_images[0])
        results['baseline']['val_no_images_f1'].append(baseline_val_no_images[1])
        results['baseline']['val_no_images_f1_per_label'].append(baseline_val_no_images[2])
        results['baseline']['val_no_texts_map'].append(baseline_val_no_texts[0])
        results['baseline']['val_no_texts_f1'].append(baseline_val_no_texts[1])
        results['baseline']['val_no_texts_f1_per_label'].append(baseline_val_no_texts[2])
        
        results['aecf']['train_loss'].append(np.mean(aecf_losses))
        results['aecf']['val_full_map'].append(aecf_val_full[0])
        results['aecf']['val_full_f1'].append(aecf_val_full[1])
        results['aecf']['val_full_f1_per_label'].append(aecf_val_full[2])
        results['aecf']['val_no_images_map'].append(aecf_val_no_images[0])
        results['aecf']['val_no_images_f1'].append(aecf_val_no_images[1])
        results['aecf']['val_no_images_f1_per_label'].append(aecf_val_no_images[2])
        results['aecf']['val_no_texts_map'].append(aecf_val_no_texts[0])
        results['aecf']['val_no_texts_f1'].append(aecf_val_no_texts[1])
        results['aecf']['val_no_texts_f1_per_label'].append(aecf_val_no_texts[2])
        results['aecf']['gate_entropy'].append(np.mean(epoch_entropies) if epoch_entropies else 0.0)
        results['aecf']['mask_rate'].append(np.mean(epoch_mask_rates) if epoch_mask_rates else 0.0)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}: "
              f"Baseline mAP={baseline_val_full[0]:.4f}, F1={baseline_val_full[1]:.4f} | "
              f"AECF mAP={aecf_val_full[0]:.4f}, F1={aecf_val_full[1]:.4f}, Entropy={results['aecf']['gate_entropy'][-1]:.4f}")
        
        if epoch >= 40:
            print(f"         Missing Texts: Baseline F1={baseline_val_no_texts[1]:.4f}, AECF F1={aecf_val_no_texts[1]:.4f}")
    
    return results

def analyze_pathology_improvements(results, num_classes):
    """Analyze which pathologies improve the most after curriculum activation for both missing scenarios."""
    print("\n" + "="*80)
    print("🔬 PATHOLOGY-LEVEL CURRICULUM IMPROVEMENT ANALYSIS")
    print("="*80)
    
    pathology_names = PATHOLOGY_NAMES[:num_classes] if num_classes <= len(PATHOLOGY_NAMES) else [f"Pathology_{i}" for i in range(num_classes)]
    
    # Analyze Missing Text Scenario
    print("📊 MISSING TEXT SCENARIO ANALYSIS:")
    print("-" * 50)
    
    pre_curriculum_no_text = np.mean([results['aecf']['val_no_texts_f1_per_label'][i] for i in range(35, 40)], axis=0)
    post_curriculum_no_text = np.mean([results['aecf']['val_no_texts_f1_per_label'][i] for i in range(55, 60)], axis=0)
    f1_improvements_no_text = post_curriculum_no_text - pre_curriculum_no_text
    top_indices_no_text = np.argsort(f1_improvements_no_text)[-4:][::-1]
    
    print("TOP 4 MOST IMPROVED PATHOLOGIES (Missing Text):")
    improvement_data_no_text = []
    for i, idx in enumerate(top_indices_no_text):
        pathology_name = pathology_names[idx] if idx < len(pathology_names) else f"Pathology_{idx}"
        pre_f1 = pre_curriculum_no_text[idx]
        post_f1 = post_curriculum_no_text[idx]
        improvement = f1_improvements_no_text[idx]
        
        print(f"{i+1}. {pathology_name:20s}: {pre_f1:.4f} → {post_f1:.4f} (+{improvement:.4f})")
        improvement_data_no_text.append({
            'pathology': pathology_name,
            'pre_f1': pre_f1,
            'post_f1': post_f1,
            'improvement': improvement
        })
    
    # Analyze Missing Image Scenario
    print("\n📊 MISSING IMAGE SCENARIO ANALYSIS:")
    print("-" * 50)
    
    pre_curriculum_no_image = np.mean([results['aecf']['val_no_images_f1_per_label'][i] for i in range(35, 40)], axis=0)
    post_curriculum_no_image = np.mean([results['aecf']['val_no_images_f1_per_label'][i] for i in range(55, 60)], axis=0)
    f1_improvements_no_image = post_curriculum_no_image - pre_curriculum_no_image
    top_indices_no_image = np.argsort(f1_improvements_no_image)[-4:][::-1]
    
    print("TOP 4 MOST IMPROVED PATHOLOGIES (Missing Image):")
    improvement_data_no_image = []
    for i, idx in enumerate(top_indices_no_image):
        pathology_name = pathology_names[idx] if idx < len(pathology_names) else f"Pathology_{idx}"
        pre_f1 = pre_curriculum_no_image[idx]
        post_f1 = post_curriculum_no_image[idx]
        improvement = f1_improvements_no_image[idx]
        
        print(f"{i+1}. {pathology_name:20s}: {pre_f1:.4f} → {post_f1:.4f} (+{improvement:.4f})")
        improvement_data_no_image.append({
            'pathology': pathology_name,
            'pre_f1': pre_f1,
            'post_f1': post_f1,
            'improvement': improvement
        })
    
    return {
        'no_text': {'data': improvement_data_no_text, 'indices': top_indices_no_text},
        'no_image': {'data': improvement_data_no_image, 'indices': top_indices_no_image}
    }

def plot_pathology_improvements(improvement_results):
    """Plot bar charts of pathology improvements for both missing scenarios."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Missing Text Scenario
    improvement_data_no_text = improvement_results['no_text']['data']
    pathologies_no_text = [data['pathology'] for data in improvement_data_no_text]
    pre_f1_no_text = [data['pre_f1'] for data in improvement_data_no_text]
    post_f1_no_text = [data['post_f1'] for data in improvement_data_no_text]
    
    x = np.arange(len(pathologies_no_text))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pre_f1_no_text, width, label='Before Curriculum (Epochs 35-40)', 
                   color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, post_f1_no_text, width, label='After Curriculum (Epochs 55-60)', 
                   color='lightgreen', alpha=0.8)
    
    # Add improvement arrows for missing text
    for i, data in enumerate(improvement_data_no_text):
        if data['improvement'] > 0:
            ax1.annotate('', xy=(i + width/2, data['post_f1']), 
                       xytext=(i - width/2, data['pre_f1']),
                       arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
            ax1.text(i, max(data['pre_f1'], data['post_f1']) + 0.02, 
                   f'+{data["improvement"]:.3f}', 
                   ha='center', va='bottom', fontweight='bold', color='darkgreen')
    
    ax1.set_xlabel('Pathologies', fontweight='bold', fontsize=12)
    ax1.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
    ax1.set_title('🖼️➡️❌ Missing Text Scenario\nTop 4 Most Improved Pathologies', 
                fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pathologies_no_text, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars for missing text
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot Missing Image Scenario
    improvement_data_no_image = improvement_results['no_image']['data']
    pathologies_no_image = [data['pathology'] for data in improvement_data_no_image]
    pre_f1_no_image = [data['pre_f1'] for data in improvement_data_no_image]
    post_f1_no_image = [data['post_f1'] for data in improvement_data_no_image]
    
    x2 = np.arange(len(pathologies_no_image))
    
    bars3 = ax2.bar(x2 - width/2, pre_f1_no_image, width, label='Before Curriculum (Epochs 35-40)', 
                   color='lightblue', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, post_f1_no_image, width, label='After Curriculum (Epochs 55-60)', 
                   color='lightgreen', alpha=0.8)
    
    # Add improvement arrows for missing image
    for i, data in enumerate(improvement_data_no_image):
        if data['improvement'] > 0:
            ax2.annotate('', xy=(i + width/2, data['post_f1']), 
                       xytext=(i - width/2, data['pre_f1']),
                       arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
            ax2.text(i, max(data['pre_f1'], data['post_f1']) + 0.02, 
                   f'+{data["improvement"]:.3f}', 
                   ha='center', va='bottom', fontweight='bold', color='darkgreen')
    
    ax2.set_xlabel('Pathologies', fontweight='bold', fontsize=12)
    ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
    ax2.set_title('📄➡️❌ Missing Image Scenario\nTop 4 Most Improved Pathologies', 
                fontweight='bold', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(pathologies_no_image, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars for missing image
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pathology_improvements_both_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Pathology improvement charts saved as 'pathology_improvements_both_scenarios.png'")
    
    # Create a comparison summary
    print("\n🔍 CROSS-SCENARIO COMPARISON:")
    print("-" * 60)
    
    # Check if any pathologies appear in both top 4 lists
    pathologies_both = set(pathologies_no_text) & set(pathologies_no_image)
    if pathologies_both:
        print(f"📋 Pathologies improved in BOTH scenarios: {', '.join(pathologies_both)}")
    else:
        print("📋 No pathologies appear in top 4 for both scenarios")
    
    # Show which scenario had bigger improvements overall
    avg_improvement_no_text = np.mean([data['improvement'] for data in improvement_data_no_text])
    avg_improvement_no_image = np.mean([data['improvement'] for data in improvement_data_no_image])
    
    print(f"📊 Average improvement (Missing Text): +{avg_improvement_no_text:.4f}")
    print(f"📊 Average improvement (Missing Image): +{avg_improvement_no_image:.4f}")
    
    if avg_improvement_no_text > avg_improvement_no_image:
        print("🏆 AECF shows stronger curriculum effect when text is missing")
    elif avg_improvement_no_image > avg_improvement_no_text:
        print("🏆 AECF shows stronger curriculum effect when images are missing")
    else:
        print("⚖️ AECF shows similar curriculum effects in both scenarios")

def plot_parallel_results(results):
    """Plot results showing curriculum activation effect."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    epochs = range(1, len(results['baseline']['train_loss']) + 1)
    
    # Training Loss
    axes[0, 0].plot(epochs, results['baseline']['train_loss'], 'b-o', label='Baseline (Concat)', linewidth=2)
    axes[0, 0].plot(epochs, results['aecf']['train_loss'], 'r-s', label='AECF', linewidth=2)
    axes[0, 0].axvline(x=40, color='green', linestyle='--', alpha=0.7, label='Curriculum ON')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Full Data Performance (F1)
    axes[0, 1].plot(epochs, results['baseline']['val_full_f1'], 'b-o', label='Baseline (Concat)', linewidth=2)
    axes[0, 1].plot(epochs, results['aecf']['val_full_f1'], 'r-s', label='AECF', linewidth=2)
    axes[0, 1].axvline(x=40, color='green', linestyle='--', alpha=0.7, label='Curriculum ON')
    axes[0, 1].set_title('Validation F1 (Full Data)', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Macro F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gate Entropy (KEY METRIC!)
    axes[0, 2].plot(epochs, results['aecf']['gate_entropy'], 'purple', linewidth=3, label='Gate Entropy')
    axes[0, 2].axvline(x=40, color='green', linestyle='--', alpha=0.7, label='Curriculum ON')
    axes[0, 2].set_title('🎯 AECF Gate Entropy', fontweight='bold', fontsize=14)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Missing Texts Performance (F1)
    axes[1, 0].plot(epochs, results['baseline']['val_no_texts_f1'], 'b-o', label='Baseline (Concat)', linewidth=2)
    axes[1, 0].plot(epochs, results['aecf']['val_no_texts_f1'], 'r-s', label='AECF', linewidth=2)
    axes[1, 0].axvline(x=40, color='green', linestyle='--', alpha=0.7, label='Curriculum ON')
    axes[1, 0].set_title('Validation F1 (No Texts)', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Missing Images Performance (F1)
    axes[1, 1].plot(epochs, results['baseline']['val_no_images_f1'], 'b-o', label='Baseline (Concat)', linewidth=2)
    axes[1, 1].plot(epochs, results['aecf']['val_no_images_f1'], 'r-s', label='AECF', linewidth=2)
    axes[1, 1].axvline(x=40, color='green', linestyle='--', alpha=0.7, label='Curriculum ON')
    axes[1, 1].set_title('Validation F1 (No Images)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Macro F1')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mask Rate
    axes[1, 2].plot(epochs, results['aecf']['mask_rate'], 'orange', linewidth=2, label='Mask Rate')
    axes[1, 2].axvline(x=40, color='green', linestyle='--', alpha=0.7, label='Curriculum ON')
    axes[1, 2].set_title('AECF Mask Rate', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Mask Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parallel_training_comparison_with_f1.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_curriculum_effect_analysis(results):
    """Analyze the effect of curriculum activation."""
    print("\n" + "="*80)
    print("🎯 CURRICULUM MASKING ACTIVATION ANALYSIS")
    print("="*80)
    
    # Before vs after curriculum activation
    pre_curriculum = {
        'baseline_full_f1': np.mean(results['baseline']['val_full_f1'][35:40]),
        'aecf_full_f1': np.mean(results['aecf']['val_full_f1'][35:40]),
        'baseline_no_texts_f1': np.mean(results['baseline']['val_no_texts_f1'][35:40]),
        'aecf_no_texts_f1': np.mean(results['aecf']['val_no_texts_f1'][35:40]),
        'gate_entropy': np.mean(results['aecf']['gate_entropy'][35:40])
    }
    
    post_curriculum = {
        'baseline_full_f1': np.mean(results['baseline']['val_full_f1'][55:60]),
        'aecf_full_f1': np.mean(results['aecf']['val_full_f1'][55:60]),
        'baseline_no_texts_f1': np.mean(results['baseline']['val_no_texts_f1'][55:60]),
        'aecf_no_texts_f1': np.mean(results['aecf']['val_no_texts_f1'][55:60]),
        'gate_entropy': np.mean(results['aecf']['gate_entropy'][55:60])
    }
    
    print("📊 BEFORE CURRICULUM (Epochs 35-40):")
    print(f"  Gate Entropy: {pre_curriculum['gate_entropy']:.4f}")
    print(f"  Baseline F1 (Full): {pre_curriculum['baseline_full_f1']:.4f}")
    print(f"  AECF F1 (Full): {pre_curriculum['aecf_full_f1']:.4f}")
    print(f"  Baseline F1 (No Texts): {pre_curriculum['baseline_no_texts_f1']:.4f}")
    print(f"  AECF F1 (No Texts): {pre_curriculum['aecf_no_texts_f1']:.4f}")
    
    print("\n📈 AFTER CURRICULUM (Epochs 55-60):")
    print(f"  Gate Entropy: {post_curriculum['gate_entropy']:.4f}")
    print(f"  Baseline F1 (Full): {post_curriculum['baseline_full_f1']:.4f}")
    print(f"  AECF F1 (Full): {post_curriculum['aecf_full_f1']:.4f}")
    print(f"  Baseline F1 (No Texts): {post_curriculum['baseline_no_texts_f1']:.4f}")
    print(f"  AECF F1 (No Texts): {post_curriculum['aecf_no_texts_f1']:.4f}")
    
    print("\n🔍 CURRICULUM EFFECT:")
    entropy_change = post_curriculum['gate_entropy'] - pre_curriculum['gate_entropy']
    robustness_change = post_curriculum['aecf_no_texts_f1'] - pre_curriculum['aecf_no_texts_f1']
    
    print(f"  Entropy Change: +{entropy_change:.4f} (should be positive!)")
    print(f"  AECF Robustness Change (F1): {robustness_change:+.4f}")
    
    if entropy_change > 0.1:
        print("  ✅ Curriculum masking activated successfully!")
    else:
        print("  ❌ Curriculum masking may not be working properly")
    
    if robustness_change > 0:
        print("  ✅ Curriculum masking improved robustness!")
    else:
        print("  ❌ Curriculum masking did not improve robustness")

def main():
    """Parallel training experiment with curriculum activation and pathology analysis."""
    print("🚀 PARALLEL TRAINING: BASELINE vs AECF WITH CURRICULUM ACTIVATION & F1 ANALYSIS")
    print("Expected: Gate entropy ~0 for epochs 1-40, then jumps up at epoch 40")
    print("Tracking per-pathology F1 improvements from curriculum masking")
    
    try:
        # Load data
        train_loader, val_loader, image_dim, text_dim, num_classes = load_data()
        print(f"Data loaded: {image_dim}D images, {text_dim}D text, {num_classes} classes")
        
        # Create both models
        baseline_model = BaselineModel(image_dim, text_dim, num_classes)
        aecf_model = AECFModel(image_dim, text_dim, num_classes)
        
        print(f"Baseline params: {sum(p.numel() for p in baseline_model.parameters()):,}")
        print(f"AECF params: {sum(p.numel() for p in aecf_model.parameters()):,}")
        
        # Train both models in parallel
        results = train_both_models(baseline_model, aecf_model, train_loader, val_loader, epochs=60)
        
        # Analysis and plotting
        plot_parallel_results(results)
        print_curriculum_effect_analysis(results)
        
        # Pathology-level analysis
        improvement_results = analyze_pathology_improvements(results, num_classes)
        plot_pathology_improvements(improvement_results)
        
        # Save models and results
        torch.save(baseline_model.state_dict(), 'final_baseline_model.pth')
        torch.save(aecf_model.state_dict(), 'final_aecf_model.pth')
        torch.save(results, 'parallel_training_results_with_f1.pth')
        torch.save({
            'improvement_results': improvement_results,
            'pathology_names': PATHOLOGY_NAMES[:num_classes] if num_classes <= len(PATHOLOGY_NAMES) else [f"Pathology_{i}" for i in range(num_classes)]
        }, 'pathology_analysis_results.pth')
        
        print("\n✅ Experiment completed! Check the gate entropy plot for curriculum activation.")
        print("✅ Pathology improvement analysis saved!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
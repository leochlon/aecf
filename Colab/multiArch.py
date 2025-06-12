# -*- coding: utf-8 -*-
"""
Multi-Architecture AECF Drop-in Testing Framework

Tests AECF as a drop-in fusion layer across different network architectures
to demonstrate:
1. Consistent improvements across diverse architectures
2. Simplicity of integration (just replace the fusion module)
3. Robustness across different design patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import your existing components
# from proper_aecf_core import MultimodalAttentionPool, CurriculumMasking

# ============================================================================
# Abstract Base Architecture with Configurable Fusion
# ============================================================================

class FusionInterface(nn.Module):
    """Abstract interface that all fusion methods must implement."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modalities: List of tensors [batch, feature_dim] for each modality
        Returns:
            fused: Tensor [batch, output_dim]
        """
        raise NotImplementedError

class BaseMultimodalArchitecture(nn.Module):
    """Base class for all architectures with configurable fusion."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # These will be implemented by subclasses
        self.image_encoder = None
        self.text_encoder = None
        self.fusion_layer = None
        self.classifier = None
    
    def create_fusion_layer(self, fusion_method: str, input_dims: List[int], 
                          output_dim: int) -> FusionInterface:
        """Factory method to create fusion layers."""
        if fusion_method == 'concat':
            return ConcatenationFusion(input_dims, output_dim)
        elif fusion_method == 'aecf':
            return AECFFusion(input_dims, output_dim)
        elif fusion_method == 'attention':
            return AttentionFusion(input_dims, output_dim)
        elif fusion_method == 'bilinear':
            return BilinearFusion(input_dims, output_dim)
        elif fusion_method == 'transformer':
            return TransformerFusion(input_dims, output_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        """Forward pass - implemented by subclasses."""
        raise NotImplementedError

# ============================================================================
# Fusion Layer Implementations
# ============================================================================

class ConcatenationFusion(FusionInterface):
    """Simple concatenation baseline."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        total_dim = sum(input_dims)
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(modalities, dim=-1)
        return self.projection(concatenated)

class AECFFusion(FusionInterface):
    """AECF-based fusion - the drop-in replacement."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        
        # Ensure all modalities have same dimension for attention
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        # AECF components
        self.curriculum_masking = CurriculumMasking(
            base_mask_prob=0.15,
            entropy_target=0.7,
            min_active=1
        )
        
        self.attention_pool = MultimodalAttentionPool(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            curriculum_masking=self.curriculum_masking,
            batch_first=True
        )
        
        # Learnable fusion query
        self.fusion_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        
        # Store info for analysis
        self.last_fusion_info = {}
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        batch_size = modalities[0].size(0)
        
        # Project all modalities to same dimension
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]
        
        # Stack for attention: [batch, num_modalities, output_dim]
        stacked = torch.stack(projected, dim=1)
        
        # Create query for each sample
        query = self.fusion_query.expand(batch_size, -1, -1)
        
        # Apply AECF attention
        fused, info = self.attention_pool(
            query=query,
            key=stacked,
            value=stacked,
            return_info=True
        )
        
        # Store info for analysis
        self.last_fusion_info = info
        
        return fused.squeeze(1)  # [batch, output_dim]

class AttentionFusion(FusionInterface):
    """Standard attention fusion without curriculum learning."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        batch_size = modalities[0].size(0)
        
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]
        stacked = torch.stack(projected, dim=1)
        
        query = self.fusion_query.expand(batch_size, -1, -1)
        
        fused, _ = self.attention(query, stacked, stacked)
        return fused.squeeze(1)

class BilinearFusion(FusionInterface):
    """Bilinear fusion for two modalities."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        assert len(input_dims) == 2, "Bilinear fusion requires exactly 2 modalities"
        
        self.proj1 = nn.Linear(input_dims[0], output_dim)
        self.proj2 = nn.Linear(input_dims[1], output_dim)
        self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        x1 = self.proj1(modalities[0])
        x2 = self.proj2(modalities[1])
        fused = self.bilinear(x1, x2)
        return self.norm(fused)

class TransformerFusion(FusionInterface):
    """Transformer-based fusion."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        batch_size = modalities[0].size(0)
        
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        stacked = torch.cat([cls_tokens] + [x.unsqueeze(1) for x in projected], dim=1)
        
        # Apply transformer
        output = self.transformer(stacked)
        
        # Return CLS token representation
        return output[:, 0]

# ============================================================================
# Different Architecture Implementations
# ============================================================================

class SimpleMLPArchitecture(BaseMultimodalArchitecture):
    """Simple MLP-based architecture."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        # Simple projections
        hidden_dim = 256
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
        
        # Configurable fusion
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        img_feat = self.image_encoder(batch['image'])
        txt_feat = self.text_encoder(batch['text'])
        
        fused = self.fusion_layer([img_feat, txt_feat])
        logits = self.classifier(fused)
        
        # Return additional info for AECF
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class DeepMLPArchitecture(BaseMultimodalArchitecture):
    """Deeper MLP with residual connections."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        hidden_dim = 512
        
        # Deeper encoders with residuals
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        img_feat = self.image_encoder(batch['image'])
        txt_feat = self.text_encoder(batch['text'])
        
        fused = self.fusion_layer([img_feat, txt_feat])
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class CNNTextArchitecture(BaseMultimodalArchitecture):
    """CNN-based feature processing."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        hidden_dim = 384
        
        # "CNN-like" processing using 1D convolutions
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 1024),  # Expand first
            nn.Unflatten(-1, (64, 16)),  # Reshape for conv
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.Unflatten(-1, (64, 16)),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        img_feat = self.image_encoder(batch['image'])
        txt_feat = self.text_encoder(batch['text'])
        
        fused = self.fusion_layer([img_feat, txt_feat])
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class MultiScaleArchitecture(BaseMultimodalArchitecture):
    """Multi-scale feature processing."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        base_dim = 256
        
        # Multi-scale processing for each modality
        self.image_scales = nn.ModuleList([
            nn.Sequential(nn.Linear(image_dim, base_dim), nn.ReLU()),
            nn.Sequential(nn.Linear(image_dim, base_dim * 2), nn.ReLU(), 
                         nn.Linear(base_dim * 2, base_dim)),
            nn.Sequential(nn.Linear(image_dim, base_dim * 4), nn.ReLU(),
                         nn.Linear(base_dim * 4, base_dim * 2), nn.ReLU(),
                         nn.Linear(base_dim * 2, base_dim))
        ])
        
        self.text_scales = nn.ModuleList([
            nn.Sequential(nn.Linear(text_dim, base_dim), nn.ReLU()),
            nn.Sequential(nn.Linear(text_dim, base_dim * 2), nn.ReLU(),
                         nn.Linear(base_dim * 2, base_dim)),
            nn.Sequential(nn.Linear(text_dim, base_dim * 4), nn.ReLU(),
                         nn.Linear(base_dim * 4, base_dim * 2), nn.ReLU(),
                         nn.Linear(base_dim * 2, base_dim))
        ])
        
        # Aggregate multi-scale features
        self.img_aggregator = nn.Linear(base_dim * 3, base_dim)
        self.txt_aggregator = nn.Linear(base_dim * 3, base_dim)
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [base_dim, base_dim], base_dim
        )
        
        self.classifier = nn.Linear(base_dim, num_classes)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        # Multi-scale processing
        img_features = [scale(batch['image']) for scale in self.image_scales]
        txt_features = [scale(batch['text']) for scale in self.text_scales]
        
        # Aggregate scales
        img_feat = self.img_aggregator(torch.cat(img_features, dim=-1))
        txt_feat = self.txt_aggregator(torch.cat(txt_features, dim=-1))
        
        fused = self.fusion_layer([img_feat, txt_feat])
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class ResNetLikeArchitecture(BaseMultimodalArchitecture):
    """ResNet-inspired architecture with skip connections."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        hidden_dim = 512
        
        # ResNet-like blocks
        self.image_input = nn.Linear(image_dim, hidden_dim)
        self.image_blocks = nn.ModuleList([
            self._make_resnet_block(hidden_dim) for _ in range(3)
        ])
        
        self.text_input = nn.Linear(text_dim, hidden_dim)
        self.text_blocks = nn.ModuleList([
            self._make_resnet_block(hidden_dim) for _ in range(3)
        ])
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def _make_resnet_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        # Process with residual connections
        img_feat = self.image_input(batch['image'])
        for block in self.image_blocks:
            img_feat = img_feat + block(img_feat)  # Skip connection
        
        txt_feat = self.text_input(batch['text'])
        for block in self.text_blocks:
            txt_feat = txt_feat + block(txt_feat)  # Skip connection
        
        fused = self.fusion_layer([img_feat, txt_feat])
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

# ============================================================================
# Multi-Architecture Testing Framework
# ============================================================================

class MultiArchitectureExperiment:
    """Framework to test AECF across multiple architectures."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        
        # Define architectures to test
        self.architectures = {
            'SimpleMLP': SimpleMLPArchitecture,
            'DeepMLP': DeepMLPArchitecture,
            'CNNText': CNNTextArchitecture,
            'MultiScale': MultiScaleArchitecture,
            'ResNetLike': ResNetLikeArchitecture,
        }
        
        # Define fusion methods to compare
        self.fusion_methods = ['concat', 'aecf', 'attention', 'bilinear', 'transformer']
        
        self.results = defaultdict(dict)
    
    def create_model(self, arch_name: str, fusion_method: str):
        """Create a model with specified architecture and fusion method."""
        arch_class = self.architectures[arch_name]
        return arch_class(
            self.image_dim, 
            self.text_dim, 
            self.num_classes, 
            fusion_method
        )
    
    def train_and_evaluate(self, model, train_loader, val_loader, 
                          epochs: int = 10, model_name: str = "Model"):
        """Train and evaluate a single model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        best_map = 0.0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Handle AECF models with entropy loss
                if hasattr(model.fusion_layer, 'last_fusion_info'):
                    logits, fusion_info = model(batch)
                    loss = criterion(logits, batch['label'])
                    
                    # Add entropy regularization for AECF
                    if 'entropy' in fusion_info:
                        entropy_loss = model.fusion_layer.curriculum_masking.entropy_loss(
                            fusion_info['entropy']
                        )
                        loss += 0.01 * entropy_loss
                else:
                    logits = model(batch)
                    loss = criterion(logits, batch['label'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            val_map = self.evaluate_model(model, val_loader)
            if val_map > best_map:
                best_map = val_map
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={np.mean(train_losses):.4f}, mAP={val_map:.4f}")
        
        return best_map
    
    def evaluate_model(self, model, val_loader):
        """Evaluate model with mAP score."""
        model.eval()
        all_preds = []
        all_labels = []
        
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                if hasattr(model.fusion_layer, 'last_fusion_info'):
                    logits, _ = model(batch)
                else:
                    logits = model(batch)
                
                all_preds.append(logits.cpu())
                all_labels.append(batch['label'].cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return self.calculate_map_score(all_preds, all_labels)
    
    def calculate_map_score(self, y_pred, y_true):
        """Calculate mAP score."""
        try:
            from sklearn.metrics import average_precision_score
            y_true_np = y_true.numpy()
            y_pred_np = torch.sigmoid(y_pred).numpy()
            
            valid_classes = y_true_np.sum(axis=0) > 0
            if not valid_classes.any():
                return 0.0
            
            map_score = average_precision_score(
                y_true_np[:, valid_classes], 
                y_pred_np[:, valid_classes], 
                average='macro'
            )
            return map_score
        except:
            return 0.0
    
    def run_comprehensive_experiment(self, train_loader, val_loader, 
                                   epochs_per_model: int = 10):
        """Run experiment across all architectures and fusion methods."""
        print("🚀 Starting Multi-Architecture AECF Experiment")
        print(f"Testing {len(self.architectures)} architectures × {len(self.fusion_methods)} fusion methods")
        print("="*80)
        
        for arch_name in self.architectures:
            print(f"\n🏗️  Testing Architecture: {arch_name}")
            print("-" * 50)
            
            for fusion_method in self.fusion_methods:
                print(f"\n  🔧 Fusion Method: {fusion_method}")
                
                try:
                    # Create model
                    model = self.create_model(arch_name, fusion_method)
                    model_name = f"{arch_name}_{fusion_method}"
                    
                    # Train and evaluate
                    map_score = self.train_and_evaluate(
                        model, train_loader, val_loader, 
                        epochs_per_model, model_name
                    )
                    
                    self.results[arch_name][fusion_method] = map_score
                    print(f"    ✅ Final mAP: {map_score:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    self.results[arch_name][fusion_method] = 0.0
        
        return self.results
    
    def analyze_results(self):
        """Analyze and display results."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE RESULTS ANALYSIS")
        print("="*80)
        
        # Create results table
        print(f"\n{'Architecture':<15} {'Concat':<8} {'AECF':<8} {'Attention':<10} {'Bilinear':<9} {'Transformer':<12} {'AECF Improve':<12}")
        print("-" * 90)
        
        aecf_wins = 0
        total_comparisons = 0
        improvements = []
        
        for arch_name in self.architectures:
            results = self.results[arch_name]
            
            # Get scores
            concat_score = results.get('concat', 0.0)
            aecf_score = results.get('aecf', 0.0)
            attention_score = results.get('attention', 0.0)
            bilinear_score = results.get('bilinear', 0.0)
            transformer_score = results.get('transformer', 0.0)
            
            # Calculate improvement
            improvement = ((aecf_score - concat_score) / concat_score * 100) if concat_score > 0 else 0
            improvements.append(improvement)
            
            # Check if AECF wins
            if aecf_score > concat_score:
                aecf_wins += 1
            total_comparisons += 1
            
            print(f"{arch_name:<15} {concat_score:<8.4f} {aecf_score:<8.4f} {attention_score:<10.4f} "
                  f"{bilinear_score:<9.4f} {transformer_score:<12.4f} {improvement:>+8.1f}%")
        
        # Summary statistics
        avg_improvement = np.mean(improvements) if improvements else 0
        win_rate = (aecf_wins / total_comparisons * 100) if total_comparisons > 0 else 0
        
        print("\n" + "="*80)
        print("📈 SUMMARY STATISTICS")
        print("="*80)
        print(f"🎯 AECF Win Rate: {aecf_wins}/{total_comparisons} ({win_rate:.1f}%)")
        print(f"📊 Average Improvement: {avg_improvement:+.1f}%")
        print(f"🏆 Best Individual Improvement: {max(improvements):+.1f}%")
        print(f"📉 Worst Individual Result: {min(improvements):+.1f}%")
        
        return {
            'results_table': self.results,
            'aecf_win_rate': win_rate,
            'average_improvement': avg_improvement,
            'improvements': improvements
        }
    
    def save_results(self, filename: str = "multi_architecture_results.json"):
        """Save results to file."""
        analysis = self.analyze_results()
        
        final_results = {
            'experiment_type': 'multi_architecture_aecf_test',
            'architectures_tested': list(self.architectures.keys()),
            'fusion_methods_tested': self.fusion_methods,
            'detailed_results': dict(self.results),
            'summary': analysis
        }
        
        Path('./results').mkdir(exist_ok=True)
        with open(f'./results/{filename}', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to ./results/{filename}")


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """Example usage of the multi-architecture testing framework."""
    
    # This would be called with your existing data setup
    # train_loader, val_loader = setup_data(batch_size=256)
    
    # For demo purposes, assume these dimensions
    image_dim = 512  # CLIP image features
    text_dim = 512   # CLIP text features
    num_classes = 80 # COCO classes
    
    # Create experiment
    experiment = MultiArchitectureExperiment(image_dim, text_dim, num_classes)
    
    # Run comprehensive test
    # results = experiment.run_comprehensive_experiment(
    #     train_loader, val_loader, epochs_per_model=8
    # )
    
    # Analyze and save
    # experiment.save_results("aecf_architecture_comparison.json")
    
    print("Framework ready! Call experiment.run_comprehensive_experiment() with your data loaders.")

if __name__ == "__main__":
    main()
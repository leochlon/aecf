#!/usr/bin/env python3
"""
Quick test to validate AECF missing data masking fix
"""

import torch
import torch.nn as nn
import numpy as np
from test_coco import MultimodalClassifier, simulate_missing_modalities_improved, setup_data

def test_aecf_masking():
    print("🧪 Testing AECF Missing Data Masking Fix")
    print("=" * 50)
    
    # Setup
    device = torch.device('cpu')
    
    # Create synthetic batch
    batch_size = 8
    feature_dim = 512
    num_classes = 80
    
    # Create test batch with some known patterns
    batch = {
        'image': torch.randn(batch_size, feature_dim),
        'text': torch.randn(batch_size, feature_dim),
        'label': torch.zeros(batch_size, num_classes)
    }
    
    # Normalize to unit norm (like real data)
    batch['image'] = batch['image'] / torch.norm(batch['image'], dim=1, keepdim=True)
    batch['text'] = batch['text'] / torch.norm(batch['text'], dim=1, keepdim=True)
    
    print(f"Original batch - all features present")
    print(f"  Image norms: {torch.norm(batch['image'], dim=1)}")
    print(f"  Text norms: {torch.norm(batch['text'], dim=1)}")
    
    # Create AECF model
    aecf_model = MultimodalClassifier(feature_dim, feature_dim, num_classes, fusion_type='aecf')
    aecf_model.eval()
    
    # Test 1: Complete data
    print(f"\n📊 Test 1: Complete Data")
    with torch.no_grad():
        logits, info = aecf_model(batch)
        att_weights = info['attention_weights']
        print(f"  Attention shape: {att_weights.shape}")
        print(f"  Average attention: img={att_weights[:, 0, 0].mean():.3f}, txt={att_weights[:, 0, 1].mean():.3f}")
        print(f"  Missing detected: img={info.get('img_missing_count', 0)}, txt={info.get('txt_missing_count', 0)}")
    
    # Test 2: Simulate missing data
    print(f"\n📊 Test 2: With Missing Data (30%)")
    batch_missing = simulate_missing_modalities_improved(batch, 0.3)
    
    print(f"After simulation:")
    img_missing = (torch.norm(batch_missing['image'], dim=1) < 1e-6).sum()
    txt_missing = (torch.norm(batch_missing['text'], dim=1) < 1e-6).sum()
    print(f"  Zero image features: {img_missing}/{batch_size}")
    print(f"  Zero text features: {txt_missing}/{batch_size}")
    
    with torch.no_grad():
        logits, info = aecf_model(batch_missing)
        att_weights = info['attention_weights']
        print(f"  Attention shape: {att_weights.shape}")
        print(f"  Average attention: img={att_weights[:, 0, 0].mean():.3f}, txt={att_weights[:, 0, 1].mean():.3f}")
        print(f"  Missing detected: img={info.get('img_missing_count', 0)}, txt={info.get('txt_missing_count', 0)}")
        
        # Check attention to zero vectors specifically
        img_zero_mask = torch.norm(batch_missing['image'], dim=1) < 1e-6
        txt_zero_mask = torch.norm(batch_missing['text'], dim=1) < 1e-6
        
        if img_zero_mask.any():
            zero_img_att = att_weights[img_zero_mask, 0, 0].mean()
            print(f"  Attention to ZERO images: {zero_img_att:.6f} (should be ~0)")
        
        if txt_zero_mask.any():
            zero_txt_att = att_weights[txt_zero_mask, 0, 1].mean()
            print(f"  Attention to ZERO text: {zero_txt_att:.6f} (should be ~0)")
    
    # Test 3: Extreme case - only one modality present
    print(f"\n📊 Test 3: Only Text Present")
    batch_only_text = batch.copy()
    batch_only_text['image'] = torch.zeros_like(batch['image'])  # All images missing
    
    with torch.no_grad():
        logits, info = aecf_model(batch_only_text)
        att_weights = info['attention_weights']
        print(f"  Average attention: img={att_weights[:, 0, 0].mean():.6f}, txt={att_weights[:, 0, 1].mean():.3f}")
        print(f"  Missing detected: img={info.get('img_missing_count', 0)}, txt={info.get('txt_missing_count', 0)}")
        print(f"  Text should get ~100% attention!")
    
    print(f"\n📊 Test 4: Only Images Present")
    batch_only_img = batch.copy()
    batch_only_img['text'] = torch.zeros_like(batch['text'])  # All text missing
    
    with torch.no_grad():
        logits, info = aecf_model(batch_only_img)
        att_weights = info['attention_weights']
        print(f"  Average attention: img={att_weights[:, 0, 0].mean():.3f}, txt={att_weights[:, 0, 1].mean():.6f}")
        print(f"  Missing detected: img={info.get('img_missing_count', 0)}, txt={info.get('txt_missing_count', 0)}")
        print(f"  Images should get ~100% attention!")
    
    print("\n✅ Masking test completed!")

if __name__ == "__main__":
    test_aecf_masking()
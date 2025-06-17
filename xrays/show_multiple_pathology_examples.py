#!/usr/bin/env python3
"""
Extended X-ray pathology viewer - show multiple examples per pathology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import re

def check_pathology_presence(text, pathology):
    """Check if a pathology is mentioned positively (not negated) in text."""
    pathology_lower = pathology.lower()
    
    # Look for the pathology name in the text
    pattern = r'\b' + re.escape(pathology_lower) + r'\b'
    matches = list(re.finditer(pattern, text))
    
    for match in matches:
        # Check for negation in a window around the match
        window_start = max(0, match.start() - 100)  # 100 chars before
        window_end = min(len(text), match.end() + 50)  # 50 chars after
        context = text[window_start:window_end]
        
        # Negation patterns
        negation_indicators = [
            r'\bno\b', r'\bnot\b', r'\babsence\s+of\b', r'\bwithout\b',
            r'\brule\s+out\b', r'\bruled\s+out\b', r'\bdenies\b',
            r'\bnegative\s+for\b', r'\bfree\s+of\b', r'\bclear\s+of\b',
            r'\bunlikely\b', r'\bexclude\b', r'\bexcluded\b', r'\bnormal\b'
        ]
        
        # Check if any negation appears before the pathology in the context
        is_negated = False
        pathology_pos_in_context = context.find(pathology_lower)
        
        for neg_pattern in negation_indicators:
            neg_matches = list(re.finditer(neg_pattern, context))
            for neg_match in neg_matches:
                # If negation appears before pathology and within reasonable distance
                if neg_match.end() < pathology_pos_in_context and (pathology_pos_in_context - neg_match.end()) < 50:
                    is_negated = True
                    break
            if is_negated:
                break
        
        # If this mention is not negated, the pathology is present
        if not is_negated:
            return True
    
    return False

def find_single_pathology_cases(df, pathology_names):
    """Find cases where only ONE of the target pathologies is present."""
    single_pathology_cases = {pathology: [] for pathology in pathology_names}
    
    print("Analyzing cases for single-pathology examples...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} cases...")
        
        # Combine findings and impression text
        text = f"{row['findings']} {row['impression']}".lower()
        
        # Check which target pathologies are present
        present_pathologies = []
        for pathology in pathology_names:
            if check_pathology_presence(text, pathology):
                present_pathologies.append(pathology)
        
        # If exactly one target pathology is present, add it
        if len(present_pathologies) == 1:
            pathology = present_pathologies[0]
            single_pathology_cases[pathology].append({
                'index': idx,
                'image_data': row['image'],
                'findings': row['findings'],
                'impression': row['impression'],
                'text': text
            })
    
    return single_pathology_cases

def show_multiple_examples():
    """Show multiple examples for each pathology in a grid layout."""
    print("Loading xray.parquet...")
    df = pd.read_parquet('xray.parquet')
    print(f"Loaded {len(df)} X-ray cases")
    
    # Pathologies to extract
    target_pathologies = ['Pneumothorax', 'Effusion', 'Atelectasis', 'Edema']
    
    print("\nSearching for single-pathology cases...")
    pathology_cases = find_single_pathology_cases(df, target_pathologies)
    
    # Create a larger grid: 4 pathologies x 3 examples each
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle('X-ray Images: Multiple Single-Pathology Examples', fontsize=16, fontweight='bold')
    
    for i, pathology in enumerate(target_pathologies):
        cases = pathology_cases[pathology]
        
        # Show up to 3 examples per pathology
        for j in range(3):
            ax = axes[i, j]
            
            if j < len(cases):
                case = cases[j]
                
                try:
                    # Convert binary data to PIL Image
                    image_bytes = case['image_data']
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Display the image
                    ax.imshow(image, cmap='gray')
                    ax.set_title(f'{pathology} #{j+1}\n(Case #{case["index"]})', 
                               fontweight='bold', fontsize=10)
                    ax.axis('off')
                    
                except Exception as e:
                    print(f"❌ Error displaying {pathology} #{j+1}: {e}")
                    ax.text(0.5, 0.5, f'Error loading\n{pathology}\nimage', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral'))
                    ax.set_title(f'{pathology} #{j+1}\n(Error)', fontweight='bold', fontsize=10)
                    ax.axis('off')
            else:
                # No more cases available
                ax.text(0.5, 0.5, f'No more\n{pathology}\ncases', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray'))
                ax.set_title(f'{pathology} #{j+1}\n(N/A)', fontweight='bold', fontsize=10)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('pathology_xray_multiple_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print case details
    print("\n" + "="*80)
    print("PATHOLOGY CASE DETAILS (First 3 examples each)")
    print("="*80)
    
    for pathology in target_pathologies:
        cases = pathology_cases[pathology]
        print(f"\n{pathology.upper()} ({len(cases)} total single-pathology cases):")
        print("-" * 60)
        
        for j, case in enumerate(cases[:3]):
            print(f"Example {j+1} (Index #{case['index']}):")
            print(f"  Findings: {case['findings'][:120]}...")
            print(f"  Impression: {case['impression'][:120]}...")
            print()
    
    return pathology_cases

if __name__ == "__main__":
    pathology_cases = show_multiple_examples()
    print(f"\n✅ Multiple X-ray examples visualization complete!")
    print("📁 Saved as 'pathology_xray_multiple_examples.png'")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    for pathology in ['Pneumothorax', 'Effusion', 'Atelectasis', 'Edema']:
        count = len(pathology_cases[pathology])
        print(f"  {pathology}: {count} single-pathology cases found")

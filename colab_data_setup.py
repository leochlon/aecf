# Download AECF cache from Google Drive and prepare for cocoAblation
from google.colab import drive
import shutil, os
import json
from pathlib import Path

print("🚀 Setting up AECF COCO data for ablation suite...")

# Mount Google Drive
drive.mount('/content/drive')
print("✅ Google Drive mounted")

# Create the cache directory that cocoAblation expects
cache_dir = './cache'
os.makedirs(cache_dir, exist_ok=True)
print(f"✅ Created cache directory: {cache_dir}")

# Original cache files from Google Drive
gdrive_cache_files = [
    "train_60k_clip_feats.pt", 
    "val_5k_clip_feats.pt", 
    "test_5k_clip_feats.pt"
]

# Target filenames that cocoAblation COCODataManager expects
target_cache_files = [
    "coco_clip_cache_train.pt",
    "coco_clip_cache_val.pt", 
    "coco_clip_cache_test.pt"
]

# Copy and rename files to match cocoAblation expectations
gdrive_cache_path = "/content/drive/MyDrive/aecf_cache"

print("📦 Copying and renaming cache files...")
for src_filename, dst_filename in zip(gdrive_cache_files, target_cache_files):
    src_path = f"{gdrive_cache_path}/{src_filename}"
    dst_path = f"{cache_dir}/{dst_filename}"
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        file_size = os.path.getsize(dst_path) / (1024*1024)  # MB
        print(f"✅ {src_filename} -> {dst_filename} ({file_size:.1f} MB)")
    else:
        print(f"❌ {src_filename} not found in Google Drive")

# Create a comprehensive manifest file for COCODataManager
print("📋 Creating dataset manifest...")

manifest = {
    "name": "coco2014",
    "version": "1.0",
    "raw_data_path": "/content/coco2014",
    "cache_path": "./cache",
    "files_required": {
        "annotations/instances_train2014.json": "",
        "annotations/instances_val2014.json": "",
        "annotations/captions_train2014.json": "",
        "annotations/captions_val2014.json": "",
        "train2014": "",
        "val2014": ""
    },
    "cache_files": {
        "coco_clip_cache_train.pt": "",
        "coco_clip_cache_val.pt": "",
        "coco_clip_cache_test.pt": "",
        "coco_manifest.json": ""
    },
    "metadata": {
        "num_classes": 80,
        "feature_dim": 512,
        "splits": ["train", "val", "test"],
        "data_format": "ClipTensor",
        "dtype": "float32",
        "label_type": "multi_label_classification",
        "description": "COCO 2014 dataset with pre-extracted CLIP features",
        "compatibility": "AECF_CLIP model ready"
    }
}

# Save manifest that COCODataManager will use for validation
manifest_path = f"{cache_dir}/coco_manifest.json"
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Created manifest: {manifest_path}")

# Verify all files are in place and ready for cocoAblation
print("\n🔍 Cache verification for cocoAblation compatibility:")
all_files_ready = True

for filename in target_cache_files + ["coco_manifest.json"]:
    filepath = f"{cache_dir}/{filename}"
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / (1024*1024)  # MB
        print(f"✅ {filename} ({file_size:.1f} MB)")
    else:
        print(f"❌ {filename} missing")
        all_files_ready = False

if all_files_ready:
    print(f"\n🎯 SUCCESS: Cache is ready for cocoAblation!")
    print(f"📁 Cache location: {cache_dir}/")
    print(f"🔧 COCODataManager will validate and use these files")
    print(f"⚡ Feature extraction will be skipped (using cached CLIP features)")
    print(f"\n🚀 Ready to run ablation suite:")
    print(f"   python test/cocoAblation.py --quick --ablations full no_gate")
    print(f"   python test/cocoAblation.py --ablations full no_gate no_entropy img_only txt_only")
else:
    print(f"\n❌ Setup incomplete - some files are missing")
    print(f"Please check your Google Drive cache directory: {gdrive_cache_path}")

print(f"\n📊 Expected data pipeline:")
print(f"1. COCODataManager.ensure_data_ready() validates cache")
print(f"2. Comprehensive dtype/shape checking (float32, [batch, 512], [batch, 80])")
print(f"3. Data loaders created with validated tensors")
print(f"4. AECF_CLIP model receives properly formatted features")
print(f"5. Ablation experiments run with rigorous consistency checks")

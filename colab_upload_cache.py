#!/usr/bin/env python3
"""
Colab-optimized script to upload .pt cache files to Google Drive.
Run this directly in your Colab notebook after generating the cache files.
"""

import os
from pathlib import Path
import time

def setup_gdrive_colab():
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive
        print("🔗 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        return True
    except ImportError:
        print("❌ This script requires Google Colab environment")
        return False
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

def find_cache_files(cache_dir="/content/coco2014"):
    """Find all .pt cache files and split JSON."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return []
    
    # Look for cache files
    cache_files = []
    
    # Standard cache file patterns
    patterns = [
        "*_clip_feats.pt",
        "splits_*.json",
        "*_60k_*.pt",
        "*_5k_*.pt"
    ]
    
    for pattern in patterns:
        found_files = list(cache_path.glob(pattern))
        cache_files.extend(found_files)
    
    # Remove duplicates
    cache_files = list(set(cache_files))
    
    if cache_files:
        print(f"📁 Found {len(cache_files)} cache files:")
        for f in cache_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"❌ No cache files found in {cache_dir}")
        print("   Looking for files matching: *_clip_feats.pt, splits_*.json")
    
    return cache_files

def copy_to_gdrive(cache_files, gdrive_folder="AECF_Cache"):
    """Copy cache files to Google Drive."""
    if not cache_files:
        print("❌ No cache files to copy")
        return False
    
    # Create target directory in Google Drive
    gdrive_path = Path("/content/drive/MyDrive") / gdrive_folder
    gdrive_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Copying files to Google Drive folder: {gdrive_folder}")
    
    total_size = 0
    copied_files = []
    
    for cache_file in cache_files:
        try:
            target_path = gdrive_path / cache_file.name
            
            # Check if file already exists and has same size
            if target_path.exists():
                local_size = cache_file.stat().st_size
                remote_size = target_path.stat().st_size
                
                if local_size == remote_size:
                    print(f"⏭️  {cache_file.name} already exists (same size), skipping")
                    continue
                else:
                    print(f"🔄 {cache_file.name} exists but different size, overwriting")
            
            print(f"📤 Copying {cache_file.name}...")
            start_time = time.time()
            
            # Copy file
            import shutil
            shutil.copy2(cache_file, target_path)
            
            duration = time.time() - start_time
            file_size = cache_file.stat().st_size / (1024 * 1024)
            total_size += file_size
            
            print(f"✅ Copied {cache_file.name} ({file_size:.1f} MB) in {duration:.1f}s")
            copied_files.append(cache_file.name)
            
        except Exception as e:
            print(f"❌ Failed to copy {cache_file.name}: {e}")
    
    if copied_files:
        print(f"\n🎉 Successfully copied {len(copied_files)} files ({total_size:.1f} MB total)")
        print(f"📍 Location: /content/drive/MyDrive/{gdrive_folder}/")
        
        # Create a manifest file
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "files": [
                {
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
                }
                for f in cache_files if f.name in copied_files
            ],
            "total_size_mb": round(total_size, 2),
            "usage": "Download these files to your Colab runtime and use make_clip_tensor_loaders_from_cache()"
        }
        
        manifest_path = gdrive_path / "cache_manifest.json"
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"📋 Created manifest: cache_manifest.json")
        return True
    else:
        print("❌ No files were copied")
        return False

def create_download_script(gdrive_folder="AECF_Cache"):
    """Create a script to download the cache files in future sessions."""
    
    download_script = f'''
# Run this code in a new Colab session to download cached features
from google.colab import drive
from pathlib import Path
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Set up paths
gdrive_cache = Path("/content/drive/MyDrive/{gdrive_folder}")
local_cache = Path("/content/coco2014")
local_cache.mkdir(parents=True, exist_ok=True)

# Download cache files
cache_files = list(gdrive_cache.glob("*.pt")) + list(gdrive_cache.glob("*.json"))

print(f"📥 Downloading {{len(cache_files)}} cache files...")
for cache_file in cache_files:
    if cache_file.name != "cache_manifest.json":  # Skip manifest
        local_path = local_cache / cache_file.name
        print(f"⬇️  {{cache_file.name}}")
        shutil.copy2(cache_file, local_path)

print("✅ Cache files downloaded! You can now run:")
print("from aecf.datasets import make_clip_tensor_loaders_from_cache")
print("dl_tr, dl_va, dl_te = make_clip_tensor_loaders_from_cache('/content/coco2014')")
'''
    
    gdrive_path = Path("/content/drive/MyDrive") / gdrive_folder
    script_path = gdrive_path / "download_cache.py"
    
    with open(script_path, 'w') as f:
        f.write(download_script)
    
    print(f"📜 Created download script: {script_path}")
    print("💡 Copy and run this script in future Colab sessions to download the cache")

def main():
    """Main function to upload cache files to Google Drive."""
    
    print("🚀 AECF Cache Upload to Google Drive")
    print("=" * 50)
    
    # Step 1: Mount Google Drive
    if not setup_gdrive_colab():
        return
    
    # Step 2: Find cache files
    cache_files = find_cache_files()
    if not cache_files:
        print("\\n💡 TIP: Make sure you've generated the cache files first:")
        print("   from aecf.datasets import setup_coco_cache_pipeline")
        print("   setup_coco_cache_pipeline('/content/coco2014')")
        return
    
    # Step 3: Copy to Google Drive
    if copy_to_gdrive(cache_files):
        # Step 4: Create download script for future use
        create_download_script()
        
        print("\\n🎉 UPLOAD COMPLETE!")
        print("📋 Summary:")
        print(f"  • Uploaded cache files to Google Drive")
        print(f"  • Created download script for future sessions")
        print(f"  • Total files: {len(cache_files)}")
        
        print("\\n💡 Next time you start a new runtime:")
        print("  1. Mount Google Drive")
        print("  2. Run the download_cache.py script")
        print("  3. Use the cached features immediately!")
    else:
        print("❌ Upload failed")

if __name__ == "__main__":
    main()

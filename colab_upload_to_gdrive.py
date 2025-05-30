#!/usr/bin/env python3
"""
Colab script to upload .pt cache files to Google Drive
Run this in a Colab cell after generating your cache files
"""

import os
import glob
from pathlib import Path

def mount_gdrive():
    """Mount Google Drive in Colab"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        return True
    except ImportError:
        print("❌ Not running in Google Colab")
        return False
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

def create_cache_folder(gdrive_path="/content/drive/MyDrive/aecf_cache"):
    """Create cache folder in Google Drive"""
    os.makedirs(gdrive_path, exist_ok=True)
    print(f"📁 Created/verified folder: {gdrive_path}")
    return gdrive_path

def upload_cache_files(local_cache_dir="./cache", gdrive_cache_dir="/content/drive/MyDrive/aecf_cache"):
    """Upload all .pt files from local cache to Google Drive"""
    
    # Find all .pt files
    pt_files = glob.glob(os.path.join(local_cache_dir, "*.pt"))
    
    if not pt_files:
        print(f"❌ No .pt files found in {local_cache_dir}")
        return False
    
    print(f"📦 Found {len(pt_files)} cache files to upload:")
    for file in pt_files:
        print(f"  - {os.path.basename(file)}")
    
    # Upload each file
    uploaded_files = []
    for local_file in pt_files:
        filename = os.path.basename(local_file)
        gdrive_file = os.path.join(gdrive_cache_dir, filename)
        
        try:
            print(f"📤 Uploading {filename}...")
            
            # Use shutil.copy2 to preserve metadata
            import shutil
            shutil.copy2(local_file, gdrive_file)
            
            # Verify upload
            if os.path.exists(gdrive_file):
                local_size = os.path.getsize(local_file)
                gdrive_size = os.path.getsize(gdrive_file)
                
                if local_size == gdrive_size:
                    print(f"  ✅ {filename} uploaded successfully ({local_size:,} bytes)")
                    uploaded_files.append(filename)
                else:
                    print(f"  ⚠️  {filename} size mismatch (local: {local_size}, gdrive: {gdrive_size})")
            else:
                print(f"  ❌ {filename} upload failed")
                
        except Exception as e:
            print(f"  ❌ Failed to upload {filename}: {e}")
    
    print(f"\n🎉 Successfully uploaded {len(uploaded_files)}/{len(pt_files)} files")
    return uploaded_files

def generate_download_script(gdrive_cache_dir="/content/drive/MyDrive/aecf_cache"):
    """Generate a download script for future use"""
    
    pt_files = glob.glob(os.path.join(gdrive_cache_dir, "*.pt"))
    
    download_script = '''# Download AECF cache files from Google Drive
# Run this in a new Colab session to download pre-computed cache files

from google.colab import drive
import shutil
import os

# Mount Google Drive
drive.mount('/content/drive')

# Create local cache directory
os.makedirs('./cache', exist_ok=True)

# Download cache files
gdrive_cache = "/content/drive/MyDrive/aecf_cache"
local_cache = "./cache"

cache_files = [
'''
    
    for file in pt_files:
        filename = os.path.basename(file)
        download_script += f'    "{filename}",\n'
    
    download_script += ''']

print(f"📥 Downloading {len(cache_files)} cache files...")
for filename in cache_files:
    gdrive_file = os.path.join(gdrive_cache, filename)
    local_file = os.path.join(local_cache, filename)
    
    if os.path.exists(gdrive_file):
        shutil.copy2(gdrive_file, local_file)
        print(f"  ✅ Downloaded {filename}")
    else:
        print(f"  ❌ {filename} not found in Google Drive")

print("🎉 Cache download complete!")
'''
    
    # Save download script to Google Drive
    script_path = os.path.join(gdrive_cache_dir, "download_cache.py")
    with open(script_path, 'w') as f:
        f.write(download_script)
    
    print(f"📝 Download script saved to: {script_path}")
    print("\n📋 Copy this code for future Colab sessions:")
    print("=" * 50)
    print(download_script)
    print("=" * 50)

def main():
    """Main upload workflow"""
    print("🚀 AECF Cache Upload to Google Drive")
    print("=" * 40)
    
    # Step 1: Mount Google Drive
    if not mount_gdrive():
        return
    
    # Step 2: Create cache folder
    gdrive_cache_dir = create_cache_folder()
    
    # Step 3: Upload cache files
    uploaded_files = upload_cache_files()
    
    if uploaded_files:
        # Step 4: Generate download script
        generate_download_script(gdrive_cache_dir)
        
        print(f"\n🎯 Next steps:")
        print(f"1. Your cache files are now in Google Drive at: {gdrive_cache_dir}")
        print(f"2. Use the generated download script in future Colab sessions")
        print(f"3. Share the download script with collaborators")
    
    return uploaded_files

if __name__ == "__main__":
    main()

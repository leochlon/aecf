# AECF Cache Upload to Google Drive
# Copy and paste this entire cell into Google Colab

import os
import glob
import shutil

def upload_aecf_cache():
    """Upload AECF cache files to Google Drive"""
    
    print("🚀 Uploading AECF cache to Google Drive...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except:
        print("❌ Failed to mount Google Drive")
        return
    
    # Create cache folder in Google Drive
    gdrive_cache = "/content/drive/MyDrive/aecf_cache"
    os.makedirs(gdrive_cache, exist_ok=True)
    print(f"📁 Cache folder: {gdrive_cache}")
    
    # Find and upload .pt files
    pt_files = glob.glob("./cache/*.pt") + glob.glob("./*.pt")
    
    if not pt_files:
        print("❌ No .pt cache files found")
        return
    
    print(f"📦 Found {len(pt_files)} cache files:")
    uploaded = 0
    
    for local_file in pt_files:
        filename = os.path.basename(local_file)
        gdrive_file = os.path.join(gdrive_cache, filename)
        
        try:
            shutil.copy2(local_file, gdrive_file)
            local_size = os.path.getsize(local_file)
            gdrive_size = os.path.getsize(gdrive_file)
            
            if local_size == gdrive_size:
                print(f"  ✅ {filename} ({local_size:,} bytes)")
                uploaded += 1
            else:
                print(f"  ⚠️  {filename} size mismatch")
        except Exception as e:
            print(f"  ❌ {filename} failed: {e}")
    
    print(f"\n🎉 Uploaded {uploaded}/{len(pt_files)} files to Google Drive")
    
    # Generate download script
    download_code = '''# Download AECF cache from Google Drive
from google.colab import drive
import shutil, os

drive.mount('/content/drive')
os.makedirs('./cache', exist_ok=True)

cache_files = ['''
    
    for file in glob.glob(os.path.join(gdrive_cache, "*.pt")):
        filename = os.path.basename(file)
        download_code += f'"{filename}", '
    
    download_code += ''']

for filename in cache_files:
    src = f"/content/drive/MyDrive/aecf_cache/{filename}"
    dst = f"./cache/{filename}"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✅ {filename}")
    else:
        print(f"❌ {filename} not found")
'''
    
    # Save download script
    script_path = os.path.join(gdrive_cache, "download_cache.py")
    with open(script_path, 'w') as f:
        f.write(download_code)
    
    print(f"\n📝 Download script saved to Google Drive")
    print("\n📋 Future download code:")
    print("=" * 40)
    print(download_code)
    print("=" * 40)

# Run the upload
upload_aecf_cache()

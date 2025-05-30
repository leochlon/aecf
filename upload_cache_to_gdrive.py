#!/usr/bin/env python3
"""
Upload cached CLIP features to Google Drive for reuse across runtime sessions.

This script uploads the .pt cache files and split JSON to Google Drive,
so you don't have to regenerate CLIP features every time you start a new runtime.

Usage:
    python upload_cache_to_gdrive.py --cache_dir ./test_coco2014 --gdrive_folder "AECF_Cache"
    
Prerequisites:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional
import time

def install_requirements():
    """Install required Google Drive API packages."""
    try:
        import googleapiclient
        import google_auth_oauthlib
        import google.auth
        print("✅ Google Drive API packages already installed")
        return True
    except ImportError:
        print("📦 Installing Google Drive API packages...")
        import subprocess
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "google-api-python-client", 
                "google-auth-httplib2", 
                "google-auth-oauthlib"
            ])
            print("✅ Google Drive API packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False

def authenticate_gdrive():
    """Authenticate with Google Drive API."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        
        # If modifying these scopes, delete the file token.json.
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        
        creds = None
        token_file = Path.home() / ".gdrive_token.json"
        
        # The file token.json stores the user's access and refresh tokens.
        if token_file.exists():
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # For Colab environment
                if 'google.colab' in sys.modules:
                    from google.colab import auth
                    auth.authenticate_user()
                    from google.auth import default
                    creds, _ = default()
                else:
                    # For local environment - need credentials.json
                    creds_file = Path("credentials.json")
                    if not creds_file.exists():
                        print("❌ credentials.json not found!")
                        print("💡 To set up Google Drive API:")
                        print("1. Go to https://console.developers.google.com/")
                        print("2. Create a new project or select existing")
                        print("3. Enable Google Drive API")
                        print("4. Create credentials (OAuth 2.0)")
                        print("5. Download credentials.json to this directory")
                        return None
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_file), SCOPES)
                    creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive authentication successful")
        return service
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None

def create_folder(service, folder_name: str, parent_id: str = None) -> Optional[str]:
    """Create a folder in Google Drive."""
    try:
        # Check if folder already exists
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and parents in '{parent_id}'"
        
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if items:
            print(f"📁 Folder '{folder_name}' already exists")
            return items[0]['id']
        
        # Create new folder
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
            
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        print(f"📁 Created folder '{folder_name}' with ID: {folder_id}")
        return folder_id
        
    except Exception as e:
        print(f"❌ Failed to create folder: {e}")
        return None

def upload_file(service, file_path: Path, folder_id: str) -> bool:
    """Upload a single file to Google Drive."""
    try:
        from googleapiclient.http import MediaFileUpload
        
        # Check if file already exists
        query = f"name='{file_path.name}' and parents in '{folder_id}'"
        results = service.files().list(q=query, fields="files(id, name, size)").execute()
        items = results.get('files', [])
        
        local_size = file_path.stat().st_size
        
        if items:
            remote_size = int(items[0].get('size', 0))
            if remote_size == local_size:
                print(f"⏭️  {file_path.name} already exists with same size, skipping")
                return True
            else:
                print(f"🔄 {file_path.name} exists but different size, updating...")
                file_id = items[0]['id']
                # Update existing file
                media = MediaFileUpload(str(file_path), resumable=True)
                service.files().update(fileId=file_id, media_body=media).execute()
                print(f"✅ Updated {file_path.name}")
                return True
        
        # Upload new file
        file_metadata = {
            'name': file_path.name,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(str(file_path), resumable=True)
        
        print(f"📤 Uploading {file_path.name} ({local_size / (1024*1024):.1f} MB)...")
        start_time = time.time()
        
        request = service.files().create(body=file_metadata, media_body=media, fields='id')
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress = status.progress() * 100
                print(f"   Progress: {progress:.1f}%", end='\r')
        
        duration = time.time() - start_time
        print(f"✅ Uploaded {file_path.name} in {duration:.1f}s")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload {file_path.name}: {e}")
        return False

def upload_cache_files(cache_dir: Path, gdrive_folder: str = "AECF_Cache"):
    """Upload all cache files to Google Drive."""
    
    print("🚀 AECF Cache Upload to Google Drive")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Authenticate
    service = authenticate_gdrive()
    if not service:
        return False
    
    # Find cache files
    cache_files = []
    
    # Look for .pt files and split JSON
    patterns = ["*.pt", "splits_*.json"]
    for pattern in patterns:
        cache_files.extend(cache_dir.glob(pattern))
    
    if not cache_files:
        print(f"❌ No cache files found in {cache_dir}")
        print("Expected files: train_*_clip_feats.pt, val_*_clip_feats.pt, test_*_clip_feats.pt, splits_*.json")
        return False
    
    print(f"📁 Found {len(cache_files)} cache files:")
    total_size = 0
    for f in cache_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   {f.name} ({size_mb:.1f} MB)")
    print(f"📊 Total size: {total_size:.1f} MB")
    
    # Create folder in Google Drive
    folder_id = create_folder(service, gdrive_folder)
    if not folder_id:
        return False
    
    # Upload files
    print(f"\n📤 Uploading to Google Drive folder: {gdrive_folder}")
    success_count = 0
    
    for file_path in cache_files:
        if upload_file(service, file_path, folder_id):
            success_count += 1
        else:
            print(f"❌ Failed to upload {file_path.name}")
    
    # Create metadata file
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [f.name for f in cache_files],
        "total_size_mb": total_size,
        "description": "AECF COCO CLIP features cache"
    }
    
    metadata_file = cache_dir / "cache_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    upload_file(service, metadata_file, folder_id)
    
    print(f"\n🎉 Upload complete! {success_count}/{len(cache_files)} files uploaded")
    print(f"📁 Google Drive folder: {gdrive_folder}")
    print(f"🔗 Access at: https://drive.google.com/drive/folders/{folder_id}")
    
    return success_count == len(cache_files)

def create_download_script(gdrive_folder: str = "AECF_Cache"):
    """Create a companion script to download files from Google Drive."""
    
    download_script = f"""#!/usr/bin/env python3
'''
Download AECF cache files from Google Drive.

Usage:
    python download_cache_from_gdrive.py --output_dir ./test_coco2014
'''

import os
import sys
from pathlib import Path
import argparse

def download_from_gdrive():
    '''Download cache files from Google Drive.'''
    
    # Install requirements if needed
    try:
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        print("Installing Google Drive API packages...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib"
        ])
    
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./test_coco2014", help="Output directory")
    parser.add_argument("--folder_name", default="{gdrive_folder}", help="Google Drive folder name")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔐 Authenticating with Google Drive...")
    
    # Simple authentication for Colab
    if 'google.colab' in sys.modules:
        from google.colab import auth
        auth.authenticate_user()
        from google.auth import default
        creds, _ = default()
    else:
        # Load saved credentials
        token_file = Path.home() / ".gdrive_token.json"
        if not token_file.exists():
            print("❌ No saved credentials found. Run upload script first.")
            return False
        creds = Credentials.from_authorized_user_file(str(token_file))
    
    service = build('drive', 'v3', credentials=creds)
    
    # Find folder
    query = f"name='{args.folder_name}' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])
    
    if not folders:
        print(f"❌ Folder '{args.folder_name}' not found in Google Drive")
        return False
    
    folder_id = folders[0]['id']
    print(f"📁 Found folder: {args.folder_name}")
    
    # List files in folder
    query = f"parents in '{folder_id}'"
    results = service.files().list(q=query, fields="files(id, name, size)").execute()
    files = results.get('files', [])
    
    print(f"📥 Downloading {len(files)} files...")
    
    for file_info in files:
        file_name = file_info['name']
        file_id = file_info['id']
        output_path = output_dir / file_name
        
        if output_path.exists():
            print(f"⏭️  {file_name} already exists, skipping")
            continue
            
        print(f"📥 Downloading {file_name}...")
        
        import io
        from googleapiclient.http import MediaIoBaseDownload
        
        request = service.files().get_media(fileId=file_id)
        with open(output_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    progress = status.progress() * 100
                    print(f"   Progress: {progress:.1f}%", end='\\r')
        
        print(f"✅ Downloaded {file_name}")
    
    print(f"🎉 Download complete! Files saved to {output_dir}")
    return True

if __name__ == "__main__":
    download_from_gdrive()
"""
    
    script_path = Path("download_cache_from_gdrive.py")
    with open(script_path, 'w') as f:
        f.write(download_script)
    
    print(f"📄 Created companion download script: {script_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Upload AECF cache files to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload from default cache directory
    python upload_cache_to_gdrive.py
    
    # Upload from custom directory
    python upload_cache_to_gdrive.py --cache_dir /path/to/cache --gdrive_folder "My_AECF_Cache"
    
    # Upload and create download script
    python upload_cache_to_gdrive.py --create_download_script
        """
    )
    
    parser.add_argument(
        "--cache_dir", 
        type=Path, 
        default="./test_coco2014",
        help="Directory containing cache files (default: ./test_coco2014)"
    )
    
    parser.add_argument(
        "--gdrive_folder", 
        default="AECF_Cache",
        help="Google Drive folder name (default: AECF_Cache)"
    )
    
    parser.add_argument(
        "--create_download_script",
        action="store_true",
        help="Create companion download script"
    )
    
    args = parser.parse_args()
    
    if not args.cache_dir.exists():
        print(f"❌ Cache directory not found: {args.cache_dir}")
        print("💡 Make sure you have run the CLIP feature extraction first")
        return False
    
    success = upload_cache_files(args.cache_dir, args.gdrive_folder)
    
    if args.create_download_script:
        create_download_script(args.gdrive_folder)
    
    if success:
        print("\n💡 NEXT STEPS:")
        print("1. Your cache files are now safely stored in Google Drive")
        print("2. In new runtime sessions, run:")
        print("   python download_cache_from_gdrive.py")
        print("3. This will restore your cache files instantly!")
    
    return success

if __name__ == "__main__":
    main()

import os
import urllib.request
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time
import shutil

# 1. Load Environment Configuration
load_dotenv()
data_drive_root = os.getenv("DATA_DRIVE_PATH")

if not data_drive_root:
    print("[!] Warning: DATA_DRIVE_PATH not found in .env. Falling back to local ./data folder.")
    data_drive_root = Path(__file__).resolve().parent.parent.parent / "data"

# Setup Hard Drive Directories
RAW_DATA_DIR = Path(data_drive_root) / "raw"
INTERIM_DIR = Path(data_drive_root) / "interim"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def download_with_retry(url, dest_path, retries=5):
    """Robust downloader that retries on network failure."""
    for attempt in range(retries):
        try:
            # Use a longer timeout and chunked file copying for stability
            with urllib.request.urlopen(url, timeout=120) as response, open(dest_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            return True
        except Exception as e:
            print(f"        [!] Network drop on attempt {attempt + 1}/{retries}: {e}")
            time.sleep(3) # Wait 3 seconds before retrying
            
    return False

def download_recent_indian_ocean_data():
    """
    Downloads the ARGO global index, filters for recent Indian Ocean profiles,
    and downloads the raw NetCDF files securely.
    """
    print("=" * 70)
    print("AcousticGuard-OS | Phase 1: Robust FTP Ingestion (Resilient Mode)")
    print("=" * 70)
    
    index_url = "ftp://ftp.ifremer.fr/ifremer/argo/ar_index_global_prof.txt"
    index_file = INTERIM_DIR / "ar_index_global_prof.txt"
    
    # Step 1: Secure the Global Index
    if not index_file.exists():
        print("[*] Downloading global ARGO index (~300MB, this may take a few minutes)...")
        success = download_with_retry(index_url, index_file)
        if not success:
            print("[!] FATAL: Failed to download the global index after 5 attempts. Check your connection.")
            return
        print("[+] Global index downloaded securely.")
    else:
        print("[*] Global ARGO index found on hard drive. Skipping download.")

    # Step 2: Load and Filter the Index
    print("[*] Parsing global index database...")
    # The ARGO index has 8 lines of metadata headers before the CSV data begins
    df = pd.read_csv(index_file, skiprows=8)
    
    # Convert dates to strings for easy filtering
    df['date_str'] = df['date'].astype(str)
    
    # Filter 1: Temporal (July 2026 - strictly recent data)
    recent_df = df[df['date_str'].str.startswith('202607', na=False)]
    
    # Filter 2: Spatial (Indian Subcontinent & Arabian Sea/Bay of Bengal)
    target_df = recent_df[
        (recent_df['latitude'] >= 5.0) & 
        (recent_df['latitude'] <= 25.0) & 
        (recent_df['longitude'] >= 65.0) & 
        (recent_df['longitude'] <= 95.0)
    ]
    
    print(f"[*] Found {len(target_df)} profiles matching the Indian Coast for July 2026.")
    
    # Step 3: Download the NetCDF Files
    ftp_base = "ftp://ftp.ifremer.fr/ifremer/argo/dac/"
    download_count = 0
    
    print("[*] Commencing resilient NetCDF downloads to external hard drive...")
    for idx, row in target_df.iterrows():
        file_path = row['file']
        filename = file_path.split('/')[-1]
        local_path = RAW_DATA_DIR / filename
        
        if not local_path.exists():
            download_url = ftp_base + file_path
            print(f"    -> Fetching: {filename}...")
            if download_with_retry(download_url, local_path):
                download_count += 1
            else:
                print(f"    [!] Skipping {filename} due to repeated network failures.")
                
    print("=" * 70)
    print(f"[+] Acquisition Complete! Downloaded {download_count} new .nc files.")
    print(f"[+] Data securely stored at: {RAW_DATA_DIR}")

if __name__ == "__main__":
    download_recent_indian_ocean_data()
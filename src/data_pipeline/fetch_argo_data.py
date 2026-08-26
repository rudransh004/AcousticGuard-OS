import os
from pathlib import Path
from dotenv import load_dotenv
import argopy
from argopy import DataFetcher as ArgoDataFetcher

# 1. Load Environment Configuration
load_dotenv()
data_drive_root = os.getenv("DATA_DRIVE_PATH")

if not data_drive_root:
    # Fallback to local data folder if .env is missing
    RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
else:
    RAW_DATA_DIR = Path(data_drive_root) / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_indian_ocean_argo(
    lon_min: float = 60.0,
    lon_max: float = 95.0,
    lat_min: float = -10.0,
    lat_max: float = 25.0,
    depth_min: float = 0.0,
    depth_max: float = 2000.0,
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-31",
    output_filename: str = "argo_indian_ocean_batch.nc"
):
    """
    Fetches raw ARGO profiles for a defined spatio-temporal bounding box
    and stores the resulting multidimensional dataset as NetCDF on the external drive.
    """
    print("=" * 70)
    print("AcousticGuard-OS | Phase 1: Programmatic ARGO Ingestion")
    print("=" * 70)
    print(f"[*] Target Region  : Lon [{lon_min}, {lon_max}], Lat [{lat_min}, {lat_max}]")
    print(f"[*] Depth Envelope : {depth_min} m to {depth_max} m")
    print(f"[*] Time Window    : {start_date} to {end_date}")
    print(f"[*] Destination    : {RAW_DATA_DIR / output_filename}")
    print("=" * 70)

    try:
        # Configure argopy to fetch data using the standard ERDDAP client
        argopy.set_options(src='erddap', mode='standard')
        fetcher = ArgoDataFetcher()

        print("[*] Contacting global ARGO data servers...")
        box = [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, start_date, end_date]
        
        # Query metadata and profile observations
        argo_ds = fetcher.region(box).to_xarray()
        
        num_profiles = len(set(argo_ds['PLATFORM_NUMBER'].values)) if 'PLATFORM_NUMBER' in argo_ds else "N/A"
        total_points = len(argo_ds['N_POINTS']) if 'N_POINTS' in argo_ds.dims else "N/A"
        
        print(f"[+] Download complete!")
        print(f"    - Unique Floats Found: {num_profiles}")
        print(f"    - Total Data Points  : {total_points}")

        # Save to external hard drive
        output_path = RAW_DATA_DIR / output_filename
        argo_ds.to_netcdf(output_path)
        print(f"[+] Successfully saved raw multi-profile NetCDF dataset to: {output_path}")

        return output_path

    except Exception as e:
        print(f"[!] Error fetching ARGO data: {e}")
        print("[!] Troubleshooting tip: Check internet connection or reduce the date range.")
        return None


if __name__ == "__main__":
    # Recent data (July 2026) near the Indian Subcontinent 
    # (Arabian Sea, Bay of Bengal, and Northern Indian Ocean)
    download_indian_ocean_argo(
        lon_min=65.0,
        lon_max=95.0,
        lat_min=5.0,
        lat_max=25.0,
        depth_min=0.0,
        depth_max=2000.0,
        start_date="2026-07-01",
        end_date="2026-07-31",
        output_filename="argo_india_recent_jul2026.nc"
    )
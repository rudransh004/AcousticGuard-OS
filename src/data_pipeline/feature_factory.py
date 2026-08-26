import os
import xarray as xr
import pandas as pd
import numpy as np
import gsw
from pathlib import Path
from dotenv import load_dotenv

# 1. Load Environment Configuration
load_dotenv()
data_drive_root = os.getenv("DATA_DRIVE_PATH")

if not data_drive_root:
    print("[!] Warning: DATA_DRIVE_PATH not found in .env. Falling back to local ./data folder.")
    data_drive_root = Path(__file__).resolve().parent.parent.parent / "data"

# Setup Hard Drive Directories
RAW_DATA_DIR = Path(data_drive_root) / "raw"
PROCESSED_DIR = Path(data_drive_root) / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_all_profiles():
    print("=" * 70)
    print("AcousticGuard-OS | Phase 4: Scaled Ocean Feature Factory")
    print("=" * 70)
    
    # Grab all downloaded NetCDF files from the hard drive
    nc_files = list(RAW_DATA_DIR.glob("*.nc"))
    if not nc_files:
        print("[!] No .nc files found in the raw directory. Run the fetcher first.")
        return

    print(f"[*] Found {len(nc_files)} NetCDF profiles. Processing TEOS-10 thermodynamics...")
    
    all_profiles_df = []
    processed_count = 0

    for file_path in nc_files:
        try:
            # 1. Load the NetCDF file
            ds = xr.open_dataset(file_path)
            
            # 2. Extract base variables securely
            pres = ds['PRES'].squeeze().values.flatten()
            temp = ds['TEMP'].squeeze().values.flatten()
            psal = ds['PSAL'].squeeze().values.flatten()
            
            # Extract coordinates (averaging them safely in case of slight sensor drift)
            lat = float(np.nanmean(ds['LATITUDE'].values))
            lon = float(np.nanmean(ds['LONGITUDE'].values))
            
            # Extract Timestamp
            try:
                time_val = pd.to_datetime(ds['JULD'].values[0])
            except:
                time_val = pd.NaT

            # 3. Create a clean Pandas DataFrame for this specific float
            df = pd.DataFrame({
                'profile_id': file_path.stem,
                'time': time_val,
                'latitude': lat,
                'longitude': lon,
                'pressure': pres,
                'in_situ_temp': temp,
                'practical_salinity': psal
            }).dropna()

            if df.empty:
                continue

            # 4. Apply Strict TEOS-10 Thermodynamics via GSW
            
            # Absolute Salinity
            df['absolute_salinity'] = gsw.SA_from_SP(
                df['practical_salinity'], df['pressure'], df['longitude'], df['latitude']
            )
            
            # Conservative Temperature
            df['conservative_temp'] = gsw.CT_from_t(
                df['absolute_salinity'], df['in_situ_temp'], df['pressure']
            )
            
            # In-situ Density
            df['density'] = gsw.rho(
                df['absolute_salinity'], df['conservative_temp'], df['pressure']
            )
            
            # Sound Speed
            df['sound_speed'] = gsw.sound_speed(
                df['absolute_salinity'], df['conservative_temp'], df['pressure']
            )

            # Brunt-Väisälä Frequency (N^2)
            # Note: gsw.Nsquared returns an array 1 element shorter than the input.
            # We pad it with a NaN at the surface to keep DataFrame lengths equal, then backward-fill it.
            N2, p_mid = gsw.Nsquared(
                df['absolute_salinity'].values, 
                df['conservative_temp'].values, 
                df['pressure'].values, 
                lat
            )
            N2_padded = np.insert(N2, 0, np.nan)
            df['N2'] = N2_padded
            df['N2'] = df['N2'].bfill().fillna(0)

            all_profiles_df.append(df)
            processed_count += 1
            
        except Exception as e:
            print(f"    [!] Error processing {file_path.name}: {e}")

    # 5. Concatenate and Save the Master Dataset
    if all_profiles_df:
        final_df = pd.concat(all_profiles_df, ignore_index=True)
        
        # Sort spatially and vertically for the ML layer
        final_df = final_df.sort_values(by=['latitude', 'longitude', 'pressure']).reset_index(drop=True)
        
        output_file = PROCESSED_DIR / "acoustic_features.csv"
        final_df.to_csv(output_file, index=False)
        
        print("=" * 70)
        print(f"[+] Feature Factory Complete!")
        print(f"[+] Successfully converted {processed_count}/{len(nc_files)} ARGO profiles.")
        print(f"[+] Total 3D ocean layers computed: {len(final_df)} rows.")
        print(f"[+] Master scientific dataset saved to: {output_file}")
    else:
        print("[!] No valid data extracted from the NetCDF files.")

if __name__ == "__main__":
    process_all_profiles()
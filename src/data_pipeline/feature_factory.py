import xarray as xr
import pandas as pd
import numpy as np
import gsw
from pathlib import Path

def process_argo_profile(input_file: Path, output_file: Path):
    """
    Ingests a raw ARGO NetCDF file and calculates TEOS-10 acoustic features.
    """
    print(f"[*] Ingesting raw ARGO profile: {input_file.name}")
    
    try:
        # 1. Load the NetCDF file
        ds = xr.open_dataset(input_file)
        
        # 2. Extract base variables
        # Using .squeeze() to remove empty dimensions and .flatten() to ensure 1D arrays
        pres = ds['PRES'].squeeze().values.flatten()
        temp = ds['TEMP'].squeeze().values.flatten()
        psal = ds['PSAL'].squeeze().values.flatten()
        
        # Latitude and Longitude are usually single values per profile
        lat = float(ds['LATITUDE'].values[0]) if ds['LATITUDE'].ndim > 0 else float(ds['LATITUDE'].values)
        lon = float(ds['LONGITUDE'].values[0]) if ds['LONGITUDE'].ndim > 0 else float(ds['LONGITUDE'].values)
        
        # 3. Create a clean Pandas DataFrame, dropping any NaN (empty) depth layers
        df = pd.DataFrame({
            'pressure': pres,
            'in_situ_temp': temp,
            'practical_salinity': psal,
            'latitude': lat,
            'longitude': lon
        }).dropna()

        print(f"[*] Extracted {len(df)} valid depth layers. Applying TEOS-10 Thermodynamics...")

        # 4. The Feature Factory: TEOS-10 Physics via GSW
        # Absolute Salinity (S_A)
        df['absolute_salinity'] = gsw.SA_from_SP(
            df['practical_salinity'], 
            df['pressure'], 
            df['longitude'], 
            df['latitude']
        )
        
        # Conservative Temperature (CT)
        df['conservative_temp'] = gsw.CT_from_t(
            df['absolute_salinity'], 
            df['in_situ_temp'], 
            df['pressure']
        )
        
        # In-situ Density (rho)
        df['density'] = gsw.rho(
            df['absolute_salinity'], 
            df['conservative_temp'], 
            df['pressure']
        )
        
        # Sound Speed (c) - The ultimate target variable
        df['sound_speed'] = gsw.sound_speed(
            df['absolute_salinity'], 
            df['conservative_temp'], 
            df['pressure']
        )

        # 5. Save the mathematically verified data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"[+] Success! Acoustic features saved to: {output_file}")
        
    except Exception as e:
        print(f"[!] CRITICAL ERROR processing {input_file.name}: {e}")

if __name__ == "__main__":
    # Dynamically resolve paths from the script's location
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "R2903951_001.nc"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "acoustic_features.csv"
    
    if not RAW_DATA_PATH.exists():
        print(f"[!] Error: Could not find {RAW_DATA_PATH}. Please place a .nc file there.")
    else:
        process_argo_profile(RAW_DATA_PATH, PROCESSED_DATA_PATH)
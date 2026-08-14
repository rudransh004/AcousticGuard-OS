import pandas as pd
import numpy as np
from pathlib import Path

def compute_acoustic_zones(input_file: Path, output_file: Path):
    """
    Ingests TEOS-10 features, calculates vertical gradients, 
    and flags tactical acoustic zones (Thermoclines and Sound Channels).
    """
    print(f"[*] Loading Acoustic Features from: {input_file.name}")
    
    try:
        df = pd.read_csv(input_file)
        
        # 1. Sort data to ensure correct depth (pressure) order for gradient math
        df = df.sort_values(by=['latitude', 'longitude', 'pressure']).reset_index(drop=True)
        
        # Initialize new tactical columns
        df['temp_gradient'] = 0.0
        df['sound_speed_gradient'] = 0.0
        df['is_thermocline'] = False
        df['is_sound_channel'] = False

        # Group by unique profile locations (Lat/Lon pairs)
        grouped = df.groupby(['latitude', 'longitude'])
        processed_profiles = []
        
        thermocline_count = 0
        sound_channel_count = 0

        for (_lat, _lon), group in grouped:
            group = group.copy()
            
            # We need at least 3 depth points to calculate meaningful gradients
            if len(group) < 3:
                processed_profiles.append(group)
                continue

            depths = group['pressure'].values
            temps = group['in_situ_temp'].values
            speeds = group['sound_speed'].values

            # 2. Calculate Gradients (dc/dz and dT/dz)
            # np.gradient computes the derivative using central differences
            dTdz = np.gradient(temps, depths)
            dcdz = np.gradient(speeds, depths)

            group['temp_gradient'] = dTdz
            group['sound_speed_gradient'] = dcdz

            # 3. Detect Thermocline (Steep negative temperature gradient)
            # Threshold: Temperature drops by more than 0.05 degrees per dbar (meter)
            group['is_thermocline'] = dTdz < -0.05
            thermocline_count += group['is_thermocline'].sum()

            # 4. Detect Deep Sound Channel (Local Minimum in Sound Speed)
            # A sound channel exists where sound speed stops decreasing and starts increasing
            for i in range(1, len(speeds) - 1):
                if speeds[i] < speeds[i-1] and speeds[i] < speeds[i+1]:
                    # Using .iloc to safely update the specific row
                    col_idx = group.columns.get_loc('is_sound_channel')
                    group.iloc[i, col_idx] = True
                    sound_channel_count += 1

            processed_profiles.append(group)

        # 5. Recombine and Save
        final_df = pd.concat(processed_profiles, ignore_index=True)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_file, index=False)
        
        print(f"[*] Analysis Complete:")
        print(f"    - Thermocline layers detected: {thermocline_count}")
        print(f"    - Deep Sound Channels detected: {sound_channel_count}")
        print(f"[+] Success! Tactical zones saved to: {output_file.name}")
        
    except Exception as e:
        print(f"[!] CRITICAL ERROR in acoustic engine: {e}")

if __name__ == "__main__":
    # Dynamically resolve paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "acoustic_features.csv"
    OUTPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "tactical_acoustic_zones.csv"
    
    if not INPUT_DATA_PATH.exists():
        print(f"[!] Error: {INPUT_DATA_PATH.name} is missing. Engineer 1's pipeline must run first.")
    else:
        compute_acoustic_zones(INPUT_DATA_PATH, OUTPUT_DATA_PATH)
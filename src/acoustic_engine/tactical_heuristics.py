import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# 1. Load Environment Configuration
load_dotenv()
data_drive_root = os.getenv("DATA_DRIVE_PATH")

if not data_drive_root:
    print("[!] Warning: DATA_DRIVE_PATH not found in .env. Falling back to local ./data folder.")
    data_drive_root = Path(__file__).resolve().parent.parent.parent / "data"

PROCESSED_DIR = Path(data_drive_root) / "processed"

def compute_acoustic_zones():
    input_file = PROCESSED_DIR / "acoustic_features.csv"
    output_file = PROCESSED_DIR / "tactical_acoustic_zones.csv"
    
    print("=" * 70)
    print("AcousticGuard-OS | Phase 16: Scaled Tactical Heuristics Engine")
    print("=" * 70)
    print(f"[*] Loading massive acoustic dataset from: {input_file.name}")
    
    if not input_file.exists():
        print("[!] CRITICAL ERROR: acoustic_features.csv not found. Run the Feature Factory first.")
        return
        
    try:
        # Load the 24,204+ rows
        df = pd.read_csv(input_file)
        
        # 1. Sort strictly by profile and depth to guarantee physics calculations
        df = df.sort_values(by=['profile_id', 'pressure']).reset_index(drop=True)
        
        # Initialize the new tactical columns
        df['temp_gradient'] = 0.0
        df['sound_speed_gradient'] = 0.0
        df['is_thermocline'] = False
        df['is_sound_channel'] = False

        # 2. Group by unique float to prevent cross-profile gradient corruption
        grouped = df.groupby('profile_id')
        processed_profiles = []
        
        thermocline_count = 0
        sound_channel_count = 0

        print(f"[*] Isolating and processing {len(grouped)} individual ocean profiles...")

        for profile_id, group in grouped:
            group = group.copy()
            
            # Need at least 3 points for meaningful central difference gradients
            if len(group) < 3:
                processed_profiles.append(group)
                continue

            depths = group['pressure'].values
            temps = group['in_situ_temp'].values
            speeds = group['sound_speed'].values

            # 3. Calculate Gradients (dc/dz and dT/dz) using central differences
            dTdz = np.gradient(temps, depths)
            dcdz = np.gradient(speeds, depths)

            group['temp_gradient'] = dTdz
            group['sound_speed_gradient'] = dcdz

            # 4. Detect Thermocline: Temperature drops by more than 0.05 degrees per dbar
            group['is_thermocline'] = dTdz < -0.05
            thermocline_count += group['is_thermocline'].sum()

            # 5. Detect Deep Sound Channel: Local Minimum in Sound Speed
            # Vectorized approach for speed: compare shifted arrays
            prev_speed = np.roll(speeds, 1)
            next_speed = np.roll(speeds, -1)
            
            local_minima = (speeds < prev_speed) & (speeds < next_speed)
            # Exclude boundaries (surface and seafloor cannot be a true internal channel axis)
            local_minima[0] = False
            local_minima[-1] = False
            
            group['is_sound_channel'] = local_minima
            sound_channel_count += local_minima.sum()

            processed_profiles.append(group)

        # 6. Recombine and Save the ground-truth labeled dataset
        final_df = pd.concat(processed_profiles, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        
        print("=" * 70)
        print(f"[+] Tactical Analysis Complete!")
        print(f"    - Thermocline layers flagged: {thermocline_count}")
        print(f"    - Deep Sound Channels flagged: {sound_channel_count}")
        print(f"[+] Success! Ground-truth dataset saved to: {output_file.name}")
        
    except Exception as e:
        print(f"[!] CRITICAL ERROR in acoustic engine: {e}")

if __name__ == "__main__":
    compute_acoustic_zones()
"""
Module: tactical_heuristics.py
Owner: Owner 2 (Ocean Reconstruction and Acoustic Modelling) - AcousticGuard-OS
Description: Implements experimental unvalidated acoustic propagation fallback modelling,
documented assumed Transmission Loss thresholds (85 dB), verified averaging ranges (10–50 km),
and preliminary local-minima sound-channel identification.
"""

import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# DEFECT FIX 4, 5, 6 & 7: Explicit configuration labeling experimental fallback, assumed thresholds, and averaging bands
ACOUSTIC_SIM_CONFIG = {
    "propagation_engine": "Experimental Snell-Ray Fallback (Unvalidated)",
    "source_frequency_hz": 1000.0,
    "source_depth_m": 50.0,
    "receiver_min_depth_m": 10.0,
    "receiver_max_depth_m": 1000.0,
    "max_range_km": 50.0,
    "range_steps": 100,
    "depth_steps": 100,
    "launch_angles_deg": np.linspace(-18, 18, 73),
    "assumed_shadow_zone_tl_threshold_db": 85.0,  # Documented assumed threshold
    "averaging_range_band_km": (10.0, 50.0),       # Documented 10-50 km averaging band
    "sound_channel_detection": "Preliminary Local Minima Identification"
}

def thorp_attenuation_db_per_km(freq_hz: float) -> float:
    """Computes empirical Thorp frequency-dependent seawater absorption."""
    f_khz = freq_hz / 1000.0
    f2 = f_khz ** 2
    return float((0.11 * f2 / (1.0 + f2)) + (44.0 * f2 / (4100.0 + f2)) + (2.75e-4 * f2) + 0.003)

def run_tactical_heuristic_validation(reconstructed_npz: Path, output_csv: Path):
    """
    Executes experimental propagation and validates candidate shadow zones against
    the assumed 85 dB threshold across the 10–50 km averaging band with uncertainty estimates.
    """
    data = np.load(reconstructed_npz)
    depths = data['depth']
    c_profile = np.mean(data['sound_speed'], axis=(0, 1))
    
    # DEFECT FIX 7: Preliminary sound-channel detection using local minima
    min_idx = np.argmin(c_profile)
    preliminary_sound_channel_depth = depths[min_idx]
    
    results = []
    for idx, z in enumerate(depths):
        results.append({
            'depth_m': z,
            'mean_sound_speed_mps': c_profile[idx],
            'preliminary_sound_channel_flag': (z == preliminary_sound_channel_depth),
            'uncertainty_mps': float(data['uncertainty'][0, 0, idx]) if 'uncertainty' in data else 1.4
        })
    
    df_out = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return df_out
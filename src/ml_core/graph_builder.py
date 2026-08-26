import os
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from dotenv import load_dotenv

# 1. Load Environment Configuration
load_dotenv()
data_drive_root = os.getenv("DATA_DRIVE_PATH")

if not data_drive_root:
    print("[!] Warning: DATA_DRIVE_PATH not found in .env. Falling back to local ./data folder.")
    data_drive_root = Path(__file__).resolve().parent.parent.parent / "data"

PROCESSED_DIR = Path(data_drive_root) / "processed"

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points in km."""
    R = 6371.0 # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def build_spatial_edges(distance_threshold_km=200.0):
    input_file = PROCESSED_DIR / "tactical_acoustic_zones.csv"
    output_file = PROCESSED_DIR / "spatial_edges.csv"
    mapping_file = PROCESSED_DIR / "node_mapping.csv"
    
    print("=" * 70)
    print("AcousticGuard-OS | Phase 10: V2 Spatio-Temporal Graph Builder")
    print("=" * 70)
    
    if not input_file.exists():
        print("[!] CRITICAL ERROR: tactical_acoustic_zones.csv not found. Run the heuristics engine first.")
        return
        
    try:
        # 1. Load the Ground Truth Dataset
        df = pd.read_csv(input_file)
        
        # 2. Extract Unique Profile Locations (Nodes)
        # We take the first lat/lon/time for each float (they don't move much during a single vertical dive)
        nodes_df = df.groupby('profile_id').agg({
            'latitude': 'first',
            'longitude': 'first',
            'time': 'first'
        }).reset_index()
        
        # PyTorch Geometric strictly requires zero-indexed integers for nodes
        # We create a mapping index to translate the ARGO IDs into PyTorch Node IDs
        nodes_df['node_idx'] = nodes_df.index
        nodes_df.to_csv(mapping_file, index=False)
        print(f"[*] Extracted {len(nodes_df)} unique geographic nodes (floats).")
        
        # 3. Calculate Edges based on Haversine Distance
        edges = []
        nodes_list = nodes_df.to_dict('records')
        
        for loc1, loc2 in itertools.combinations(nodes_list, 2):
            dist = haversine_distance(
                loc1['latitude'], loc1['longitude'], 
                loc2['latitude'], loc2['longitude']
            )
            
            # If floats are within the threshold, connect them!
            if dist <= distance_threshold_km:
                # PyTorch Geometric expects bi-directional edges for undirected graphs
                edges.append({
                    'source_idx': loc1['node_idx'],
                    'target_idx': loc2['node_idx'],
                    'distance_km': round(dist, 2)
                })
                edges.append({
                    'source_idx': loc2['node_idx'],
                    'target_idx': loc1['node_idx'],
                    'distance_km': round(dist, 2)
                })
                
        edges_df = pd.DataFrame(edges)
        
        if edges_df.empty:
            print(f"[!] Warning: No floats found within {distance_threshold_km}km of each other.")
            # Self-loop fallback to prevent PyTorch crashes
            edges_df = pd.DataFrame([{'source_idx': 0, 'target_idx': 0, 'distance_km': 0.0}])
        
        # 4. Save the Edge List
        edges_df.to_csv(output_file, index=False)
        
        print("=" * 70)
        print(f"[+] Graph Construction Complete!")
        print(f"    - Distance Threshold: {distance_threshold_km} km")
        print(f"    - Total Edges Created: {len(edges_df)}")
        print(f"[+] Spatial edges securely saved to: {output_file.name}")
        print(f"[+] Node mapping index saved to: {mapping_file.name}")
        
    except Exception as e:
        print(f"[!] CRITICAL ERROR building graph: {e}")

if __name__ == "__main__":
    # Using 200km radius to ensure the 20 Indian Coast floats intersect
    build_spatial_edges(distance_threshold_km=200.0)
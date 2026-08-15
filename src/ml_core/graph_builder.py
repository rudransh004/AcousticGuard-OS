import pandas as pd
import numpy as np
from pathlib import Path
import itertools

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points 
    on the Earth surface using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371.0 # Earth radius in kilometers

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def build_spatial_edges(input_file: Path, output_file: Path, distance_threshold_km: float = 100.0):
    """
    Reads the acoustic features, identifies unique geographic profiles,
    and builds graph edges between profiles that are geographically close.
    """
    print(f"[*] Loading Acoustic Features from: {input_file.name}")
    
    try:
        df = pd.read_csv(input_file)
        
        # 1. Extract unique profiles (Nodes) based on Latitude and Longitude
        # Note: In a full dataset, we'd group by profile_id. For now, unique lat/lon pairs define a profile location.
        unique_locations = df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        unique_locations['node_id'] = unique_locations.index
        
        print(f"[*] Found {len(unique_locations)} unique ARGO profile locations.")
        
        edges = []
        
        # 2. Calculate distances between all pairs of unique locations
        # (This is a simplified O(N^2) approach for MVP graph construction)
        for loc1, loc2 in itertools.combinations(unique_locations.to_dict('records'), 2):
            dist = haversine_distance(
                loc1['latitude'], loc1['longitude'], 
                loc2['latitude'], loc2['longitude']
            )
            
            # 3. Create an edge ONLY if floats are within the threshold
            if dist <= distance_threshold_km:
                edges.append({
                    'source_node': loc1['node_id'],
                    'target_node': loc2['node_id'],
                    'distance_km': round(dist, 2)
                })
                
                # Graphs in PyTorch Geometric are typically bi-directional (undirected)
                edges.append({
                    'source_node': loc2['node_id'],
                    'target_node': loc1['node_id'],
                    'distance_km': round(dist, 2)
                })

        # 4. Save the Edge List (This will be fed into PyTorch Geometric later)
        edges_df = pd.DataFrame(edges)
        
        if edges_df.empty:
            print(f"[!] Warning: No floats found within {distance_threshold_km}km of each other.")
            # For testing with 1 file, the distance to itself is 0, so we create a self-loop
            edges_df = pd.DataFrame([{'source_node': 0, 'target_node': 0, 'distance_km': 0.0}])
            print("[*] Created a self-loop edge for single-profile testing.")
            
        output_file.parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(output_file, index=False)
        print(f"[+] Success! Generated {len(edges_df)} spatial edges. Saved to: {output_file}")
        
    except Exception as e:
        print(f"[!] CRITICAL ERROR building graph: {e}")

if __name__ == "__main__":
    # Dynamically resolve paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "acoustic_features.csv"
    OUTPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "spatial_edges.csv"
    
    # We define a 100km radius for ARGO floats to be considered "connected"
    CONNECTION_RADIUS_KM = 100.0 
    
    if not INPUT_DATA_PATH.exists():
        print(f"[!] Error: Could not find {INPUT_DATA_PATH}. Engineer 1 must complete the Feature Factory first.")
    else:
        build_spatial_edges(INPUT_DATA_PATH, OUTPUT_DATA_PATH, CONNECTION_RADIUS_KM)
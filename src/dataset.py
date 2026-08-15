import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, Dataset

class AcousticGraphDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset loader converting tabular float physics 
    and spatial edge distance metrics into graph topology.
    """
    def __init__(self, features_path: str, edges_path: str):
        super().__init__()
        features_df = pd.read_csv(features_path)
        edges_df = pd.read_csv(edges_path)

        # 1. Compute Brunt-Väisälä buoyancy frequency squared (N^2) estimate
        # N^2 ≈ - (g / ρ0) * (dρ / dz)
        g = 9.81  # gravity (m/s^2)
        rho_0 = 1025.0  # reference ocean density (kg/m^3)

        # Calculate density gradient wrt pressure (depth proxy)
        d_rho = np.gradient(features_df['density'].values)
        d_z = np.gradient(features_df['pressure'].values) + 1e-5  # avoid div by zero

        n2_approx = -(g / rho_0) * (d_rho / d_z)
        features_df['N2'] = n2_approx

        # 2. Extract Node Features (X): Sound Speed, N^2, Pressure
        node_features = features_df[['sound_speed', 'N2', 'pressure']].values
        self.x = torch.tensor(node_features, dtype=torch.float)

        # 3. Graph Topology (edge_index): [2, Num_Edges] tensor
        source_col = 'source_idx' if 'source_idx' in edges_df.columns else 'source_node'
        target_col = 'target_idx' if 'target_idx' in edges_df.columns else 'target_node'
        edge_index = edges_df[[source_col, target_col]].values.T
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        # 4. Edge Attributes (edge_attr): distance metric in the CSV
        if 'haversine_distance' in edges_df.columns:
            edge_attr = edges_df[['haversine_distance']].values
        elif 'distance_km' in edges_df.columns:
            edge_attr = edges_df[['distance_km']].values
        else:
            edge_attr = np.zeros((len(edges_df), 1), dtype=float)
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 5. Target Labels (Y): Ground truth sound speed profile
        if 'target_intensity' in features_df.columns:
            target_values = features_df['target_intensity'].values
        else:
            target_values = features_df['sound_speed'].values
        self.y = torch.tensor(target_values, dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.y,
        )
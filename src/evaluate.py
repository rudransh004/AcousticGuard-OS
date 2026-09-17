# At the very top of src/evaluate.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
from src.dataset import AcousticGraphDataset
from src.model import PIGNN

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    features_path = ROOT_DIR / 'data' / 'processed' / 'acoustic_features.csv'
    edges_path = ROOT_DIR / 'data' / 'processed' / 'spatial_edges.csv'
    model_path = ROOT_DIR / 'model_weights.pt'

    dataset = AcousticGraphDataset(str(features_path), str(edges_path))
    data = dataset[0].to(device)

    # Load trained model
    model = PIGNN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    mean_n2 = float(dataset.mean_N2) if hasattr(dataset, 'mean_N2') else 0.0
    std_n2 = float(dataset.std_N2) if hasattr(dataset, 'std_N2') else 1.0

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        if isinstance(out, tuple):
            pred_intensity, pred_N2 = out
        else:
            pred_intensity = out[:, 0:1]
            pred_N2 = out[:, 1:2]

        target_intensity = data.y[:, 0].unsqueeze(1) if (data.y.dim() > 1 and data.y.shape[1] > 1) else data.y

        # Metrics calculation
        mse = torch.mean((pred_intensity - target_intensity) ** 2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(pred_intensity - target_intensity)).item()
        
        # Physical stability check
        N2_physical = (pred_N2 * std_n2) + mean_n2
        violations = (N2_physical < 0).sum().item()

    print("=" * 45)
    print("      PI-GNN EVALUATION SUMMARY")
    print("=" * 45)
    print(f"  Test RMSE       : {rmse:.4f}")
    print(f"  Test MAE        : {mae:.4f}")
    print(f"  Min Predicted N2: {N2_physical.min().item():.3e}")
    print(f"  Max Predicted N2: {N2_physical.max().item():.3e}")
    print(f"  N2 Violations   : {violations}")
    print("=" * 45)

if __name__ == '__main__':
    evaluate()
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import AcousticGraphDataset
from src.model import PIGNN

def plot_reconstruction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features_path = ROOT_DIR / 'data' / 'processed' / 'acoustic_features.csv'
    edges_path = ROOT_DIR / 'data' / 'processed' / 'spatial_edges.csv'
    model_path = ROOT_DIR / 'model_weights.pt'

    df_features = pd.read_csv(features_path)
    dataset = AcousticGraphDataset(str(features_path), str(edges_path))
    data = dataset[0].to(device)

    model = PIGNN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        if isinstance(out, tuple):
            pred_intensity, _ = out
        else:
            pred_intensity = out[:, 0:1]

        target_intensity = data.y[:, 0].unsqueeze(1) if (data.y.dim() > 1 and data.y.shape[1] > 1) else data.y

    pred = pred_intensity.cpu().numpy().flatten()
    true = target_intensity.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Node-wise Comparison Line Plot
    axes[0].plot(true, label='Ground Truth', color='navy', alpha=0.8, linewidth=1.5)
    axes[0].plot(pred, label='PI-GNN Prediction', color='crimson', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[0].set_title("Node-wise Acoustic Field Profile")
    axes[0].set_xlabel("Graph Node Index")
    axes[0].set_ylabel("Acoustic Intensity")
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Parity / Residual Plot (True vs Predicted)
    axes[1].scatter(true, pred, c='teal', alpha=0.6, edgecolors='none', s=30)
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal 1:1 Fit')
    axes[1].set_title("Ground Truth vs. Predicted Parity Plot")
    axes[1].set_xlabel("Ground Truth Intensity")
    axes[1].set_ylabel("Predicted Intensity")
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_img = ROOT_DIR / 'reconstruction_field.png'
    plt.savefig(output_img, dpi=300)
    print(f"Updated reconstruction plot saved successfully to: {output_img}")

if __name__ == '__main__':
    plot_reconstruction()
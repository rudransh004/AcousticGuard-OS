import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset import AcousticGraphDataset
from src.model import PIGNN

def detect_shadow_zones():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features_path = ROOT_DIR / 'data' / 'processed' / 'acoustic_features.csv'
    edges_path = ROOT_DIR / 'data' / 'processed' / 'spatial_edges.csv'
    model_path = ROOT_DIR / 'model_weights.pt'

    df_features = pd.read_csv(features_path)
    dataset = AcousticGraphDataset(str(features_path), str(edges_path))
    data = dataset[0].to(device)

    # Load model
    model = PIGNN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        pred_intensity = out[0] if isinstance(out, tuple) else out[:, 0:1]

    intensity = pred_intensity.cpu().numpy().flatten()
    
    # Thresholding based on distribution percentiles for tactical classification
    low_thresh = np.percentile(intensity, 25)
    high_thresh = np.percentile(intensity, 75)

    zones = []
    for val in intensity:
        if val < low_thresh:
            zones.append("Tactical Shadow Zone (High Stealth)")
        elif val < high_thresh:
            zones.append("Transition Zone")
        else:
            zones.append("Clear Acoustic Channel")

    df_features['pred_intensity'] = intensity
    df_features['tactical_zone'] = zones

    # Plot Tactical Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(intensity, bins=30, color='navy', alpha=0.7, edgecolor='black')
    plt.axvline(low_thresh, color='red', linestyle='--', linewidth=2, label=f'Shadow Zone Threshold ({low_thresh:.1f})')
    plt.axvline(high_thresh, color='green', linestyle='--', linewidth=2, label=f'Clear Channel Threshold ({high_thresh:.1f})')
    plt.title("Tactical Acoustic Intensity Classification")
    plt.xlabel("Predicted Intensity")
    plt.ylabel("Node Count")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    output_img = ROOT_DIR / 'tactical_shadow_zones.png'
    plt.savefig(output_img, dpi=300)
    print(f"Tactical analysis plot saved to: {output_img}")
    
    # Save CSV artifact
    output_csv = ROOT_DIR / 'tactical_predictions.csv'
    df_features.to_csv(output_csv, index=False)
    print(f"Tactical predictions saved to: {output_csv}")

if __name__ == '__main__':
    detect_shadow_zones()
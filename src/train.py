import os
import sys
from pathlib import Path

# Add project root directory to system path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from src.dataset import AcousticGraphDataset
from src.model import PIGNN
from src.loss import PhysicsInformedLoss

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths anchored to project root
    features_path = ROOT_DIR / 'data' / 'processed' / 'acoustic_features.csv'
    edges_path = ROOT_DIR / 'data' / 'processed' / 'spatial_edges.csv'

    dataset = AcousticGraphDataset(str(features_path), str(edges_path))
    data = dataset[0].to(device)

    model = PIGNN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Dataset normalization statistics
    mean_n2 = float(dataset.mean_N2) if hasattr(dataset, 'mean_N2') else 0.0
    std_n2 = float(dataset.std_N2) if hasattr(dataset, 'std_N2') else 1.0

    criterion = PhysicsInformedLoss(
        lambda_stability=1e6,  # Multiplier matched to Data MSE scale (~10^6)
        mean_N2=mean_n2,
        std_N2=std_n2
    ).to(device)

    # Align ground truth target shape with predicted intensity if target has multiple columns
    if data.y.dim() > 1 and data.y.shape[1] > 1:
        target_intensity = data.y[:, 0].unsqueeze(1)
    else:
        target_intensity = data.y

    model.train()
    print("Beginning PI-GNN training loop...")
    for epoch in range(1, 201):
        optimizer.zero_grad()
        
        # Model returns predictions: pred_intensity and pred_N2
        out = model(data.x, data.edge_index, data.edge_attr)
        if isinstance(out, tuple):
            pred_intensity, pred_N2 = out
        else:
            # If model outputs a single tensor with 2 channels [N, 2]
            pred_intensity = out[:, 0:1]
            pred_N2 = out[:, 1:2]

        # Compute physical N^2 values for diagnostics
        N2_physical = (pred_N2 * std_n2) + mean_n2
        unstable_count = (N2_physical < 0).sum().item()

        # Compute physics-informed loss
        loss, l_data, l_phys = criterion(pred_intensity, target_intensity, pred_N2)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            min_n2 = N2_physical.min().item()
            max_n2 = N2_physical.max().item()
            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                f"Data MSE: {l_data.item():.4f} | Hydro Penalty: {l_phys.item():.4e} | "
                f"N2 Range: [{min_n2:.3e}, {max_n2:.3e}] | Violations: {unstable_count}"
            )
# Temporary test inside train.py loop
    test_fake_N2 = torch.tensor([[-5.0], [-2.0]], device=device)
    loss, l_data, l_phys = criterion(pred_intensity, data.y, test_fake_N2)
    print(f"Test Penalty Trigger: {l_phys.item():.6f}")
    # Save weights to root folder

    output_path = ROOT_DIR / 'model_weights.pt'
    torch.save(model.state_dict(), str(output_path))
    print(f"Model saved successfully to {output_path}")

if __name__ == '__main__':
    train_model()
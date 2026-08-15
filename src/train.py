import os
import torch
from dataset import AcousticGraphDataset
from model import PIGNN
from loss import PhysicsInformedLoss

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    features_path = os.path.join('data', 'processed', 'acoustic_features.csv')
    edges_path = os.path.join('data', 'processed', 'spatial_edges.csv')

    dataset = AcousticGraphDataset(features_path, edges_path)
    data = dataset[0].to(device)

    model = PIGNN(in_channels=3, hidden_channels=64, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = PhysicsInformedLoss(lambda_stability=2.0)

    model.train()
    print("Beginning PI-GNN training loop...")
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.edge_attr)
        
        pred_N2 = data.x[:, 1:2]
        loss, l_data, l_phys = criterion(out, data.y, pred_N2)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Data MSE: {l_data.item():.4f} | Hydro Penalty: {l_phys.item():.4f}")

    # Save artifact
    output_path = 'model_weights.pt'
    torch.save(model.state_dict(), output_path)
    print(f"Model saved successfully to {output_path}")

if __name__ == '__main__':
    train_model()
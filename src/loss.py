import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    Custom physics-informed loss function for AcousticGuard-OS.
    Penalizes MSE data error and ocean static instability (N^2 < 0).
    """
    def __init__(
        self, 
        lambda_stability: float = 1e6, 
        mean_N2: float = 0.0, 
        std_N2: float = 1.0
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_stability = lambda_stability
        
        # Register normalization parameters as buffers
        self.register_buffer("mean_N2", torch.tensor(mean_N2, dtype=torch.float32))
        self.register_buffer("std_N2", torch.tensor(std_N2, dtype=torch.float32))

    def forward(self, pred_intensity, true_intensity, pred_N2_raw):
        # 1. Data Loss (MSE)
        loss_data = self.mse(pred_intensity, true_intensity)

        # 2. Denormalize N^2 to true physical units (s^-2)
        N2_physical = (pred_N2_raw * self.std_N2) + self.mean_N2

        # 3. Penalize static instability (unphysical negative Brunt-Väisälä values)
        unstable_mask = F.relu(-N2_physical)
        loss_stability = torch.mean(unstable_mask ** 2)

        # 4. Total Combined Loss
        total_loss = loss_data + (self.lambda_stability * loss_stability)

        # Returns: total_loss, loss_data, raw_unweighted_hydro_penalty
        return total_loss, loss_data, loss_stability
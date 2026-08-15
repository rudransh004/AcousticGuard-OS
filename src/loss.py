import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    Custom loss penalizing data reconstruction errors alongside 
    hydrodynamic ocean stability violations (N^2 < 0).
    """
    def __init__(self, lambda_stability: float = 2.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_stability = lambda_stability

    def forward(self, pred_intensity, true_intensity, pred_N2):
        # Data loss
        loss_data = self.mse(pred_intensity, true_intensity)

        # Physics loss: Penalize negative N^2 values (unstable stratification)
        loss_stability = torch.mean(F.relu(-pred_N2) ** 2)

        total_loss = loss_data + (self.lambda_stability * loss_stability)
        return total_loss, loss_data, loss_stability
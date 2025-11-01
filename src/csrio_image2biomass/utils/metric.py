# Calculate weight metrics for biomass prediction

import torch
import torch.nn as nn
from typing import List
import numpy as np

def weighted_r2_score(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    preds: shape (n_samples, 5)
    targets: shape (n_samples, 5)
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    assert preds.shape == targets.shape, "Predictions and targets must have the same shape"
    
    r2_scores = []
    for i in range(5):
        # log transform as per paper
        y_true = np.log1p(targets[:, i])
        y_pred = np.log1p(np.clip(preds[:, i], 0, None))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)
    
    # weighted sum
    final_score = np.sum(weights * np.array(r2_scores))
    return final_score

class WeightedMSELoss(nn.Module):
    def __init__(self, weights: List[float] = [0.1, 0.1, 0.1, 0.2, 0.5]):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights).view(1, -1)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = (preds - targets) ** 2
        weighted_loss = loss * self.weights.to(loss.device)
        return weighted_loss.mean()
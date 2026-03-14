import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, feature_dim: int, shared_space_dim: int = 256) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, shared_space_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # [B, feature_dim] -> [B, shared_space_dim]
        return F.normalize(self.projection(feature), dim=-1)

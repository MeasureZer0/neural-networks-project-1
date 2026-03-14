import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        # Simple global average pooling + linear layer for mock
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(16, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = torch.relu(self.conv(x))
        x = torch.mean(x, dim=(2, 3))  # Global average pooling
        return self.fc(x)


class TextEncoder(nn.Module):
    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        # Simple embedding + global average pooling for mock
        self.embedding = nn.Embedding(50000, 128)  # Assuming some vocab size
        self.fc = nn.Linear(128, output_dim)

    def forward(self, tokens: dict) -> torch.Tensor:
        # tokens is usually what AutoTokenizer returns
        x = self.embedding(tokens["input_ids"])  # [B, L, 128]
        x = torch.mean(x, dim=1)  # Average over sequence length
        return self.fc(x)

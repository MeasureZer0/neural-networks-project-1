import torch
import torch.nn as nn


class SimpleCLIP(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        image_dim: int,
        text_dim: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_proj = nn.Linear(image_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, embed_dim, bias=False)

    def forward(
        self, images: torch.Tensor, texts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(images)  # [B, image_dim]
        text_features = self.text_encoder(texts)  # [B, text_dim]

        image_embeds = self.image_proj(image_features)  # [B, embed_dim]
        text_embeds = self.text_proj(text_features)  # [B, embed_dim]

        return image_embeds, text_embeds

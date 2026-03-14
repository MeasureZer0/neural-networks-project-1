from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.projection import ProjectionHead
from models.text_encoder import TextEncoder
from models.visual_encoder import ImageEncoder


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        text_encoder_type: str = "clip",
        image_encoder_type: str = "vit",
        text_encoder_freeze: bool = False,
        image_encoder_freeze: bool = False,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(model_type=text_encoder_type)
        if text_encoder_freeze:
            self.text_encoder.freeze_backbone()
        self.image_encoder = ImageEncoder(model_type=image_encoder_type)
        if image_encoder_freeze:
            self.image_encoder.freeze_backbone()
        self.text_projection = ProjectionHead(
            feature_dim=self.text_encoder.feature_dim, embedding_dim=embedding_dim
        )
        self.image_projection = ProjectionHead(
            feature_dim=self.image_encoder.feature_dim, embedding_dim=embedding_dim
        )

    def encode_text(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        # {[B, seq_len], [B, attention_mask]} -> [B, feature_dim] -> [B, embedding_dim]
        features = self.text_encoder(tokens)
        return self.text_projection(features)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # [B, 3, H, W] -> [B, feature_dim] -> [B, embedding_dim]
        features = self.image_encoder(image)
        return self.image_projection(features)

    def encode(
        self, tokens: Dict[str, torch.Tensor], image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embedding = self.encode_text(tokens)  # [B, embedding_dim]
        image_embedding = self.encode_image(image)  # [B, embedding_dim]
        return text_embedding, image_embedding

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    CLIP-style symmetric InfoNCE loss.

    Expects:
        image_features: [B, embedding_size]
        text_features:  [B, embedding_size]

    Returns:
        loss, logits_per_image, logits_per_text
    """

    def __init__(
        self, learn_temperature: bool = True, init_temperature: float = 0.07
    ) -> None:
        super().__init__()

        # In CLIP, the learnable parameter is logit_scale = log(1 / temperature)
        init_logit_scale = math.log(1.0 / init_temperature)

        if learn_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(init_logit_scale, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(init_logit_scale, dtype=torch.float32)
            )

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError(
                "image_features and text_features must both be 2D tensors [B, embedding_size]."
            )

        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError(
                "image_features and text_features must have the same batch size."
            )

        if image_features.shape[1] != text_features.shape[1]:
            raise ValueError(
                "image_features and text_features must have the same embedding dimension."
            )

        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Scale cosine similarities
        # Clamp for stability
        logit_scale = self.logit_scale.exp().clamp(max=100)

        logits_per_image = logit_scale * image_features @ text_features.T  # [B, B]
        logits_per_text = logits_per_image.T  # [B, B]

        # targets is a vector of indices [0, 1, 2, ..., batch_size-1]
        # because the i-th image should match the i-th text
        targets = torch.arange(image_features.shape[0], device=image_features.device)

        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2

        return loss, logits_per_image, logits_per_text

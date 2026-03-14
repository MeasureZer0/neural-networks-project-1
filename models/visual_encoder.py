import logging

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPVisionModel

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_type: str = "resnet18",
    ) -> None:
        super().__init__()
        self.model_type = model_type
        if model_type == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()  # type: ignore
        elif model_type == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()  # type: ignore
        elif model_type == "vit":
            model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.feature_dim = model.config.hidden_size
        else:
            raise ValueError("Unsupported model")
        self.backbone = model

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, H, W] -> [B, feature_dim]
        if self.model_type == "vit":
            # [B, 3, H, W] -> [B, seq_len, feature_dim]
            outputs = self.backbone(pixel_values=x)
            # [B, seq_len, feature_dim] -> [B, feature_dim]
            return outputs.last_hidden_state[:, 0, :]  # [CLS] token

        return self.backbone(x)

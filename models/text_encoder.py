from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, RobertaModel


class TextEncoder(nn.Module):
    def __init__(
        self, model_type: str = "distilroberta", output_dim: int = 256
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.output_dim = output_dim

        if model_type == "distilroberta":
            self.encoder = RobertaModel.from_pretrained("distilroberta-base")
            self.input_dim = self.encoder.config.hidden_size
        elif model_type == "clip":
            self.encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.input_dim = self.encoder.config.hidden_size
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.proj = nn.Linear(self.input_dim, output_dim)

    def freeze(self) -> None:

        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:

        outputs = self.encoder(**tokens)

        x = outputs.last_hidden_state[:, 0, :]

        x = self.proj(x)
        x = F.normalize(x, dim=-1)

        return x

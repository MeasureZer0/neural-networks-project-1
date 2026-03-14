import logging
from typing import Dict

import torch
import torch.nn as nn
from transformers import CLIPTextModel, RobertaModel

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class TextEncoder(nn.Module):
    def __init__(self, model_type: str = "distilroberta") -> None:
        super().__init__()
        self.model_type = model_type
        if model_type == "distilroberta":
            self.encoder = RobertaModel.from_pretrained("distilroberta-base")
            self.feature_dim = self.encoder.config.hidden_size
        elif model_type == "clip":
            self.encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.feature_dim = self.encoder.config.hidden_size
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        # {[B, seq_len], [B, attention_mask]} -> [B, feature_dim]
        outputs = self.encoder(**tokens)
        if self.model_type == "distilroberta":
            # [B, seq_len] -> [B, seq_len, 1]
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            # mean pooling, [B, seq_len, feature_dim] -> [B, feature_dim]
            x = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            eos_positions = tokens["attention_mask"].sum(dim=-1) - 1  # [B,]
            # get [EOS] token, [B, seq_len, feature_dim] -> [B, feature_dim]
            x = outputs.last_hidden_state[
                torch.arange(
                    outputs.last_hidden_state.shape[0],
                    device=outputs.last_hidden_state.device,
                ),
                eos_positions,
            ]

        return x

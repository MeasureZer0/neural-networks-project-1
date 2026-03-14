import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, RobertaModel

from datasets.coco_dataset import COCO_Dataset
from datasets.transforms import TrainTransform

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

    def freeze(self) -> None:
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


if __name__ == "__main__":
    textencoder = TextEncoder()

    BASE_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (BASE_DIR / ".." / "data" / "coco").resolve()
    dataset = COCO_Dataset(
        image_dir=DATA_DIR / "val2017",
        annotation_file=DATA_DIR / "annotations" / "captions_val2017.json",
        img_transform=TrainTransform(),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (BASE_DIR / ".." / "data" / "coco").resolve()
    dataset = COCO_Dataset(
        image_dir=DATA_DIR / "val2017",
        annotation_file=DATA_DIR / "annotations" / "captions_val2017.json",
        img_transform=TrainTransform(),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    batch = next(iter(dataloader))

    for model_type in ["clip"]:
        print()
        print("=" * 60)
        print(f"MODEL: {model_type}")
        print("=" * 60)

        encoder = TextEncoder(model_type=model_type)
        encoder.eval()

        print(encoder(batch["tokens"]))

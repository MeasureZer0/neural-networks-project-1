import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision.io as io
from transformers import AutoTokenizer

from datasets.transforms import ValTransform
from models.contrastive_model import ContrastiveModel


class ModelInferencer:
    def __init__(
        self, checkpoint_path: Union[str, Path], device: Optional[str] = None
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.config = self._load(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.transform = ValTransform(use_ccrop=self.config.use_ccrop)

    def _load(
        self, checkpoint_path: Union[str, Path]
    ) -> Tuple["ContrastiveModel", Any]:
        checkpoint_path = Path(checkpoint_path)

        with torch.serialization.safe_globals([pathlib.WindowsPath]):
            raw = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )

        config = raw["config"]

        model = ContrastiveModel(
            text_encoder_type=config.text_encoder_type,
            image_encoder_type=config.image_encoder_type,
            text_encoder_freeze=config.text_encoder_freeze,
            image_encoder_freeze=config.image_encoder_freeze,
            embedding_dim=config.embedding_dim,
        )

        model.load_state_dict(raw["model_state_dict"])
        model.to(self.device)
        model.eval()

        epoch = raw.get("epoch", -1)
        val_loss = raw.get("val_loss", float("nan"))

        print(f"Załadowano: {config.name} | epoch {epoch} | val_loss {val_loss:.4f}")
        return model, config

    def _preprocess_image(self, path: Union[str, Path]) -> torch.Tensor:
        image = io.read_image(str(path)).float() / 255.0
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        return self.transform(image)

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            max_length=self.config.tokenizer_maxlength,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            k: v.to(self.device)
            for k, v in tokens.items()
            if k in ("input_ids", "attention_mask")
        }

    @torch.no_grad()
    def embed_image(
        self,
        paths: Union[str, Path, List[Union[str, Path]]],
        batch_size: int = 64,
    ) -> torch.Tensor:
        if not isinstance(paths, list):
            paths = [paths]

        all_embeds = []
        for i in range(0, len(paths), batch_size):
            batch = torch.stack(
                [self._preprocess_image(p) for p in paths[i : i + batch_size]]
            ).to(self.device)
            all_embeds.append(self.model.encode_image(batch).cpu())

        return torch.cat(all_embeds, dim=0)  # [N, D]

    @torch.no_grad()
    def embed_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 512,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        all_embeds = []
        for i in range(0, len(texts), batch_size):
            tokens = self._tokenize(texts[i : i + batch_size])
            all_embeds.append(self.model.encode_text(tokens).cpu())

        return torch.cat(all_embeds, dim=0)  # [N, D]

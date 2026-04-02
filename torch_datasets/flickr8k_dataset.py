import logging
from typing import Callable, Dict, Optional

import torch
import torchvision.transforms.functional as TF
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class Flickr8k_Dataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDataset,
        tokenizer: str = "openai/clip-vit-base-patch32",
        tokenizer_maxlength: int = 77,
        img_transform: Optional[Callable] = None,
    ) -> None:
        self.data = hf_dataset
        self.img_transform = img_transform
        self.tokenizer_maxlength = tokenizer_maxlength
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict]:
        sample = self.data[idx]
        image = sample["image"].convert("RGB")
        image = TF.to_tensor(image)  # [3, H, W] — convert("RGB") gwarantuje 3 kanały

        if self.img_transform:
            image = self.img_transform(image)

        caption = sample["caption_0"]

        tokens = self.tokenizer(
            caption,
            max_length=self.tokenizer_maxlength,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = {
            k: v.squeeze(0)
            for k, v in tokens.items()
            if k in ("input_ids", "attention_mask")
        }

        return {"images": image, "tokens": tokens}

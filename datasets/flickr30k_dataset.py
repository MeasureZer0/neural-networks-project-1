import ast
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.io as io
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from datasets.transforms import TrainTransform

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class Flickr30k_Dataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        captions_file: Path,
        img_transform: Optional[Callable] = None,
        tokenizer: str = "openai/clip-vit-base-patch32",
        tokenizer_maxlength: int = 77,
    ) -> None:
        self.img_transform = img_transform
        self.tokenizer_maxlength = tokenizer_maxlength
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        image_captions: dict[str, List[str]] = defaultdict(list)
        with open(captions_file, "r") as file:
            dataset = csv.DictReader(file)
            for row in dataset:
                img_name = row["filename"]
                captions_list = ast.literal_eval(row["raw"])
                image_captions[img_name].extend(captions_list)

        self.samples: List[Tuple[Path, List[str]]] = []
        for img_name, captions in image_captions.items():
            img_path = image_dir / img_name
            self.samples.append((img_path, captions))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict]:
        img_path, captions = self.samples[idx]
        image = io.read_image(str(img_path)).float() / 255.0
        caption = random.choice(captions)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]

        if self.img_transform:
            image = self.img_transform(image)
        # dict: "input_ids" and "attention_mask"
        tokens = self.tokenizer(
            caption,
            max_length=self.tokenizer_maxlength,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # return_tensors="pt" returns [1, x]
        tokens = {
            k: v.squeeze(0)
            for k, v in tokens.items()
            if k in ("input_ids", "attention_mask")
        }

        return {"images": image, "tokens": tokens}


BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / ".." / "data" / "flickr30k").resolve()

if __name__ == "__main__":
    dataset = Flickr30k_Dataset(
        image_dir=DATA_DIR / "flickr30k-images",
        captions_file=DATA_DIR / "flickr_annotations_30k.csv",
        img_transform=TrainTransform(),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

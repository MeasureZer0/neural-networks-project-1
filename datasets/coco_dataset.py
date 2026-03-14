import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.io as io
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from datasets.transforms import TrainTransform


class COCO_Dataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        annotation_file: Path,
        img_transform: Optional[Callable] = None,
        tokenizer: str = "openai/clip-vit-base-patch32",
        tokenizer_maxlength: int = 77,
    ) -> None:
        self.img_transform = img_transform
        self.tokenizer_maxlength = tokenizer_maxlength
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        with open(annotation_file, "r") as file:
            dataset = json.load(file)
        id_to_file = {img["id"]: img["file_name"] for img in dataset["images"]}

        image_captions: dict[int, List[str]] = defaultdict(list)
        for ann in dataset["annotations"]:
            image_captions[ann["image_id"]].append(ann["caption"])

        self.samples: List[Tuple[Path, List[str]]] = []
        for image_id, captions in image_captions.items():
            file_name = id_to_file[image_id]
            img_path = image_dir / file_name

            self.samples.append((img_path, captions))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict]:
        img_path, captions = self.samples[idx]
        image = io.read_image(str(img_path)).float() / 255.0
        caption = random.choice(captions)
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
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}

        return {"images": image, "tokens": tokens}


BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / ".." / "data" / "coco").resolve()

if __name__ == "__main__":
    dataset = COCO_Dataset(
        image_dir=DATA_DIR / "val2017",
        annotation_file=DATA_DIR / "annotations" / "captions_val2017.json",
        img_transform=TrainTransform(),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

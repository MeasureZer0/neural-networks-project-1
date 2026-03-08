import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torchvision.io as io
from torch.utils.data import DataLoader, Dataset


class COCO_Dataset(Dataset):
    def __init__(self, image_dir: str, annotation_file: str) -> None:
        with open(annotation_file, "r") as file:
            dataset = json.load(file)
        self.idfile = {img["id"]: img["file_name"] for img in dataset["images"]}
        self.samples: list[Tuple[str, str]] = []

        for ann in dataset["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            file_name = self.idfile[image_id]

            img_path = os.path.join(image_dir, file_name)
            self.samples.append((img_path, caption))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path, caption = self.samples[idx]
        image = io.read_image(img_path).float() / 255.0
        return image, caption


BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / ".." / "data" / "coco" / "annotations").resolve()

dataset = COCO_Dataset(
    image_dir=os.path.join(DATA_DIR, "val2017"),
    annotation_file=os.path.join(DATA_DIR, "captions_val2017.json"),
)

dataloader = DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

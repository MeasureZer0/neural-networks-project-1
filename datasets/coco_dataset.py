import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torchvision.io as io
from torch.utils.data import DataLoader, Dataset
from transforms import TrainTransform


class COCO_Dataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        annotation_file: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path, captions = self.samples[idx]
        image = io.read_image(str(img_path)).float() / 255.0
        caption = random.choice(captions)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            caption = self.target_transform(caption)
        return image, caption


BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / ".." / "data" / "coco").resolve()

if __name__ == "__main__":
    dataset = COCO_Dataset(
        image_dir=DATA_DIR / "val2017",
        annotation_file=DATA_DIR / "annotations" / "captions_val2017.json",
        transform=TrainTransform(),
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

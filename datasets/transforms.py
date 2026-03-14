from typing import Callable, Optional, Tuple

import torch
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
)


class TrainTransform:
    def __init__(
        self,
        size: int = 224,
        crop_scale: Optional[Tuple] = (0.5, 1.0),
        hflip_p: Optional[float] = 0.5,
        jitter_params: Optional[Tuple[float, float, float, float]] = (
            0.4,
            0.4,
            0.2,
            0.1,
        ),
    ) -> None:
        self.size = size
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.jitter_params = jitter_params
        self.transforms_list: list[Callable] = []

        if self.crop_scale:
            self.transforms_list.append(
                RandomResizedCrop(size=self.size, scale=self.crop_scale)
            )
        else:
            self.transforms_list.append(Resize((self.size, self.size)))
        if self.hflip_p:
            self.transforms_list.append(RandomHorizontalFlip(p=self.hflip_p))
        if self.jitter_params:
            self.transforms_list.append(ColorJitter(*self.jitter_params))
        self.transforms_list.append(
            Normalize(
                mean=[0.47087333, 0.44731208, 0.40772682],
                std=[0.2517867, 0.2472999, 0.25216556],
            )
        )
        self.transform = Compose(self.transforms_list)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)


class ValTransform:
    def __init__(self, size: int = 224, use_ccrop: bool = False) -> None:
        self.size = size
        self.transforms_list: list[Callable] = []
        if use_ccrop:
            self.transforms_list.append(Resize(self.size + 32))
            self.transforms_list.append(CenterCrop(self.size))
        else:
            self.transforms_list.append(Resize((self.size, self.size)))
        self.transforms_list.append(
            Normalize(
                mean=[0.47087333, 0.44731208, 0.40772682],
                std=[0.2517867, 0.2472999, 0.25216556],
            )
        )
        self.transform = Compose(self.transforms_list)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)

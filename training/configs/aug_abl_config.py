from dataclasses import dataclass
from typing import Tuple

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """Does heavier augmentation help or hurt retrieval?"""

    name: str = "abl_augmentation"
    crop_scale: Tuple[float, float] | None = (0.5, 1.0)
    hflip_p: float | None = 0.5
    jitter_params: Tuple[float, float, float, float] | None = (0.4, 0.4, 0.2, 0.1)

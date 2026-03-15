from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """EXP-A: Is the pretrained representation already good enough?"""

    name: str = "abl_frozen_both"
    text_encoder_freeze: bool = True
    image_encoder_freeze: bool = True

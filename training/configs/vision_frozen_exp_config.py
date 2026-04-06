from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """EXP-B: Does fine-tuning text encoder help?"""

    name: str = "abl_frozen_vision"
    text_encoder_freeze: bool = False
    image_encoder_freeze: bool = True

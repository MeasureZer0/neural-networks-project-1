from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """EXP-C: Does fine-tuning vision encoder help?"""

    name: str = "abl_frozen_text"
    text_encoder_freeze: bool = True
    image_encoder_freeze: bool = False

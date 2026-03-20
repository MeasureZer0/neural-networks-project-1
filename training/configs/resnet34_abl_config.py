from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """ABL-1: ResNet34 instead of ViT. How much does vision backbone matter?"""

    name: str = "abl_resnet34"
    image_encoder_type: str = "resnet34"

from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """ABL-1: ResNet50 instead of ViT. How much does vision backbone matter?"""

    name: str = "abl_resnet50"
    image_encoder_type: str = "resnet50"

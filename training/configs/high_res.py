from dataclasses import dataclass

from training.configs.base_config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    name: str = "high_res_config"
    image_dim: int = 1024
    batch_size: int = 16

from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """Smaller projection head."""

    name: str = "abl_embed_256"
    embedding_dim: int = 256

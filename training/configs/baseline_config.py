from dataclasses import dataclass
from typing import Optional, Tuple

from training.configs.base_config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    name: str = "baseline"

    # Augmentation
    crop_scale: Optional[Tuple[float, float]] = (0.9, 1.0)
    hflip_p: Optional[float] = None
    jitter_params: None = None
    use_ccrop: bool = True

    # Optimizer
    lr: float = 5e-4
    weight_decay: float = 0.2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-6

    # LR Schedule — warmup + cosine decay
    warmup_epochs: int = 5
    use_cosine_schedule: bool = True

    # Tokenizer
    tokenizer: str = "openai/clip-vit-base-patch32"
    tokenizer_maxlength: int = 77

    # Model
    text_encoder_type: str = "clip"
    image_encoder_type: str = "vit"
    text_encoder_freeze: bool = False
    image_encoder_freeze: bool = False
    embedding_dim: int = 512

    # Training
    epochs: int = 30
    batch_size: int = 8
    use_fp16: bool = True
    grad_clip_norm: float = 1.0

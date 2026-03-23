from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """Final model."""

    name: str = "final_model"
    # Optimizer
    lr: int = 3e-5
    weight_decay: float = 5e-4
    adam_beta1: float = 0.95
    adam_beta2: float = 0.99
    adam_eps: float = 1e-7

    # LR Schedule — warmup
    warmup_epochs: int = 3

    # Training
    epochs: int = 20
    grad_clip_norm: float = 2.0
    temperature: float = 0.075

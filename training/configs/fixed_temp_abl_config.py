from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """How much does learnable temperature contribute?"""

    name: str = "abl_fixed_temperature"
    learn_temperature: bool = False
    init_temperature: float = 0.07

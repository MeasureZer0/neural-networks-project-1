from dataclasses import dataclass

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    """ABL-2: DistilRoBERTa instead of CLIP text transformer. Does text encoder matter?"""

    name: str = "abl_distilroberta"
    text_encoder_type: str = "distilroberta"
    tokenizer: str = "distilroberta-base"
    tokenizer_maxlength: int = 32

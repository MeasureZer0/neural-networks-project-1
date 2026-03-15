from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class Config:
    name: str = "baseline"

    # Augmentation
    size: int = 224
    crop_scale: Optional[Tuple[float, float]] = (0.9, 1.0)
    hflip_p: Optional[float] = None
    jitter_params: Optional[Tuple[float, float, float, float]] = None
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
    batch_size: int = 32
    use_fp16: bool = True
    grad_clip_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    data_dir: Path = Path("data/coco")
    train_image_dir: Path = data_dir / "train2017"
    train_annotation_file: Path = data_dir / "annotations" / "captions_train2017.json"
    val_image_dir: Path = data_dir / "val2017"
    val_annotation_file: Path = data_dir / "annotations" / "captions_val2017.json"
    checkpoint_dir: str = "checkpoints"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "multimodal-clip-experiments"

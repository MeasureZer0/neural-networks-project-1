from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class Config:
    # Metadata
    name: str = "base_config"

    # Transformations
    size: int = 224
    crop_scale: Optional[Tuple] = (0.5, 1.0)
    hflip_p: Optional[float] = 0.5
    jitter_params: Optional[Tuple[float, float, float, float]] = (
        0.4,
        0.4,
        0.2,
        0.1,
    )
    use_ccrop: bool = False

    # Training Loop
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True

    # Model Architecture
    tokenizer: str = "openai/clip-vit-base-patch32"
    tokenizer_maxlength: int = 77
    text_encoder_type: str = "clip"
    image_encoder_type: str = "vit"
    text_encoder_freeze: bool = False
    image_encoder_freeze: bool = False
    embedding_dim: int = 256

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

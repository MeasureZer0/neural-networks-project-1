from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    # Metadata
    name: str = "base_config"

    # Training Loop
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
    use_wandb: bool = False
    wandb_project: str = "clip-training"

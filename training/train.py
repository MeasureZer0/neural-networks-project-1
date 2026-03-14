import argparse
import importlib

import torch
from torch.utils.data import DataLoader

from datasets.coco_dataset import COCO_Dataset
from datasets.transforms import TrainTransform, ValTransform
from models.contrastive_model import ContrastiveModel
from training.checkpointing import load_checkpoint
from training.configs.base_config import Config
from training.loss import InfoNCELoss
from training.trainer import Trainer


def get_config(config_name: str) -> Config:
    try:
        # Try to import from training.configs
        module_name = f"training.configs.{config_name}"
        module = importlib.import_module(module_name)
        # Expecting a class named 'Config' in the module
        if hasattr(module, "Config"):
            config_cls = module.Config
            # If it's a class (not the base Config instance we might have imported)
            if isinstance(config_cls, type) and issubclass(config_cls, Config):
                return config_cls()
            # If it's already an instance of Config (like in base_config if we weren't careful)
            elif isinstance(config_cls, Config):
                return config_cls
    except (ImportError, AttributeError) as e:
        print(f"Error loading config {config_name}: {e}")
        print("Using default Config")

    return Config()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Contrastive CLIP-like model")
    parser.add_argument(
        "--config",
        type=str,
        default="base_config",
        help="Name of the config file to use",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    config = get_config(args.config)
    print(f"Using config: {config.name}")

    # Conceptual setup
    model = ContrastiveModel(
        text_encoder_type=config.text_encoder_type,
        image_encoder_type=config.image_encoder_type,
        text_encoder_freeze=config.text_encoder_freeze,
        image_encoder_freeze=config.image_encoder_freeze,
        embedding_dim=config.embedding_dim,
    ).to(config.device)

    criterion = InfoNCELoss().to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    scheduler = None  # Can add StepLR or CosineAnnealingLR

    start_epoch = 1
    if args.resume:
        start_epoch, val_loss = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        model.to(config.device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, val_loss: {val_loss:.4f}")

    # Data loaders
    train_dataset = COCO_Dataset(
        image_dir=config.train_image_dir,
        annotation_file=config.train_annotation_file,
        img_transform=TrainTransform(),
        tokenizer=config.tokenizer,
        tokenizer_maxlength=config.tokenizer_maxlength,
    )

    val_dataset = COCO_Dataset(
        image_dir=config.val_image_dir,
        annotation_file=config.val_annotation_file,
        img_transform=ValTransform(),
        tokenizer=config.tokenizer,
        tokenizer_maxlength=config.tokenizer_maxlength,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        config=config,
        start_epoch=start_epoch,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

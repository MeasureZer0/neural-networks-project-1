import argparse
import importlib

import torch
from torch.utils.data import DataLoader

from datasets.coco_dataset import COCO_Dataset
from datasets.transforms import TrainTransform
from models.clip import SimpleCLIP
from models.dummy import TextEncoder, VisionEncoder
from training.configs.base_config import Config
from training.loss import CLIPLoss
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
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument(
        "--config",
        type=str,
        default="base_config",
        help="Name of the config file to use",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    print(f"Using config: {config.name}")

    # Initialize components (Placeholders for encoders)
    image_encoder = VisionEncoder(output_dim=config.image_dim)
    text_encoder = TextEncoder(output_dim=config.text_dim)

    # Conceptual setup
    model = SimpleCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_dim=config.image_dim,
        text_dim=config.text_dim,
        embed_dim=config.embed_dim,
    ).to(config.device)

    criterion = CLIPLoss().to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    scheduler = None  # Can add StepLR or CosineAnnealingLR

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
        img_transform=TrainTransform(),
        tokenizer=config.tokenizer,
        tokenizer_maxlength=config.tokenizer_maxlength,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        config=config,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

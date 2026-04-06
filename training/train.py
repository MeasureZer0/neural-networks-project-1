import argparse
import importlib
import logging
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.contrastive_model import ContrastiveModel
from torch_datasets.coco_dataset import COCO_Dataset
from torch_datasets.transforms import TrainTransform, ValTransform
from training.checkpointing import load_checkpoint
from training.configs.baseline_config import Config
from training.loss import InfoNCELoss
from training.trainer import Trainer

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


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


def apply_sweep_config(config: Config) -> Config:
    import wandb

    if wandb.run is None:
        return config

    sweep = wandb.config

    for field in [
        "lr",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "warmup_epochs",
        "embedding_dim",
        "grad_clip_norm",
        "learn_temperature",
        "init_temperature",
        "use_ccrop",
        "epochs",
        "batch_size",
    ]:
        if field in sweep:
            setattr(config, field, sweep[field])

    if "crop_scale_min" in sweep:
        config.crop_scale = (sweep["crop_scale_min"], 1.0)

    if "hflip_p" in sweep:
        config.hflip_p = sweep["hflip_p"] or None

    if "use_jitter" in sweep:
        config.jitter_params = (0.4, 0.4, 0.2, 0.1) if sweep["use_jitter"] else None

    return config


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Contrastive CLIP-like model")
    parser.add_argument(
        "--config",
        type=str,
        default="baseline_config",
        help="Name of the config file to use",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    config = get_config(args.config)
    config = apply_sweep_config(config)
    print(f"Using config: {config.name}")

    # Conceptual setup
    model = ContrastiveModel(
        text_encoder_type=config.text_encoder_type,
        image_encoder_type=config.image_encoder_type,
        text_encoder_freeze=config.text_encoder_freeze,
        image_encoder_freeze=config.image_encoder_freeze,
        embedding_dim=config.embedding_dim,
    ).to(config.device)

    # InfoNCELoss

    criterion = InfoNCELoss(
        learn_temperature=getattr(config, "learn_temperature", True),
        init_temperature=getattr(config, "init_temperature", 0.07),
    ).to(config.device)

    # Optimizer

    decay, no_decay = [], []

    for _, name, param in [
        *((model, n, p) for n, p in model.named_parameters()),
        *((criterion, n, p) for n, p in criterion.named_parameters()),
    ]:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(
            getattr(config, "adam_beta1", 0.9),
            getattr(config, "adam_beta2", 0.98),
        ),
        eps=getattr(config, "adam_eps", 1e-6),
    )

    # LR Schedule

    scheduler = None
    if getattr(config, "use_cosine_schedule", False):
        estimated_steps_per_epoch = 118_000 // config.batch_size
        total_steps = estimated_steps_per_epoch * config.epochs
        warmup_steps = estimated_steps_per_epoch * getattr(config, "warmup_epochs", 5)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        print(
            f"Scheduler: cosine with {getattr(config, 'warmup_epochs', 5)} warmup epochs "
            f"({warmup_steps} steps) / {total_steps} total steps"
        )

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
        img_transform=TrainTransform(
            size=config.size,
            crop_scale=config.crop_scale,
            hflip_p=config.hflip_p,
            jitter_params=config.jitter_params,
        ),
        tokenizer=config.tokenizer,
        tokenizer_maxlength=config.tokenizer_maxlength,
    )

    val_dataset = COCO_Dataset(
        image_dir=config.val_image_dir,
        annotation_file=config.val_annotation_file,
        img_transform=ValTransform(use_ccrop=config.use_ccrop),
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

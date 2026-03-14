import os
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.configs.base_config import Config


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: str,
        config: Config,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.checkpoint_dir = getattr(config, "checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Future-proofing for wandb or other loggers
        self.use_wandb = getattr(config, "use_wandb", False)
        if self.use_wandb:
            import wandb

            self.wandb = wandb
            self.wandb.init(
                project=getattr(config, "wandb_project", "clip-training"),
                config=vars(config),
            )

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

        for batch in pbar:
            images = batch["images"].to(self.device)
            tokens = {k: v.to(self.device) for k, v in batch["tokens"].items()}

            self.optimizer.zero_grad()

            # Forward pass
            image_features, text_features = self.model(images, tokens)

            # Loss calculation
            loss, _, _ = self.criterion(image_features, text_features)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            # Log to wandb if enabled
            if self.use_wandb:
                self.wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate_one_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Validating Epoch {epoch}")

        for images, texts in pbar:
            images = images.to(self.device)
            texts = texts.to(self.device)

            image_features, text_features = self.model(images, texts)
            loss, _, _ = self.criterion(image_features, text_features)

            total_loss += loss.item()
            pbar.set_postfix({"val_loss": loss.item()})

        avg_loss = total_loss / len(dataloader)

        if self.use_wandb:
            self.wandb.log({"val/loss": avg_loss})

        return avg_loss

    def save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False
    ) -> None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = getattr(self.config, "name", "base_config")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "val_loss": val_loss,
            "config": self.config,
            "timestamp": timestamp,
        }

        filename = f"{config_name}_{timestamp}_epoch_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir, f"{config_name}_best_model.pt"
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = getattr(self.config, "epochs", 10)
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.validate_one_epoch(val_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best=is_best)

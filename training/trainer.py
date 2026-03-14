import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.checkpointing import save_checkpoint
from training.configs.base_config import Config


def _mean_reciprocal_rank(logits: torch.Tensor) -> torch.Tensor:
    # [B, B]
    targets = torch.arange(logits.shape[0], device=logits.device)
    sorted_indices = logits.argsort(dim=-1, descending=True)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1
    return (1.0 / ranks.float()).mean()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: str,
        config: Config,
        start_epoch: int = 1,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.start_epoch = start_epoch

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

    def _batch_to_device(
        self, batch: Dict[str, torch.Tensor | Dict]
    ) -> Tuple[torch.Tensor, Dict]:
        images = batch["images"].to(self.device)  # type: ignore
        tokens = {k: v.to(self.device) for k, v in batch["tokens"].items()}  # type: ignore
        return images, tokens

    def _batch_metrics(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logits_per_images: torch.Tensor,
        prefix: str = "train",
    ) -> Dict:
        targets = torch.arange(image_features.shape[0], device=image_features.device)

        # Batch accuracy
        batch_acc = (logits_per_images.argmax(dim=-1) == targets).float().mean()

        # MMR
        mrr = _mean_reciprocal_rank(logits_per_images)

        # Diagonal similarity
        sim_matrix = image_features @ text_features.T
        diag_sim = sim_matrix.diagonal().mean()

        # Off-diagonal similarity
        n = sim_matrix.shape[0]
        off_diag_sim = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (n * n - n)

        return {
            f"{prefix}/batch_accuracy": batch_acc.item(),
            f"{prefix}/mrr": mrr.item(),
            f"{prefix}/diagonal_similarity": diag_sim.item(),
            f"{prefix}/off_diagonal_similarity": off_diag_sim.item(),
        }

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        accumulated_metrics: dict = {}
        n_batches = len(dataloader)
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

        for batch in pbar:
            images, tokens = self._batch_to_device(batch)

            self.optimizer.zero_grad()
            image_features, text_features = self.model(images, tokens)
            loss, logits_per_image, _ = self.criterion(image_features, text_features)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            total_loss += loss.item()

            metrics = self._batch_metrics(
                image_features, text_features, logits_per_image, prefix="train"
            )
            for key, value in metrics.items():
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.use_wandb:
                self.wandb.log(
                    {
                        "batch/loss": loss.item(),
                        "batch/grad_norm": grad_norm.item(),
                        "batch/temperature": self.criterion.logit_scale.exp().item(),  # type: ignore
                    }
                )

        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in accumulated_metrics.items()}

        if self.use_wandb:
            self.wandb.log(
                {
                    "epoch/train_loss": avg_loss,
                    "epoch/train_lr": self.optimizer.param_groups[0]["lr"],
                    **avg_metrics,
                }
            )

        return avg_loss

    @torch.no_grad()
    def validate_one_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        accumulated_metrics: dict = {}
        n_batches = len(dataloader)
        pbar = tqdm(dataloader, desc=f"Validating Epoch {epoch}")

        for batch in pbar:
            images, tokens = self._batch_to_device(batch)
            image_features, text_features = self.model(images, tokens)
            loss, logits_per_image, _ = self.criterion(image_features, text_features)

            total_loss += loss.item()

            metrics = self._batch_metrics(
                image_features, text_features, logits_per_image, prefix="val"
            )
            for key, value in metrics.items():
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in accumulated_metrics.items()}

        if self.use_wandb:
            self.wandb.log(
                {
                    "epoch/val_loss": avg_loss,
                    **avg_metrics,
                }
            )

        return avg_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = getattr(self.config, "epochs", 10)
        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, epochs + 1):
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

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = getattr(self.config, "name", "base_config")

            state = {
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

            save_checkpoint(
                state=state,
                checkpoint_dir=self.checkpoint_dir,
                config_name=config_name,
                is_best=is_best,
            )

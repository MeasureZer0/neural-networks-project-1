import os
from datetime import datetime
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: Union[str, os.PathLike],
    config_name: str = "baseline_config",
    filename: Optional[str] = None,
    is_best: bool = False,
) -> None:
    """
    Save training checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config_name}_{timestamp}_checkpoint.pth.tar"

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, f"{config_name}_model_best.pth"))


def load_checkpoint(
    checkpoint_path: Union[str, os.PathLike],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[int, float]:
    """
    Load training checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["epoch"], checkpoint["val_loss"]

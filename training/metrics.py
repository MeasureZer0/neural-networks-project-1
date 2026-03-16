from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def mean_reciprocal_rank(logits: torch.Tensor) -> float:
    B = logits.shape[0]
    targets = torch.arange(B, device=logits.device)
    sorted_indices = logits.argsort(dim=-1, descending=True)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1
    return (1.0 / ranks.float()).mean().item()


def similarity_stats(
    image_features: torch.Tensor, text_features: torch.Tensor
) -> Dict[str, float]:
    sim_matrix = image_features @ text_features.T
    diag = sim_matrix.diagonal()
    n = sim_matrix.shape[0]
    off_diag_sum = sim_matrix.sum() - diag.sum()

    return {
        "diag_sim": diag.mean().item(),
        "off_diag_sim": (off_diag_sum / (n * n - n)).item(),
    }


def recall_at_k(logits: torch.Tensor, ks: List[int]) -> Dict[str, float]:
    B = logits.shape[0]
    targets = torch.arange(B, device=logits.device)
    sorted_indices = logits.argsort(dim=-1, descending=True)

    results = {}
    for k in ks:
        k_clamped = min(k, B)
        topk = sorted_indices[:, :k_clamped]
        hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item()
    return results


@torch.no_grad()
def full_retrieval_eval(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    ks: List[int],
    use_fp16: bool = False,
) -> Dict[str, float]:
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    for batch in tqdm(dataloader, desc="Full retrival eval", leave=False):
        images = batch["images"].to(device)
        tokens = {k: v.to(device) for k, v in batch["tokens"].items()}

        with torch.cuda.amp.autocast(enabled=use_fp16):
            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(tokens)

        all_image_embeds.append(image_embeds.float().cpu())
        all_text_embeds.append(text_embeds.float().cpu())
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    sim_matrix = image_embeds @ text_embeds.T

    results = {}
    i2t = recall_at_k(sim_matrix, ks=ks)
    for key, val in i2t.items():
        results[f"full/i2t_{key}"] = val
    t2i = recall_at_k(sim_matrix.T, ks=ks)
    for key, val in t2i.items():
        results[f"full/t2i_{key}"] = val
    return results

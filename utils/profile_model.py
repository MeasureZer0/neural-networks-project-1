from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from models.contrastive_model import ContrastiveModel
from models.text_encoder import TextEncoder
from models.visual_encoder import ImageEncoder
from torch_datasets.coco_dataset import COCO_Dataset
from torch_datasets.transforms import ValTransform
from utils.profiler import flop_stats, model_stats, print_stats, vram_stats


def profile_encoders(
    batch: Dict,
    device: torch.device,
) -> None:
    images = batch["images"]
    tokens = {k: v for k, v in batch["tokens"].items()}

    for use_fp16 in [False, True]:
        precision = "FP16" if use_fp16 else "FP32"

        print(f"\n\n{'#' * 55}")
        print(f"  IMAGE ENCODERS — {precision}")
        print(f"{'#' * 55}")

        for model_type in ["resnet18", "resnet34", "vit"]:
            model = ImageEncoder(model_type=model_type)
            model.eval()

            stats = {}
            stats.update(model_stats(model, use_fp16=use_fp16))
            stats.update(flop_stats(model, images))
            stats.update(vram_stats(model, device, images, use_fp16=use_fp16))
            print_stats(f"ImageEncoder: {model_type} ({precision})", stats)

        print(f"\n\n{'#' * 55}")
        print(f"  TEXT ENCODERS — {precision}")
        print(f"{'#' * 55}")

        for model_type in ["distilroberta", "clip"]:
            model = TextEncoder(model_type=model_type)
            model.eval()

            stats = {}
            text_inputs = (tokens["input_ids"], tokens["attention_mask"])
            try:
                stats.update(flop_stats(model, text_inputs))
            except Exception as e:
                print(f"  [FLOPs Error] Skipping FLOPs for {model_type}: {e}")
                stats["flops_G"] = 0.0
            stats.update(vram_stats(model, device, tokens, use_fp16=use_fp16))
            print_stats(f"TextEncoder: {model_type} ({precision})", stats)


def profile_contrastive_models(batch: Dict, device: torch.device) -> None:
    images = batch["images"]
    tokens = {k: v for k, v in batch["tokens"].items()}

    configs = [
        {
            "name": "EXP-A: ViT + CLIP (full fine-tune)",
            "text_encoder_type": "clip",
            "image_encoder_type": "vit",
            "text_encoder_freeze": False,
            "image_encoder_freeze": False,
        },
        {
            "name": "EXP-B: ViT + CLIP (frozen vision)",
            "text_encoder_type": "clip",
            "image_encoder_type": "vit",
            "text_encoder_freeze": False,
            "image_encoder_freeze": True,
        },
        {
            "name": "EXP-C: ViT + CLIP (frozen text)",
            "text_encoder_type": "clip",
            "image_encoder_type": "vit",
            "text_encoder_freeze": True,
            "image_encoder_freeze": False,
        },
        {
            "name": "EXP-D: ViT + CLIP (frozen both)",
            "text_encoder_type": "clip",
            "image_encoder_type": "vit",
            "text_encoder_freeze": True,
            "image_encoder_freeze": True,
        },
        {
            "name": "ABL-1: ResNet34 + CLIP",
            "text_encoder_type": "clip",
            "image_encoder_type": "resnet34",
            "text_encoder_freeze": False,
            "image_encoder_freeze": False,
        },
        {
            "name": "ABL-2: ViT + DistilRoBERTa",
            "text_encoder_type": "distilroberta",
            "image_encoder_type": "vit",
            "text_encoder_freeze": False,
            "image_encoder_freeze": False,
        },
    ]

    for use_fp16 in [False, True]:
        precision = "FP16" if use_fp16 else "FP32"

        print(f"\n\n{'#' * 55}")
        print(f"  FULL MODELS — {precision}")
        print(f"{'#' * 55}")

        for cfg in configs:
            name = cfg["name"]
            model = ContrastiveModel(
                text_encoder_type=cfg["text_encoder_type"],
                image_encoder_type=cfg["image_encoder_type"],
                text_encoder_freeze=cfg["text_encoder_freeze"],
                image_encoder_freeze=cfg["image_encoder_freeze"],
            )
            model.eval()

            stats = {}
            stats.update(model_stats(model, use_fp16=use_fp16))
            stats.update(
                vram_stats(
                    model,
                    device,
                    sample_input=(images, tokens),
                    use_fp16=use_fp16,
                )
            )
            print_stats(f"{name} ({precision})", stats)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BASE_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (BASE_DIR / ".." / "data" / "coco").resolve()

    dataset = COCO_Dataset(
        image_dir=DATA_DIR / "val2017",
        annotation_file=DATA_DIR / "annotations" / "captions_val2017.json",
        img_transform=ValTransform(),
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    profile_encoders(batch, device)
    profile_contrastive_models(batch, device)

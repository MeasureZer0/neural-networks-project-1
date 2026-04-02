import argparse
import sys
from pathlib import Path

from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader

from models.inferencer import ModelInferencer
from torch_datasets.flickr8k_dataset import Flickr8k_Dataset
from torch_datasets.transforms import ValTransform
from training.metrics import full_retrieval_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flickr8k retrieval evaluation")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    inferencer = ModelInferencer(args.checkpoint)
    config = inferencer.config

    hf_dataset = hf_load_dataset("jxie/flickr8k", split=args.split)
    dataset = Flickr8k_Dataset(
        hf_dataset=hf_dataset,
        tokenizer=config.tokenizer,
        tokenizer_maxlength=config.tokenizer_maxlength,
        img_transform=ValTransform(use_ccrop=config.use_ccrop),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    metrics = full_retrieval_eval(
        model=inferencer.model,
        dataloader=dataloader,
        device=inferencer.device,
        ks=args.ks,
        use_fp16=args.fp16,
    )

    print(f"\nFlickr8k retrieval [{args.split}] — {len(dataset)} samples")
    print(f"{'Metric':<20} {'Score':>8}")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name:<20} {value:>8.4f}")


if __name__ == "__main__":
    main()

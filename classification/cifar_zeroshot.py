import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import PILToTensor
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot CIFAR evaluation")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10", "cifar100"),
        default="cifar10",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/cifar"),
        help="Where CIFAR should be stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="a photo of a {}",
        help="Prompt template used to build class descriptions",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR automatically if missing",
    )
    return parser.parse_args()


def load_dataset(
    dataset_name: str, data_root: Path, download: bool
) -> CIFAR10 | CIFAR100:
    dataset_cls = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    return dataset_cls(
        root=str(data_root),
        train=False,
        download=download,
        transform=PILToTensor(),
    )


def build_class_prompts(class_names: list[str], template: str) -> list[str]:
    prompts = []
    for class_name in class_names:
        normalized_name = class_name.replace("_", " ")
        try:
            prompts.append(template.format(normalized_name))
        except IndexError:
            prompts.append(template.format(class_name=normalized_name))
    return prompts


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models.inferencer import ModelInferencer

    inferencer = ModelInferencer(args.checkpoint)
    dataset = load_dataset(args.dataset, args.data_root, args.download)
    class_prompts = build_class_prompts(dataset.classes, args.prompt_template)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=f"Zero-shot {args.dataset}"):
        predictions, logits = inferencer.classify_zero_shot(
            images=images,
            class_prompts=class_prompts,
            image_batch_size=len(images),
        )
        labels = labels.cpu()

        top1_correct += (predictions == labels).sum().item()
        top5 = logits.topk(k=min(5, logits.shape[1]), dim=-1).indices
        top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    print(f"Dataset: {args.dataset}")
    print(f"Classes: {len(class_prompts)}")
    print(f"Prompt template: {args.prompt_template}")
    print(f"Top-1 accuracy: {top1_correct / total:.4f}")
    print(f"Top-5 accuracy: {top5_correct / total:.4f}")


if __name__ == "__main__":
    main()

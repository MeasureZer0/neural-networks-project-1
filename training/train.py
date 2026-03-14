import torch
from torch.utils.data import DataLoader

from datasets.coco_dataset import COCO_Dataset
from datasets.transforms import TrainTransform
from models.clip import SimpleCLIP
from models.dummy import TextEncoder, VisionEncoder
from training.config import Config
from training.loss import CLIPLoss
from training.trainer import Trainer


def main() -> None:
    config = Config()

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

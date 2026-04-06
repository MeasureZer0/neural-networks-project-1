import random
from pathlib import Path

from datasets.coco_dataset import COCO_Dataset
from datasets.transforms import ValTransform
from torch.utils.data import DataLoader, Subset

from models.inferencer import ModelInferencer
from models.retrieval import EmbeddingIndex
from torch_datasets.coco_dataset import COCO_Dataset
from torch_datasets.transforms import ValTransform
from training.configs.baseline_config import Config

if __name__ == "__main__":
    config = Config()
    inferencer = ModelInferencer("checkpoints/final_model_model_best.pth")

    index_images_path = Path("checkpoints/index_images.faiss")
    index_texts_path = Path("checkpoints/index_texts.faiss")

    max_samples = 1000

    val_dataset = COCO_Dataset(
        image_dir=config.val_image_dir,
        annotation_file=config.val_annotation_file,
        img_transform=ValTransform(use_ccrop=config.use_ccrop),
        return_meta=True,
    )

    if not index_images_path.exists():
        val_loader = DataLoader(
            Subset(val_dataset, range(max_samples)),
            batch_size=64,
            shuffle=False,
            num_workers=4,
        )

        image_index, text_index = inferencer.build_index_from_dataloader(
            val_loader,
            image_save_path=index_images_path,
            text_save_path=index_texts_path,
        )
    else:
        image_index = EmbeddingIndex(config.embedding_dim)
        image_index.load(index_images_path)
        text_index = EmbeddingIndex(config.embedding_dim)
        text_index.load(index_texts_path)

    random_idx = random.randint(0, max_samples - 1)
    sample = val_dataset[random_idx]

    query_caption = sample["caption"]
    query_image_path = Path(sample["path"])  # type: ignore

    print(f"Query caption: {query_caption}")
    print(f"Query image:   {query_image_path}")

    t2i_results = inferencer.text_to_image(
        queries=[query_caption],  # type: ignore
        image_index=image_index,
        k=5,
    )
    print("t2i:", t2i_results)

    i2t_results = inferencer.image_to_text(
        image_paths=[query_image_path],
        text_index=text_index,
        k=5,
    )
    print("i2t:", i2t_results)

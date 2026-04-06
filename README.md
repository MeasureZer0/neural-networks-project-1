# simple-clip - CLIP-style Vision-Language Model from Scratch

A from-scratch implementation of a **CLIP-style contrastive multimodal model** trained to align image and text embeddings in a shared latent space. The model supports text-to-image retrieval, image-to-text retrieval, zero-shot image classification, and interactive embedding space exploration.

---

## Architecture

The model follows the dual-encoder contrastive learning paradigm introduced by [Radford et al. (2021)](#references):

```
Image → ImageEncoder → ProjectionHead → L2-normalized embedding ─┐
                                                                    ├──► InfoNCE Loss
Text  → TextEncoder  → ProjectionHead → L2-normalized embedding ─┘
```

### Encoders

**Image encoder** (`models/visual_encoder.py`) — one of:
- `vit` — frozen `openai/clip-vit-base-patch32` ViT, [CLS] token extracted
- `resnet34` / `resnet18` — ImageNet-pretrained ResNets, fc layer replaced with `nn.Identity`

**Text encoder** (`models/text_encoder.py`) — one of:
- `clip` — `openai/clip-vit-base-patch32` text transformer, [EOS] token extracted
- `distilroberta` — `distilroberta-base`, mean pooling over attention mask

**Projection head** (`models/projection.py`) — a single linear layer followed by L2 normalization, mapping encoder outputs to a shared `embedding_dim`-dimensional space (default 512).

### Loss

Symmetric InfoNCE loss (`training/loss.py`) with a learnable log-scale temperature parameter (CLIP-style `logit_scale = log(1/τ)`). The loss is the average of image→text and text→image cross-entropy over in-batch negatives.

---

## Datasets

Training and evaluation use:

| Dataset | Usage |
|---|---|
| **MS-COCO 2017** ([Lin et al., 2014](#references)) | Primary train/val dataset. Each image is paired with 5 captions; one is sampled randomly per iteration. |
| **Flickr30k** ([Young et al., 2014](#references)) | Alternative dataset via `Flickr30k_Dataset`. |
| **CIFAR-10 / CIFAR-100** ([Krizhevsky, 2009](#references)) | Zero-shot classification evaluation. |

Data is expected at `data/coco/` and `data/flickr30k/`. Download and symlink as needed.

---

## Training

```bash
python -m training.train --config baseline_config
```

Available configs in `training/configs/`:

| Config | Description |
|---|---|
| `baseline_config` | ViT image encoder + CLIP text encoder, image encoder frozen |
| `final_model_config` | Tuned hyperparameters, 20 epochs |
| `both_frozen_exp_config` | Both encoders frozen — tests pretrained alignment |
| `vision_frozen_exp_config` | Vision frozen, text fine-tuned |
| `text_frozen_exp_config` | Text frozen, vision fine-tuned |
| `resnet34_abl_config` | ResNet34 image encoder ablation |
| `distilroberta_abl_config` | DistilRoBERTa text encoder ablation |
| `embed_128_abl_config` / `embed_256_abl_config` | Projection dimension ablations |
| `aug_abl_config` | Heavy augmentation ablation |
| `fixed_temp_abl_config` | Fixed vs. learnable temperature ablation |

Resume from a checkpoint:

```bash
python -m training.train --config final_model_config --resume checkpoints/last.ckpt
```

### Hyperparameter Sweeps

Bayesian sweeps via Weights & Biases:

```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

The sweep explores learning rate, weight decay, Adam betas, gradient clipping, warmup epochs, embedding dimension, temperature, and augmentation parameters.

---

## Evaluation

### Retrieval (COCO val2017)

```bash
python evaluate.py
```

Builds FAISS indices over up to 1 000 val images/captions and runs text→image and image→text retrieval, printing top-5 results.

### Zero-shot Classification (CIFAR)

```bash
python -m classification.cifar_zeroshot checkpoints/final_model_model_best.pth \
    --dataset cifar10 \
    --download
```

Uses prompt template `"a photo of a {}"` by default. Reports Top-1 and Top-5 accuracy.

### Training-time Metrics

Logged every epoch (and to W&B if enabled):

- **Recall@K** (K = 1, 5, 10) for image→text and text→image
- **Mean Reciprocal Rank (MRR)**
- Mean diagonal / off-diagonal cosine similarity
- Full retrieval eval on the last epoch

---

## Interactive App

```bash
python app/main.py
```

A Tkinter GUI that loads a checkpoint and COCO val2017, then exposes:

- **Text → Image search** — retrieve images by caption query
- **Image → Image search** — find visually similar images
- **Image → Text retrieval** — find matching captions for an uploaded image
- **Zero-shot classification** — label an image against a custom comma-separated class list
- **Embedding interpolation** — slider between two text embeddings to explore the latent space

---

## Retrieval Backend

`EmbeddingIndex` (`models/retrieval.py`) wraps `faiss.IndexFlatIP` (exact inner-product search) with pickle-serialized metadata. Indices can be saved to and loaded from disk.

`ModelInferencer` (`models/inferencer.py`) provides a high-level API for:

```python
inferencer = ModelInferencer("checkpoints/final_model_model_best.pth")

# Embed
img_emb = inferencer.embed_image("path/to/image.jpg")   # [N, D]
txt_emb = inferencer.embed_text("a dog on a beach")     # [N, D]

# Retrieve
results = inferencer.text_to_image(["a dog"], image_index, k=5)

# Classify
predictions, logits = inferencer.classify_zero_shot(images, class_prompts)
```

---

## Model Profiling

```bash
python utils/profile_model.py
```

Reports parameter counts (total / trainable / frozen), memory footprint, GFLOPs, and VRAM peak for each encoder configuration in both FP32 and FP16, using `fvcore`.

---

## Installation

```bash
git clone https://github.com/MeasureZer0/neural-networks-project-1.git nn-1
cd nn-1
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

Requires Python 3.13. PyTorch is resolved automatically:

- **macOS** — CPU wheel
- **Linux / Windows** — CUDA 13.0 wheel

---

## Results

Best configuration: **ViT + CLIP text transformer, vision encoder frozen** (`final_model_config`), trained for 20 epochs on MS-COCO 2017 train (~118k images) with FP16 mixed precision.

### Retrieval — COCO val2017 (1 000 samples)

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Image → Text (i2t) | 0.377 | 0.677 | 0.792 |
| Text → Image (t2i) | 0.380 | 0.660 | 0.775 |

### Zero-Shot Classification — CIFAR-10

| Metric | Score |
|---|---|
| Top-1 accuracy | **81.02%** |
| Top-5 accuracy | **98.09%** |

Prompt template: `"a photo of a {}"`.

### Training Curve (final model, 20 epochs)

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 3.268 | 2.511 |
| 5 | 0.799 | 1.507 |
| 10 | 0.580 | 1.354 |
| 15 | 0.470 | 1.283 |
| 20 | **0.428** | **1.259** |

### Model Profiling (FP16)

| Model | Params (M) | Trainable (M) | Size (MB) | VRAM peak (MB) |
|---|---|---|---|---|
| ViT + CLIP (full fine-tune) | 150.95 | 150.95 | 287.91 | 312.59 |
| ViT + CLIP (frozen vision) ✓ | 150.95 | 63.49 | 287.91 | 312.59 |
| ViT + CLIP (frozen text) | 150.95 | 87.78 | 287.91 | 312.59 |
| ViT + CLIP (frozen both) | 150.95 | 0.33 | 287.91 | 312.59 |
| ResNet34 + CLIP | 84.71 | 84.71 | 161.58 | 179.81 |
| ViT + DistilRoBERTa | 169.97 | 169.97 | 324.19 | 349.37 |

✓ — best performing configuration.

---

## References

- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
- Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). *From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions*. TACL. [ACL Anthology](https://aclanthology.org/Q14-1006/)
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto. [Link](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE T-BD. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734) *(FAISS)*

---

## License

MIT — see [LICENSE](LICENSE).
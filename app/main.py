import json
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.io as io
from PIL import Image, ImageTk

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inferencer import ModelInferencer
from models.retrieval import EmbeddingIndex


class EmbeddingExplorerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Embedding Space Explorer")
        self.root.geometry("1000x800")

        self.inferencer: ModelInferencer | None = None
        self.image_index: Any | None = None
        self.results: list = []
        self.annotations: list[str] = []
        self.text_embeddings: torch.Tensor | None = None

        self._emb_a: Any | None = None
        self._emb_b: Any | None = None
        self._photo_refs: list = []  # prevent GC

        self.setup_ui()
        self.load_model_async()

    def _get_cache_paths(self, checkpoint_path: Path) -> tuple[Path, Path]:
        cache_dir = Path("checkpoints")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = checkpoint_path.stem
        image_index_path = cache_dir / f"{cache_key}_coco_val2017_image_index.faiss"
        text_embeddings_path = (
            cache_dir / f"{cache_key}_coco_val2017_text_embeddings.pt"
        )
        return image_index_path, text_embeddings_path

    def _load_or_build_image_index(
        self,
        image_paths: list[Path],
        cache_path: Path,
    ) -> EmbeddingIndex:
        assert self.inferencer is not None

        if cache_path.exists() and cache_path.with_suffix(".pkl").exists():
            index = EmbeddingIndex(self.inferencer.config.embedding_dim)
            index.load(cache_path)
            return index

        index = self.inferencer.build_image_index(image_paths, save_path=cache_path)
        return index

    def _load_or_build_text_embeddings(
        self,
        annotations: list[str],
        cache_path: Path,
    ) -> torch.Tensor:
        assert self.inferencer is not None

        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_annotations = payload.get("annotations")
            cached_embeddings = payload.get("embeddings")
            if cached_annotations == annotations and isinstance(
                cached_embeddings, torch.Tensor
            ):
                return cached_embeddings

        embeddings = self.inferencer.embed_text(annotations)
        torch.save(
            {"annotations": annotations, "embeddings": embeddings},
            cache_path,
        )
        return embeddings

    def setup_ui(self) -> None:
        # Layout components
        top_frame = tk.Frame(self.root)
        top_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.search_entry = tk.Entry(top_frame, width=80)
        self.search_entry.pack(side="left", padx=5)
        self.search_entry.bind("<Return>", lambda e: self.search_async())

        self.search_button = tk.Button(
            top_frame, text="Search Caption", command=self.search_async
        )
        self.search_button.pack(side="left", padx=5)

        self.upload_button = tk.Button(
            top_frame,
            text="Search by Image",
            command=self.upload_and_search_image,
        )
        self.upload_button.pack(side="left", padx=5)

        self.image_to_text_button = tk.Button(
            top_frame,
            text="Search Text by Image",
            command=self.image_to_text_async,
        )
        self.image_to_text_button.pack(side="left", padx=5)

        self.status_label = tk.Label(
            self.root, text="Initializing...", bd=1, relief="sunken", anchor="w"
        )
        self.status_label.pack(side="bottom", fill="x")

        # Result display area
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(
            self.root, orient="vertical", command=self.canvas.yview
        )
        self.scroll_frame = tk.Frame(self.canvas)

        # Handle mousewheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        # Interpolation controls
        interp_frame = tk.Frame(self.root)
        interp_frame.pack(fill="x", padx=10, pady=5)

        self.entry_a = tk.Entry(interp_frame, width=30)
        self.entry_a.bind("<KeyRelease>", self.reset_interpolation)
        self.entry_a.insert(0, "Enter text A")
        self.entry_a.bind("<FocusIn>", self.clear_entry)
        self.entry_a.pack(side="left", padx=5)

        self.entry_b = tk.Entry(interp_frame, width=30)
        self.entry_b.bind("<KeyRelease>", self.reset_interpolation)
        self.entry_b.insert(0, "Enter text B")
        self.entry_b.bind("<FocusIn>", self.clear_entry)
        self.entry_b.pack(side="left", padx=5)

        self.slider = tk.Scale(
            interp_frame,
            from_=0,
            to=1,
            resolution=0.05,
            orient="horizontal",
            label="Interpolation",
            command=self.on_slider_change,
        )
        self.slider.pack(side="left", fill="x", expand=True)

        # Zero-shot labeler section
        labeler_frame = tk.LabelFrame(
            self.root, text="Zero-Shot Labeler", padx=5, pady=5
        )
        labeler_frame.pack(fill="x", padx=10, pady=5)

        self.classes_entry = tk.Entry(labeler_frame, width=60)
        self.classes_entry.insert(0, "dog, cat, car, tree, person")
        self.classes_entry.pack(side="left", padx=5)

        self.select_image_button = tk.Button(
            labeler_frame,
            text="Select Image & Classify",
            command=self.classify_image_async,
        )
        self.select_image_button.pack(side="left", padx=5)

        self.labeler_result_frame = tk.Frame(self.root)
        self.labeler_result_frame.pack(fill="x", padx=10, pady=3)

        # Canvas and scrollbar setup
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def load_model_async(self) -> None:
        def worker() -> None:
            try:
                self.update_status("Loading model...")
                # Assuming high-level checkpoint exists or using a default model
                # You might need to specify the path to your checkpoint here
                checkpoint_path = Path(
                    "checkpoints/last.ckpt"
                )  # Update with actual path if needed
                if not checkpoint_path.exists():
                    self.update_status(
                        "Checkpoint not found at checkpoints/last.ckpt. Please update path."
                    )
                    return

                self.inferencer = ModelInferencer(checkpoint_path)
                image_index_cache_path, text_embeddings_cache_path = (
                    self._get_cache_paths(checkpoint_path)
                )

                # Default to indexing COCO val2017 for demo
                image_dir = Path("data/coco/val2017")
                image_paths = list(image_dir.glob("*.jpg"))
                if image_dir.exists():
                    self.update_status("Loading image index cache (COCO val2017)...")
                    self.image_index = self._load_or_build_image_index(
                        image_paths, image_index_cache_path
                    )
                    self.update_status("Image index ready.")
                else:
                    self.update_status(
                        f"Image directory {image_dir} not found. Ready for text-only searches."
                    )

                # Load COCO annotations for image to text retrieval
                ann_path = Path("data/coco/annotations/captions_val2017.json")
                if ann_path.exists():
                    with open(ann_path, "r", encoding="utf-8") as f:
                        ann_data = json.load(f)

                    self.annotations = [a["caption"] for a in ann_data["annotations"]]
                    self.update_status("Loading text embedding cache (COCO val2017)...")
                    self.text_embeddings = self._load_or_build_text_embeddings(
                        self.annotations, text_embeddings_cache_path
                    )
                    self.update_status(f"Loaded {len(self.annotations)} annotations.")
                else:
                    self.annotations = []
                    self.text_embeddings = None
                    self.update_status(f"Annotation file {ann_path} not found.")

                self.update_status("Ready to search.")

            except Exception as e:
                self.update_status(f"Error: {str(e)}")

        threading.Thread(target=worker, daemon=True).start()

    def search_async(self) -> None:
        query = self.search_entry.get()
        if not query or not self.inferencer or not self.image_index:
            return

        def worker() -> None:
            try:
                self.update_status(f"Searching for: '{query}'...")
                assert self.inferencer is not None and self.image_index is not None
                query_embedding = self.inferencer.embed_text(query)
                scores, indices, metadata = self.image_index.search(
                    query_embedding, k=10
                )

                self.root.after(0, lambda: self.display_results(metadata, scores))
                self.update_status("Search complete.")
            except Exception as e:
                self.update_status(f"Search Error: {str(e)}")

        threading.Thread(target=worker, daemon=True).start()

    def display_results(self, metadata: list, scores: list) -> None:
        # Clear previous results
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        for widget in self.labeler_result_frame.winfo_children():
            widget.destroy()

        self._photo_refs.clear()  # Keep reference to avoid GC
        for i, (m, score) in enumerate(zip(metadata[0], scores[0], strict=True)):
            img_path = str(m)
            if not img_path or not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)  # Keep reference to avoid GC

                frame = tk.Frame(
                    self.scroll_frame, bd=2, relief="groove", padx=5, pady=5
                )
                frame.grid(row=i // 3, column=i % 3, padx=10, pady=10)

                label = tk.Label(frame, image=photo)
                label.pack()

                tk.Label(frame, text=f"Score: {score:.4f}").pack()
                # Optional: Add button for Image-to-Image based on this result
                tk.Button(
                    frame,
                    text="Similar Images",
                    command=lambda p=img_path: self.search_similar_async(p),
                ).pack()
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    def search_similar_async(self, img_path: str) -> None:
        def worker() -> None:
            try:
                self.update_status(
                    f"Searching images similar to {os.path.basename(img_path)}..."
                )
                assert self.inferencer is not None and self.image_index is not None
                query_embedding = self.inferencer.embed_image(img_path)
                scores, _, metadata = self.image_index.search(query_embedding, k=10)
                self.root.after(0, lambda: self.display_results(metadata, scores))
                self.update_status("Image-to-Image search complete.")
            except Exception as e:
                self.update_status(f"Similar Search Error: {str(e)}")

        threading.Thread(target=worker, daemon=True).start()

    def image_to_text_async(self) -> None:
        if (
            not self.inferencer
            or self.text_embeddings is None
            or len(self.annotations) == 0
        ):
            self.update_status("Text embeddings not ready")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        def worker() -> None:
            try:
                self.update_status("Running image to text retrieval...")

                assert self.inferencer is not None
                assert self.text_embeddings is not None
                img_emb = self.inferencer.embed_image(file_path)
                text_embs = self.text_embeddings

                sims = torch.matmul(text_embs, img_emb.T).squeeze()
                sorted_idx = torch.argsort(sims, descending=True)
                results = [
                    (self.annotations[i], sims[i].item()) for i in sorted_idx[:10]
                ]

                self.root.after(
                    0, lambda: self.display_classification_results(file_path, results)
                )

                self.update_status("Image to text retrieval complete.")

            except Exception as e:
                self.update_status(f"Image to text error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def classify_image_async(self) -> None:
        if not self.inferencer:
            return

        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if not file_path:
                return

            classes_raw = self.classes_entry.get()
            class_names = [c.strip() for c in classes_raw.split(",") if c.strip()]
            if len(class_names) == 0:
                self.update_status("No classes provided.")
                return

            template = "a photo of a {}"
            class_prompts = [template.format(c) for c in class_names]

            def worker() -> None:
                try:
                    self.update_status("Running zero-shot classification...")
                    image_tensor = io.read_image(file_path)
                    predictions, logits = self.inferencer.classify_zero_shot(  # type: ignore
                        images=[image_tensor],
                        class_prompts=class_prompts,
                    )

                    probs = F.softmax(logits, dim=-1)[0]
                    sorted_indices = torch.argsort(probs, descending=True)
                    results = [
                        (class_names[i], probs[i].item()) for i in sorted_indices
                    ]

                    self.root.after(
                        0,
                        lambda: self.display_classification_results(file_path, results),
                    )

                    self.update_status("Classification complete.")

                except Exception as e:
                    self.update_status(f"Classification error: {e}")

            threading.Thread(target=worker, daemon=True).start()

        except Exception as e:
            self.update_status(f"File selection error: {e}")

    def display_classification_results(self, image_path: str, results: list) -> None:
        for widget in self.labeler_result_frame.winfo_children():
            widget.destroy()

        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        self._photo_refs.clear()

        img = Image.open(image_path)
        img.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(img)
        self._photo_refs.append(photo)

        img_label = tk.Label(self.labeler_result_frame, image=photo)
        img_label.pack()

        for class_name, prob in results:
            text = f"{class_name}: {prob:.4f}"
            tk.Label(self.labeler_result_frame, text=text).pack(anchor="w")

    def upload_and_search_image(self) -> None:
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if file_path:
                self.search_similar_async(file_path)
        except Exception as e:
            self.update_status(f"Upload Image Error: {str(e)}")

    def on_slider_change(self, val: str) -> None:
        if not self.inferencer or not self.image_index:
            return

        def worker() -> None:
            try:
                assert self.inferencer is not None
                assert self.image_index is not None

                alpha = float(val)
                if self._emb_a is None or self._emb_b is None:
                    self._emb_a = self.inferencer.embed_text(self.entry_a.get())
                    self._emb_b = self.inferencer.embed_text(self.entry_b.get())

                emb = (1 - alpha) * self._emb_a + alpha * self._emb_b

                scores, _, metadata = self.image_index.search(emb, k=10)
                self.root.after(0, lambda: self.display_results(metadata, scores))
            except Exception as e:
                self.update_status(f"Interpolation error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def reset_interpolation(self, event: tk.Event | None = None) -> None:
        self._emb_a = None
        self._emb_b = None
        self.slider.set(0)

    def update_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text))

    def clear_entry(self, event: tk.Event) -> None:
        widget = event.widget
        if isinstance(widget, tk.Entry):
            widget.delete(0, tk.END)
            widget.config(fg="black")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmbeddingExplorerApp(root)
    root.mainloop()

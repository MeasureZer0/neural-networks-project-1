import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any

from PIL import Image, ImageTk

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inferencer import ModelInferencer


class EmbeddingExplorerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Embedding Space Explorer")
        self.root.geometry("1000x800")

        self.inferencer: ModelInferencer | None = None
        self.image_index: Any | None = None
        self.results: list = []

        # interpolation state
        self._emb_a: Any | None = None
        self._emb_b: Any | None = None
        self._photo_refs: list = []  # prevent GC

        self.setup_ui()
        self.load_model_async()

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
            command=self.upload_and_search_image,  # type: ignore
        )
        self.upload_button.pack(side="left", padx=5)

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
                checkpoint_path = (
                    "checkpoints/last.ckpt"  # Update with actual path if needed
                )
                if not os.path.exists(checkpoint_path):
                    self.update_status(
                        "Checkpoint not found at checkpoints/last.ckpt. Please update path."
                    )
                    return

                self.inferencer = ModelInferencer(checkpoint_path)
                self.update_status(
                    "Model loaded. Building image index (COCO val2017)..."
                )

                # Default to indexing COCO val2017 for demo
                image_dir = Path("data/coco/val2017")
                image_paths = list(image_dir.glob("*.jpg"))
                if os.path.exists(image_dir):
                    assert self.inferencer is not None
                    self.image_index = self.inferencer.build_image_index(image_paths)
                    self.update_status("Ready to search.")
                else:
                    self.update_status(
                        f"Image directory {image_dir} not found. Ready for text-only searches."
                    )
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

        for i, (m, score) in enumerate(zip(metadata, scores[0], strict=True)):
            img_path = m.get("file_path")
            if not img_path or not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)

                frame = tk.Frame(
                    self.scroll_frame, bd=2, relief="groove", padx=5, pady=5
                )
                frame.grid(row=i // 3, column=i % 3, padx=10, pady=10)

                label = tk.Label(frame, image=photo)
                # label.image = photo # Keep reference to avoid garbage collection
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

    def upload_and_search_image(self) -> None:
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if file_path:
                self.search_similar_async(file_path)
        except Exception as e:
            self.update_status(f"Upload Image Error: {str(e)}")

    def update_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text))


if __name__ == "__main__":
    root = tk.Tk()
    app = EmbeddingExplorerApp(root)
    root.mainloop()

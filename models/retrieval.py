import pickle
from pathlib import Path

import faiss
import numpy as np
import torch


class EmbeddingIndex:
    def __init__(self, embedding_dim: int) -> None:
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def add(self, embeddings: torch.Tensor, metadata: list | None = None) -> None:
        embeddings = embeddings.cpu().float().numpy()
        self.index.add(embeddings)
        if metadata:
            self.metadata.extend(metadata)

    def search(
        self, queries: torch.Tensor, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray, list]:
        queries = queries.cpu().float().numpy()
        scores, indicies = self.index.search(queries, k)
        meta = []
        if self.metadata:
            meta = [[self.metadata[i] for i in row] for row in indicies]
        return scores, indicies, meta

    def save(self, path: str | Path) -> None:
        faiss.write_index(self.index, str(path))

        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved: {path}  ({self.index.ntotal} vectors)")

    def load(self, path: str | Path) -> None:
        self.index = faiss.read_index(str(path))

        with open(path.with_suffix(".pkl"), "rb") as f:
            self.metadata = pickle.load(f)

        print(f"Index loaded: {path}  ({self.index.ntotal} vectors)")

    def __len__(self) -> int:
        return self.index.ntotal

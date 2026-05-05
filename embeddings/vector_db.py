# embeddings/vector_db.py — FAISS-backed vector store with persistence

import faiss
import numpy as np
import os
from config import EMBED_DIM, FAISS_INDEX_PATH


class VectorDB:
    """
    Wraps a flat L2 FAISS index with save/load so the index
    survives across sessions.
    """

    def __init__(self, dim: int = EMBED_DIM, index_path: str = FAISS_INDEX_PATH):
        self.dim        = dim
        self.index_path = index_path
        self.index      = faiss.IndexFlatIP(dim)   # Inner product (cosine when normalised)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self):
        faiss.write_index(self.index, self.index_path)
        print(f"[vector_db] Index saved to {self.index_path}  ({self.index.ntotal} vectors)")

    def load(self) -> bool:
        """Load index from disk. Returns True if successful, False if not found."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"[vector_db] Index loaded from {self.index_path}  ({self.index.ntotal} vectors)")
            return True
        return False

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray):
        """Add one or more L2-normalised vectors."""
        vecs = np.array(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]
        self.index.add(vecs)

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int = 3) -> list[int]:
        """
        Return the indices of the top-k most similar stored vectors.

        Parameters
        ----------
        query : np.ndarray  shape (dim,) or (1, dim)
        k     : number of neighbours

        Returns
        -------
        list of int indices (length min(k, ntotal))
        """
        q = np.array(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[np.newaxis, :]

        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        _, indices = self.index.search(q, k)
        return indices[0].tolist()

    @property
    def size(self) -> int:
        return self.index.ntotal
# embeddings/embedder.py — sentence-transformer wrapper

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

# Loaded once at module import; shared across the app
_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading sentence transformer: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns
    -------
    np.ndarray of shape (len(texts), EMBED_DIM), dtype float32
    """
    if isinstance(texts, str):
        texts = [texts]
    embedder = get_embedder()
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)
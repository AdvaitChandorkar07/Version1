# utils/umap_reduce.py — UMAP dimensionality reduction

import numpy as np
import umap


def umap_reduce(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Reduce high-dimensional embeddings with UMAP.

    Parameters
    ----------
    embeddings   : (N, D) float32 array — raw sentence-transformer embeddings
    n_components : target dimensionality (keep ≥ 5 so PCA still has signal)

    Returns
    -------
    np.ndarray of shape (N, n_components)

    Notes
    -----
    UMAP needs at least ~10–15 samples to find meaningful clusters.
    We enforce this with an assertion so failures are obvious.
    """
    n = embeddings.shape[0]
    assert n >= 10, (
        f"UMAP needs at least 10 samples, got {n}. "
        "Add more onboarding questions or skip UMAP."
    )

    # n_neighbors must be < n_samples
    n_neighbors = min(5, n - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced.astype(np.float32)
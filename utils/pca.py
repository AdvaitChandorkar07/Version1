# utils/pca.py — PCA applied to raw embeddings (not post-UMAP)
#
# Key design decision:
#   PCA is run on the RAW sentence-transformer embeddings (384-dim),
#   NOT on UMAP-reduced output. This preserves the geometric properties
#   needed for residual-stream steering vectors.
#
#   UMAP output is used separately for cluster validation only.

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA


def fit_pca(embeddings: np.ndarray, n_components: int = 32) -> SklearnPCA:
    """
    Fit PCA on raw embeddings.

    Parameters
    ----------
    embeddings   : (N, D) float32 — raw sentence-transformer embeddings
    n_components : how many principal components to keep

    Returns
    -------
    Fitted sklearn PCA object
    """
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = SklearnPCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)
    print(f"[pca] Fitted PCA: {n_components} components, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return pca


def derive_steering_vectors(
    pca: SklearnPCA,
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Project each embedding onto the PCA directions to produce steering vectors.

    The result has the same dimensionality as the original embeddings (384-dim)
    so it can be added directly to the LLM's residual stream without a
    projection step.

    Parameters
    ----------
    pca        : fitted PCA object
    embeddings : (N, D) raw embeddings

    Returns
    -------
    np.ndarray of shape (N, D) — one steering vector per preference sample
    """
    # Project into PCA space then reconstruct back to original space.
    # This gives us the "preference direction" living in the original geometry.
    projected    = pca.transform(embeddings)          # (N, n_components)
    reconstructed = pca.inverse_transform(projected)  # (N, D)

    # L2-normalise each steering vector
    norms = np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8
    steering_vecs = reconstructed / norms

    return steering_vecs.astype(np.float32)
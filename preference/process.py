# preference/process.py — build semantic + steering vectors from raw preferences
#
# Pipeline:
#   raw text answers
#     → sentence-transformer embeddings  (384-dim)
#     → UMAP reduction                   (used only for cluster validation / debug)
#     → PCA on raw embeddings            (steering vectors, still 384-dim)
#     → persist both vector sets

import os
import numpy as np
from embeddings.embedder import embed
from utils.umap_reduce import umap_reduce
from utils.pca import fit_pca, derive_steering_vectors
from config import SEMANTIC_VECS_PATH, STEERING_VECS_PATH


def build_preference_vectors(
    answers: list[str],
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw preference answers into semantic + steering vectors.

    Parameters
    ----------
    answers : list of str — one answer per onboarding question
    force   : re-compute even if saved files exist

    Returns
    -------
    semantic_vecs  : (N, 384)  — for FAISS ANN search
    steering_vecs  : (N, 384)  — for residual-stream injection
    """
    if not force and _saved_vectors_exist():
        semantic_vecs = np.load(SEMANTIC_VECS_PATH)
        steering_vecs = np.load(STEERING_VECS_PATH)
        print(f"[process] Loaded saved vectors  "
              f"(semantic: {semantic_vecs.shape}, steering: {steering_vecs.shape})")
        return semantic_vecs, steering_vecs

    print("[process] Embedding preference answers...")
    semantic_vecs = embed(answers)           # (N, 384), already L2-normalised

    print("[process] Running UMAP for cluster validation...")
    try:
        umap_vecs = umap_reduce(semantic_vecs)
        _log_cluster_info(umap_vecs)
    except AssertionError as e:
        print(f"[process] UMAP skipped: {e}")

    print("[process] Deriving steering vectors via PCA...")
    pca = fit_pca(semantic_vecs, n_components=min(32, len(answers)))
    steering_vecs = derive_steering_vectors(pca, semantic_vecs)

    np.save(SEMANTIC_VECS_PATH, semantic_vecs)
    np.save(STEERING_VECS_PATH, steering_vecs)
    print(f"[process] Vectors saved:  {SEMANTIC_VECS_PATH}, {STEERING_VECS_PATH}")

    return semantic_vecs, steering_vecs


# ── helpers ───────────────────────────────────────────────────────────────────

def _saved_vectors_exist() -> bool:
    return os.path.exists(SEMANTIC_VECS_PATH) and os.path.exists(STEERING_VECS_PATH)


def _log_cluster_info(umap_vecs: np.ndarray):
    """Simple variance check — high variance across dims suggests real clusters."""
    var = umap_vecs.var(axis=0)
    print(f"[process] UMAP component variances (first 5): {var[:5].round(4)}")
    if var.mean() < 0.01:
        print("[process] ⚠️  Low UMAP variance — preferences may be too uniform.")
    else:
        print("[process] ✅ UMAP shows meaningful spread in preference space.")
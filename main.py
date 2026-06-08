import sys
import numpy as np
import torch

from config import TOP_K, MAX_NEW_TOKENS, STEERING_ALPHA
from preference.collect import collect_preferences
from preference.process import build_preference_vectors
from embeddings.embedder import embed
from embeddings.vector_db import VectorDB
from models.llama_loader import load_model
from models.steering import generate_with_steering
from rag.retriever import RAGRetriever
from rag.retriever import RAGRetriever

from config import (
    TOP_K,
    MAX_NEW_TOKENS,
    STEERING_ALPHA,
    RAG_TOP_K,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def aggregate_steering_vectors(
    steering_vecs: np.ndarray,
    indices: list[int],
) -> torch.Tensor:
    """
    Average the top-k retrieved steering vectors into one composite vector.

    Parameters
    ----------
    steering_vecs : (N, D) numpy array of all stored steering vectors
    indices       : list of int indices returned by FAISS search

    Returns
    -------
    torch.Tensor of shape (D,)
    """
    selected = steering_vecs[indices]           # (k, D)
    mean_vec = selected.mean(axis=0)            # (D,)
    # Re-normalise
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
    return torch.tensor(mean_vec, dtype=torch.float32)


def build_prompt(query: str,contexts: list[str],) -> str:
    """
    Wrap the user query in a simple instruction-style prompt.
    Adjust this template if your LLaMA variant uses a different chat format.
    """
    context_block = "\n\n".join(contexts)

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Use retrieved information when relevant.\n\n"
        f"{context_block}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{query}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pre-Appointment App  —  Local LLaMA + Representation Editing")
    print("=" * 60)

    # ── Step 1: preferences ───────────────────────────────────────────────────
    # Pass force=True to re-run onboarding: collect_preferences(force=True)
    answers = collect_preferences()

    # ── Step 2: vectors ───────────────────────────────────────────────────────
    # Pass force=True to recompute: build_preference_vectors(answers, force=True)
    semantic_vecs, steering_vecs = build_preference_vectors(answers)

    # ── Step 3: FAISS index ───────────────────────────────────────────────────
    db = VectorDB()
    if not db.load():                        # try loading saved index
        print("[main] Building FAISS index from scratch...")
        db.add(semantic_vecs)
        db.save()
    print(f"[main] FAISS index ready — {db.size} vectors.\n")

    # ── Step 4: load LLaMA ────────────────────────────────────────────────────
    model, tokenizer = load_model()
    rag = RAGRetriever()

    # ── Step 5: chat loop ─────────────────────────────────────────────────────
    print("\n=== Chat ready. Type 'quit' or 'exit' to stop. ===\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Interrupted — goodbye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("[main] Goodbye.")
            break

        # Embed the query
        q_emb = embed([query])[0]                           # (384,)

        # Retrieve top-k most relevant preference contexts
        indices = db.search(q_emb, k=TOP_K)

        if not indices:
            print("[main] ⚠️  No preference vectors found — generating without steering.")
            steering_vec = torch.zeros(semantic_vecs.shape[1])
        else:
            steering_vec = aggregate_steering_vectors(steering_vecs, indices)

        # Build prompt and generate
        contexts = rag.retrieve(query, k=RAG_TOP_K)
        prompt = build_prompt(query, contexts)

        try:
            response = generate_with_steering(
                model,
                tokenizer,
                prompt,
                steering_vec,
                max_new_tokens=MAX_NEW_TOKENS,
                alpha=STEERING_ALPHA,
            )
            print(f"\nAssistant: {response}\n")

        except Exception as e:
            print(f"[main] Generation error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
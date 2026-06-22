import sys
import numpy as np
import torch

from config import (
    TOP_K,
    MAX_NEW_TOKENS,
    STEERING_ALPHA,
    RAG_TOP_K,
)

from database.db import initialize_database
from database.user_repository import UserRepository

from models.user import User
from models.llama_loader import load_model
from models.steering import generate_with_steering

from preference.collect import collect_preferences
from preference.process import build_preference_vectors

from embeddings.embedder import embed
from embeddings.vector_db import VectorDB

from embeddings.storage import (
    save_semantic,
    save_steering,
    load_semantic,
    load_steering,
)

from rag.retriever import RAGRetriever

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

    # ── Step 1: Login / Signup ───────────────────────────────────────────────
    user_id = input("User ID: ").strip()
    user = UserRepository.get_user(user_id)

    if user is None:

        print("\nNew user detected.\n")

        name = input("Name: ").strip()

        user = User(
            user_id=user_id,
            name=name
        )

        UserRepository.create_user(user)

        answers = collect_preferences()

        semantic_vecs, steering_vecs = build_preference_vectors(
            answers
        )

        semantic_path = save_semantic(
            user.user_id,
            semantic_vecs
        )

        steering_path = save_steering(
            user.user_id,
            steering_vecs
        )

        UserRepository.update_vectors(
            user.user_id,
            semantic_path,
            steering_path
        )

    # EXISTING USER
    else:

        print(f"\nWelcome back {user.name}\n")

        semantic_vecs = load_semantic(user.semantic_path)

        steering_vecs = load_steering(user.steering_path)

        if len(semantic_vecs.shape) == 1:
            semantic_vecs = semantic_vecs.reshape(1, -1)

        if len(steering_vecs.shape) == 1:
            steering_vecs = steering_vecs.reshape(1, -1)

    # ── Step 3: FAISS index ───────────────────────────────────────────────────
    db = VectorDB()

    print("[main] Building user-specific FAISS index...")

    db.add(semantic_vecs)

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
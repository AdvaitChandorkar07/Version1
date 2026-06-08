import json

from embeddings.embedder import embed
from embeddings.vector_db import VectorDB

from config import (
    RAG_INDEX_PATH,
    RAG_CHUNKS_PATH,
)

from rag.chunking import chunk_text

def build_rag_index(files):

    chunks = []

    for file in files:

        with open(file, "r", encoding="utf8") as f:

            text = f.read()

        chunks.extend(
            chunk_text(text)
        )

    embeddings = embed(chunks)

    db = VectorDB()

    db.add(embeddings)

    db.save(RAG_INDEX_PATH)

    with open(RAG_CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nBuilt RAG index with {len(chunks)} chunks")
import json

from embeddings.embedder import embed
from embeddings.vector_db import VectorDB

from config import (
    RAG_INDEX_PATH,
    RAG_CHUNKS_PATH,
)


class RAGRetriever:

    def __init__(self):

        self.db = VectorDB()

        self.db.load(RAG_INDEX_PATH)

        with open(RAG_CHUNKS_PATH) as f:
            self.chunks = json.load(f)

    def retrieve(self, query, k=3):

        q_emb = embed([query])[0]

        indices = self.db.search(q_emb, k)

        return [
            self.chunks[i]
            for i in indices
        ]
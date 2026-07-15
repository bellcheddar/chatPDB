#!/usr/bin/env python3
"""
retrieve.py — lightweight retrieval wrapper over the chatPDB ChromaDB store.

Ported from chem_sage/rag/retrieve.py — identical pattern, chatPDB collection name.

Usage (Python API):
    from rag.retrieve import Retriever, format_context
    r = Retriever()
    chunks = r.retrieve("what resolution was PDB entry 1CRN solved at?", n=5)
    print(format_context(chunks))
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CHROMA_STORE = ".chroma"
COLLECTION = "chatpdb"


class Retriever:
    def __init__(self, store: str = CHROMA_STORE, collection: str = COLLECTION, embed_model: str = EMBED_MODEL):
        self._embedder = SentenceTransformer(embed_model)
        client = chromadb.PersistentClient(
            path=store,
            settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False),
        )
        self._collection = client.get_collection(collection)

    def retrieve(self, query: str, n: int = 5) -> list[dict]:
        """Return the top-n corpus chunks closest to query.

        Each element: {"text": str, "source": str, "score": float}
        score is cosine similarity (0-1; higher = more relevant).
        """
        vec = self._embedder.encode([query]).tolist()
        res = self._collection.query(
            query_embeddings=vec,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunks.append({
                "text": text,
                "source": Path(meta.get("source", "unknown")).name,
                "score": round(1.0 - dist, 3),  # cosine distance -> similarity
            })
        return chunks


def format_context(chunks: list[dict], max_chars: int = 3000) -> str:
    """Render retrieved chunks as a context block for prompt injection.

    Stops adding chunks once max_chars is reached so the prompt stays
    within a sensible token budget.
    """
    parts: list[str] = []
    total = 0
    for c in chunks:
        block = f"[Source: {c['source']} | relevance {c['score']:.2f}]\n{c['text']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)

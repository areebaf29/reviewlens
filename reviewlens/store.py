"""Vector stores behind one interface: NumpyStore (zero-dependency, file-persisted) and
ChromaStore (a real vector database, persistent on disk)."""
import json
import os
from typing import Protocol

import numpy as np

from .ingest import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def query(self, vector: np.ndarray, k: int, min_rating: int | None = None, max_rating: int | None = None) -> list[tuple[Chunk, float]]: ...
    def all_vectors(self) -> tuple[list[Chunk], np.ndarray]: ...
    def count(self) -> int: ...


def _ok(c: Chunk, lo, hi) -> bool:
    return (lo is None or c.rating >= lo) and (hi is None or c.rating <= hi)


class NumpyStore:
    def __init__(self, path: str | None = None):
        self.path, self.chunks, self.vecs = path, [], np.zeros((0, 0))
        if path and os.path.exists(path + ".npz"):
            self.vecs = np.load(path + ".npz")["v"]
            self.chunks = [Chunk(**d) for d in json.load(open(path + ".json", encoding="utf-8"))]

    def add(self, chunks, vectors):
        self.chunks += list(chunks)
        self.vecs = vectors if not self.vecs.size else np.vstack([self.vecs, vectors])
        if self.path:
            np.savez_compressed(self.path + ".npz", v=self.vecs)
            json.dump([c.__dict__ for c in self.chunks], open(self.path + ".json", "w", encoding="utf-8"))

    def query(self, vector, k, min_rating=None, max_rating=None):
        if not self.chunks:
            return []
        sims = self.vecs @ vector
        out = []
        for i in np.argsort(-sims, kind="stable"):
            if _ok(self.chunks[i], min_rating, max_rating):
                out.append((self.chunks[i], float(sims[i])))
                if len(out) == k:
                    break
        return out

    def all_vectors(self):
        return self.chunks, self.vecs

    def count(self):
        return len(self.chunks)


class ChromaStore:
    def __init__(self, path: str | None = None, collection: str = "reviews"):
        import chromadb  # optional dependency
        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        try:
            client.delete_collection(collection)
        except Exception:
            pass
        self.col = client.create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks, vectors):
        for i in range(0, len(chunks), 500):
            cs = chunks[i:i + 500]
            self.col.add(ids=[c.id for c in cs], embeddings=vectors[i:i + 500].tolist(), documents=[c.text for c in cs],
                         metadatas=[{"review_id": c.review_id, "rating": c.rating} for c in cs])

    def query(self, vector, k, min_rating=None, max_rating=None):
        where = None
        conds = ([{"rating": {"$gte": min_rating}}] if min_rating is not None else []) + \
                ([{"rating": {"$lte": max_rating}}] if max_rating is not None else [])
        if conds:
            where = conds[0] if len(conds) == 1 else {"$and": conds}
        r = self.col.query(query_embeddings=[vector.tolist()], n_results=min(k, max(1, self.count())), where=where)
        return [(Chunk(i, m["review_id"], d, int(m["rating"])), 1.0 - dist)  # cosine distance -> similarity
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def all_vectors(self):
        r = self.col.get(include=["embeddings", "documents", "metadatas"])
        chunks = [Chunk(i, m["review_id"], d, int(m["rating"])) for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])]
        return chunks, np.asarray(r["embeddings"])

    def count(self):
        return self.col.count()


def get_store(name: str = "numpy", path: str | None = None) -> VectorStore:
    return ChromaStore(path) if name == "chroma" else NumpyStore(path)

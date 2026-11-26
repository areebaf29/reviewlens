"""Hybrid retrieval: dense vectors from the vector store + BM25, fused with Reciprocal Rank Fusion."""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .ingest import Chunk
from .store import VectorStore
from .text import tokenize


@dataclass
class Retrieved:
    chunk: Chunk
    score: float       # fused RRF score (ranking only)
    dense: float       # cosine similarity to the question
    coverage: float    # share of the question's IDF weight present in the chunk


class BM25:
    def __init__(self, chunks: list[Chunk], k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.toks = [tokenize(c.text) for c in chunks]
        self.tf = [Counter(t) for t in self.toks]
        self.avgdl = sum(map(len, self.toks)) / max(1, len(chunks)) or 1.0
        df = Counter(t for ts in self.toks for t in set(ts))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.max_idf = max(self.idf.values(), default=1.0)

    def scores(self, query: str) -> np.ndarray:
        q = [t for t in tokenize(query) if t in self.idf]
        s = np.zeros(len(self.chunks))
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def coverage(self, query: str, i: int) -> float:
        q = set(tokenize(query))
        if not q:
            return 0.0
        total = sum(self.idf.get(t, 2 * self.max_idf) for t in q)  # unseen words count as very informative
        return sum(self.idf[t] for t in q if t in self.idf and t in self.tf[i]) / total


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, rrf_k: int = 60, dense_weight: float = 1.0, pool: int = 50):
        self.store, self.embedder, self.rrf_k, self.dense_weight, self.pool = store, embedder, rrf_k, dense_weight, pool
        chunks, _ = store.all_vectors()
        self.bm25 = BM25(chunks)
        self._pos = {c.id: i for i, c in enumerate(chunks)}

    def search(self, query: str, k: int = 5, mode: str = "hybrid", min_rating=None, max_rating=None) -> list[Retrieved]:
        if not query.strip():
            return []
        qv = self.embedder.encode([query])[0]
        dense = self.store.query(qv, self.pool, min_rating, max_rating)
        dsim = {c.id: s for c, s in dense}
        bm = self.bm25.scores(query)
        lex = [i for i in np.argsort(-bm, kind="stable")[: self.pool] if bm[i] > 0
               and (min_rating is None or self.bm25.chunks[i].rating >= min_rating)
               and (max_rating is None or self.bm25.chunks[i].rating <= max_rating)]
        score: dict[str, float] = {}
        if mode in ("hybrid", "dense"):
            for r, (c, _) in enumerate(dense, 1):
                score[c.id] = score.get(c.id, 0) + self.dense_weight / (self.rrf_k + r)
        if mode in ("hybrid", "bm25"):
            for r, i in enumerate(lex, 1):
                cid = self.bm25.chunks[i].id
                score[cid] = score.get(cid, 0) + 1 / (self.rrf_k + r)
        out = []
        for cid, sc in sorted(score.items(), key=lambda kv: -kv[1])[:k]:
            i = self._pos[cid]
            sim = dsim.get(cid)
            if sim is None:  # lexical-only hit: compute its dense similarity directly
                sim = float(self.store.all_vectors()[1][i] @ qv) if mode != "bm25" else 0.0
            out.append(Retrieved(self.bm25.chunks[i], sc, sim, self.bm25.coverage(query, i)))
        return out

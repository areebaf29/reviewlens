"""Theme mining: k-means over the chunk embeddings in the vector store, labelled by distinctive terms."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .text import tokenize


@dataclass
class Topic:
    label: str
    size: int
    avg_rating: float
    chunk_ids: list[str]


def _kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
        C.append(X[rng.choice(len(X), p=d / d.sum() if d.sum() > 0 else None)])
    C = np.array(C)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def mine_topics(store, k=6, max_rating=3) -> list[Topic]:
    chunks, V = store.all_vectors()
    idx = [i for i, c in enumerate(chunks) if c.rating <= max_rating and c.rating > 0]
    if not idx:
        return []
    k = min(k, len(idx))
    lab = _kmeans(V[idx], k)
    df = Counter(t for c in chunks for t in set(tokenize(c.text)))
    out = []
    for j in range(k):
        mem = [chunks[idx[i]] for i in np.where(lab == j)[0]]
        if not mem:
            continue
        cnt = Counter(t for c in mem for t in set(tokenize(c.text)))
        score = lambda t: cnt[t] / len(mem) * (len(chunks) / df[t]) ** 0.5
        out.append(Topic(", ".join(sorted(cnt, key=lambda t: -score(t))[:3]), len(mem),
                         sum(c.rating for c in mem) / len(mem), [c.id for c in mem]))
    return sorted(out, key=lambda t: -t.size)

"""Embedders. `SentenceTransformerEmbedder` is the real neural one (MiniLM by default).
`LSAEmbedder` is a dependency-free offline fallback (TF-IDF + SVD) fit on your corpus."""
import math
from collections import Counter
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    def fit(self, texts: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...  # L2-normalised rows


def _norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.name, self._m = model, SentenceTransformer(model)

    def fit(self, texts):  # pretrained, nothing to fit
        pass

    def encode(self, texts):
        return np.asarray(self._m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=64))


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 32):
        self.dims = dims

    def _vec(self, text):
        v = np.zeros(len(self.vocab))
        for t, c in Counter(tokenize(text)).items():
            j = self.vocab.get(t)
            if j is not None:
                v[j] = (1 + math.log(c)) * self.idf[t]
        return v

    def fit(self, texts):
        toks = [set(tokenize(t)) for t in texts]
        df = Counter(t for ts in toks for t in ts)
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(texts)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        X = np.vstack([self._vec(t) for t in texts])
        k = max(1, min(self.dims, min(X.shape) - 1))
        self.proj = np.linalg.svd(X, full_matrices=False)[2][:k].T

    def encode(self, texts):
        return _norm(np.vstack([self._vec(t) for t in texts]) @ self.proj)


def get_embedder(name: str = "minilm") -> Embedder:
    if name == "lsa":
        return LSAEmbedder()
    if name == "minilm":
        return SentenceTransformerEmbedder()
    return SentenceTransformerEmbedder(name)  # any sentence-transformers model id

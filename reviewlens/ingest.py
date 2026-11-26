"""Ingestion + chunking. Reads CSV (id,rating,text[,title,date]) or JSONL and splits long
reviews into overlapping sentence windows so each chunk stays focused and citable."""
import csv
import json
from dataclasses import dataclass, field

from .text import sentences


@dataclass(frozen=True)
class Chunk:
    id: str            # "<review_id>#<n>"
    review_id: str
    text: str
    rating: int = 0
    meta: dict = field(default_factory=dict, compare=False)


def read_records(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        missing = {"text"} - set(rd.fieldnames or [])
        if missing:
            raise ValueError("input is missing required column: text")
        return list(rd)


def chunk_text(text: str, max_words: int = 60, overlap: int = 1) -> list[str]:
    """Sentence-window chunking: pack whole sentences up to max_words, carry `overlap` sentences over."""
    sents = sentences(text)
    if not sents:
        return []
    if len(text.split()) <= max_words:
        return [text.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


def ingest(path: str, max_words: int = 60, overlap: int = 1) -> list[Chunk]:
    out, seen = [], set()
    for i, row in enumerate(read_records(path), 1):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        rid = str(row.get("id") or i).strip()
        if rid in seen:
            raise ValueError(f"duplicate review id: {rid}")
        seen.add(rid)
        try:
            rating = max(0, min(5, int(float(row.get("rating") or 0))))
        except (TypeError, ValueError):
            raise ValueError(f"review {rid}: rating must be a number") from None
        title = (row.get("title") or "").strip()
        body = f"{title}. {text}" if title else text
        for n, c in enumerate(chunk_text(body, max_words, overlap)):
            out.append(Chunk(f"{rid}#{n}" if n else rid, rid, c, rating, {"date": row.get("date", "")}))
    if not out:
        raise ValueError("no reviews found")
    return out

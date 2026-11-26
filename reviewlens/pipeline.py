"""ingest -> chunk -> embed -> vector store -> hybrid retrieve -> relevance gate -> generate -> citation check."""
import re
from dataclasses import dataclass, field

import numpy as np

from .embed import Embedder, get_embedder
from .generate import ExtractiveLLM, LLM, get_llm
from .ingest import Chunk, ingest
from .retrieve import HybridRetriever, Retrieved
from .store import VectorStore, get_store
from .verify import atomic_claims, split_claims

NO_EVIDENCE = "Not enough evidence in the reviews to answer that."
_cite = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool = False
    citations: list[str] = field(default_factory=list)   # valid ids the answer cites
    sources: list[Retrieved] = field(default_factory=list)  # context actually given to the generator
    generator: str = ""
    generative: bool = False
    note: str = ""      # e.g. why we fell back to extractive output


class RAGPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder, llm: LLM | None = None,
                 min_dense: float = 0.30, min_coverage: float = 0.4, k: int = 5, rel_cutoff: float = 0.8, verifier=None, corpus_label: str = ""):
        self.store, self.embedder, self.llm = store, embedder, llm or ExtractiveLLM()
        self.retriever = HybridRetriever(store, embedder)
        self.corpus_label = corpus_label  # shown in the UI, e.g. 'Reviews of many different laptops, mixed'
        self.verifier = verifier  # optional NLIVerifier: checks each generated claim against its cited chunks
        self.min_dense, self.min_coverage, self.k, self.rel_cutoff = min_dense, min_coverage, k, rel_cutoff

    def relevant(self, hits: list[Retrieved]) -> list[Retrieved]:
        """Relevance gate. Keep a chunk if it is close in embedding space AND shares some of the question's
        key terms, or if it covers most of them. Then drop chunks far weaker than the best one, so a
        stray 'runs smoothly' does not ride along with real battery answers."""
        keep = [h for h in hits if h.dense >= self.min_dense and h.coverage >= self.min_coverage * 0.5
                or h.coverage >= max(self.min_coverage, 0.6)]
        if not keep:
            return []
        top = max(h.dense for h in keep)
        return [h for h in keep if h.dense >= self.rel_cutoff * top]

    def ask(self, question: str, k: int | None = None, **filters) -> Answer:
        k = k or self.k
        hits, seen = [], set()
        for h in self.relevant(self.retriever.search(question, k=k * 4, **filters)):
            key = " ".join(h.chunk.text.lower().split())
            if key not in seen:  # identical text twice adds no evidence
                seen.add(key)
                hits.append(h)
        hits = hits[:k]
        if not hits:
            return Answer(question, NO_EVIDENCE, abstained=True)
        chunks = [h.chunk for h in hits]
        valid = {c.id for c in chunks}
        note, llm = "", self.llm
        try:
            text = llm.generate(question, chunks)
        except Exception as e:  # network/model failure: degrade to extractive, say so
            text, llm, note = ExtractiveLLM().generate(question, chunks), ExtractiveLLM(), f"generator failed ({type(e).__name__}); extractive fallback"
        if llm.generative:  # a model that hedges inside an answer: drop the hedge, abstain only if nothing else is left
            stripped = re.sub(r"insufficient evidence\.?", "", text, flags=re.I).strip()
            if not re.search(r"[A-Za-z]{3,}", _cite.sub("", stripped)):
                text = "INSUFFICIENT EVIDENCE"
            else:
                text = stripped
        if "INSUFFICIENT EVIDENCE" in text.upper():
            return Answer(question, NO_EVIDENCE, abstained=True, sources=hits, generator=llm.name, generative=llm.generative)
        cited = [i.strip() for m in _cite.findall(text) for i in m.split(",")]
        good = [i for i in cited if i in valid]
        prose = len(re.findall(r"[A-Za-z]{2,}", _cite.sub(" ", text)))
        if llm.generative and (not good or prose < 4):  # ignored the citation/prose contract -> do not present it as grounded
            note = note or "generator gave no usable cited answer; showing extractive evidence instead"
            text, llm = ExtractiveLLM().generate(question, chunks), ExtractiveLLM()
            good = [c.id for c in chunks]
        elif llm.generative:  # strip citations the model invented
            text = _cite.sub(lambda m: m.group(0) if all(i.strip() in valid for i in m.group(1).split(",")) else "", text)
        if llm.generative and self.verifier is not None:
            by_id = {c.id: c.text for c in chunks}
            kept, dropped = [], 0
            for claim, ids in atomic_claims(text):
                prem = " ".join(by_id[i] for i in ids if i in by_id) or " ".join(by_id.values())
                if self.verifier.supported(prem, claim):
                    kept.append(claim + "".join(f" [{i}]" for i in dict.fromkeys(ids) if i in by_id))
                else:
                    dropped += 1
            if not kept:
                note = "no generated claim passed the support check; showing extractive evidence instead"
                text, llm = ExtractiveLLM().generate(question, chunks), ExtractiveLLM()
                good = [c.id for c in chunks]
            else:
                text = " ".join(kept)
                good = [i for _, ids in atomic_claims(text) for i in ids if i in by_id]
                note = note or (f"{dropped} generated statement(s) removed: not supported by their cited reviews" if dropped else "")
        return Answer(question, text, False, list(dict.fromkeys(good)), hits, llm.name, llm.generative, note)


def build_pipeline(path: str, embedder: str = "minilm", store: str = "numpy", llm: str | None = None,
                   store_path: str | None = None, max_words: int = 60, verify: bool | None = None, corpus_label: str = "", **kw) -> RAGPipeline:
    chunks = ingest(path, max_words=max_words)
    emb = get_embedder(embedder)
    emb.fit([c.text for c in chunks])
    vs = get_store(store, store_path)
    if vs.count() == 0:
        vs.add(chunks, emb.encode([c.text for c in chunks]))
    model = get_llm(llm)
    if verify is None:
        verify = model.generative
    verifier = None
    if verify:
        from .verify import NLIVerifier
        verifier = NLIVerifier()
    return RAGPipeline(vs, emb, model, verifier=verifier, corpus_label=corpus_label, min_dense=kw.pop("min_dense", 0.40 if embedder != "lsa" else 0.5), **kw)

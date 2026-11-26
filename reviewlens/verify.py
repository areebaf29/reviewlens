"""Claim support check. Citation-id validation only proves a cited chunk EXISTS. This module asks a
natural-language-inference model whether the cited chunks actually ENTAIL each generated sentence.
It is a model, not an oracle: it can wrongly keep or drop a sentence."""
import re

_cite = re.compile(r"\[([^\[\]]+)\]")


def split_claims(text: str) -> list[tuple[str, list[str]]]:
    """Split generated text into (sentence, cited ids). Trailing '[id][id]' groups attach to the sentence before them."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: list[tuple[str, list[str]]] = []
    for p in parts:
        ids = [i.strip() for m in _cite.findall(p) for i in m.split(",")]
        prose = _cite.sub("", p).strip()
        if not re.search(r"[A-Za-z]{2,}", prose):  # cite-only fragment
            if claims:
                claims[-1][1].extend(ids)
            continue
        prose = re.sub(r"\s*\b(as per|according to|as stated in|as mentioned in|per)\b\s*(?=[,.;]|$)", "", prose)  # dangling attribution left by removed ids
        prose = re.sub(r"\s+([,.;])", r"\1", prose).strip()
        claims.append((prose, ids))
    return claims


def atomic_claims(text: str) -> list[tuple[str, list[str]]]:
    """Like split_claims, but also splits compound sentences ('X, and Y', 'X; Y', 'X, with Y') so each
    fact is checked on its own; NLI models call long multi-fact sentences 'neutral'."""
    out = []
    for prose, ids in split_claims(text):
        for part in re.split(r",\s+(?:and|but|with|while)\s+|;\s+", prose):
            part = part.strip(" ,.")
            if re.search(r"[A-Za-z]{3,}", part):
                out.append((part[0].upper() + part[1:] + ".", ids))
    return out


class NLIVerifier:
    def __init__(self, model="cross-encoder/nli-deberta-v3-xsmall", threshold=0.5):
        from sentence_transformers import CrossEncoder  # optional dependency
        self.m, self.threshold, self.name = CrossEncoder(model), threshold, model
        self.entail = next(i for i, l in self.m.config.id2label.items() if l.lower() == "entailment")

    def supported(self, premise: str, claim: str) -> bool:
        return float(self.m.predict([(premise, claim)], apply_softmax=True)[0][self.entail]) >= self.threshold

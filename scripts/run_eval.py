"""Evaluate retrieval, abstention and generation on REAL laptop reviews (SemEval-2014 ABSA).
Run scripts/fetch_semeval.py first.  Usage: python scripts/run_eval.py [--llm extractive|local|openai] [--store numpy|chroma]
Ground truth: a sentence is relevant to a question if the human annotators marked one of that
question's aspect terms in it. Questions are paraphrases that avoid the aspect word where possible."""
import argparse, csv, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reviewlens.pipeline import build_pipeline

ASPECTS = {  # question -> annotated aspect terms that make a sentence relevant
 "How long does it run unplugged?": {"battery life", "battery"},
 "What do owners think of the display?": {"screen", "display"},
 "Is typing comfortable?": {"keyboard", "keys"},
 "Is it worth what you pay?": {"price", "cost"},
 "How was the help from the support team?": {"customer service", "service", "tech support", "support", "customer support"},
 "Is it quick and responsive?": {"speed", "performance", "processor"},
 "How is the cursor pad?": {"touchpad", "trackpad", "mouse", "track pad", "mouse pad"},
 "What about the looks and design?": {"design", "looks"},
 "How do the speakers sound?": {"sound", "speakers", "audio"},
 "How does the operating system behave?": {"windows 8", "windows 7", "os", "operating system", "windows"},
 "Can it handle games and graphics?": {"graphics", "graphics card", "video card", "games"},
 "How well is it built?": {"build quality", "build", "construction", "case"},
}
OFFTOPIC = ["Who won the cricket world cup?", "What is the capital of France?", "How do I bake sourdough bread?",
 "What is the best pizza topping?", "Explain quantum entanglement.", "When does the next train to Mumbai leave?",
 "Is the stock market going up?", "How tall is Mount Everest?", "Recommend a good horror movie.",
 "What is the weather in Paris tomorrow?", "How do vaccines work?", "Who wrote Hamlet?"]

ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="extractive"); ap.add_argument("--store", default="numpy")
ap.add_argument("--embedder", default="minilm"); ap.add_argument("--gen-n", type=int, default=12)
a = ap.parse_args()
path = "data/semeval_laptops.csv"
rel = {}
rows = list(csv.DictReader(open(path, encoding="utf-8")))
pipe = build_pipeline(path, a.embedder, a.store, a.llm)
for q, terms in ASPECTS.items():
    rel[q] = {r["id"] for r in rows if terms & {x.strip('"').lower() for x in r["aspects"].split("|")}}
print(f"{len(rows)} real review sentences, {pipe.store.count()} chunks, embedder={pipe.embedder.name}, store={a.store}")
print(f"{len(ASPECTS)} paraphrased in-domain questions (relevant sentences per question: {min(map(len, rel.values()))}-{max(map(len, rel.values()))})\n")

def retrieval(mode, k=10):
    hit5 = rr = prec = 0
    for q, R in rel.items():
        ids = [h.chunk.review_id for h in pipe.retriever.search(q, k=k, mode=mode)]
        hit5 += any(i in R for i in ids[:5]); prec += sum(i in R for i in ids[:5]) / 5
        rr += next((1 / n for n, i in enumerate(ids, 1) if i in R), 0)
    n = len(rel); return hit5 / n, prec / n, rr / n
print("RETRIEVAL      Hit@5  P@5   MRR")
for m in ("bm25", "dense", "hybrid"):
    h, p, r = retrieval(m); print(f"  {m:8s}     {h:.2f}   {p:.2f}  {r:.2f}")

gate = lambda q: bool(pipe.relevant(pipe.retriever.search(q, k=20)))  # relevance gate only (no generation)
ans_in = [gate(q) for q in ASPECTS]; ans_off = [gate(q) for q in OFFTOPIC]
print(f"\nABSTENTION  answered in-domain: {sum(ans_in)}/{len(ans_in)}   abstained off-topic: {sum(not x for x in ans_off)}/{len(ans_off)}")

print(f"\nGENERATION  generator={pipe.llm.name}")
valid = faithful = fallback = n = gen_ok = 0
for q in list(ASPECTS)[: a.gen_n]:
    x = pipe.ask(q)
    if x.abstained: continue
    n += 1; fallback += bool(x.note)
    gen_ok += x.generative
    ctx = {t for h in x.sources for t in re.findall(r"[a-z0-9]+", h.chunk.text.lower())}
    body = re.sub(r"\[[^\]]*\]", " ", x.text.lower())
    toks = [t for t in re.findall(r"[a-z0-9]+", body) if len(t) > 3]
    if x.generative: faithful += (sum(t in ctx for t in toks) / max(1, len(toks)))
    valid += bool(x.citations)
if not pipe.llm.generative:
    print("  (extractive generator: nothing is generated; run with --llm local or --llm openai for generation metrics)"); sys.exit()
print(f"  answered by the generator with valid citations: {gen_ok}/{n}   fell back to extractive: {fallback}/{n}")
print(f"  mean share of generated-answer words found in retrieved context (crude faithfulness proxy, generated answers only): {faithful / max(1, gen_ok):.2f}")

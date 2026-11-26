# ReviewLens

A retrieval-augmented generation (RAG) pipeline that answers questions about customer reviews, shows the exact reviews it retrieved, cites them by id, and says "not enough evidence" when the reviews don't cover the question.

Problem: nobody can read 10,000 reviews. Ask "How long does it run unplugged?" and get an answer grounded in the reviews that actually mention it.

## Pipeline

```
reviews (CSV/JSONL) -> ingest + sentence-window chunking
  -> embeddings (MiniLM neural model, or offline LSA fallback)
  -> vector store (Chroma persistent DB, or numpy store; same interface)
question -> hybrid retrieval (dense vectors + BM25, fused with Reciprocal Rank Fusion, star-rating filters)
  -> relevance gate (abstain if nothing is close or covers the question's terms)
  -> generator (LLM) with numbered excerpts -> answer with [chunk id] citations
  -> citation check (ids must exist) + support check (an NLI model must find each statement entailed by its cited reviews)
     unsupported statements are removed; nothing usable left -> extractive quotes, labelled as such
low-star chunks -> k-means over stored embeddings -> complaint themes
```

### Generators (`--llm`, or `REVIEWLENS_LLM`)

| Option | What it is | Cost |
|---|---|---|
| `extractive` (default) | **Not generation.** Verbatim review sentences with ids. Always labelled "extractive" in CLI, UI and API. | free |
| `local` | Qwen3-0.6B via `transformers` (needs ~1.2 GB RAM, ~10 s per answer on one CPU core). Writes real paraphrases but is small. | free |
| `openai` | Any OpenAI-compatible `/chat/completions` endpoint (OpenAI, Ollama, vLLM, LM Studio). Set `OPENAI_API_KEY`, `REVIEWLENS_BASE_URL`, `REVIEWLENS_MODEL`. | paid providers charge you; nothing is called unless you select it |

The `openai` client is tested against a mock server only. I have not run it against a real provider.

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py                 # clearly synthetic demo reviews
python -m reviewlens ask "How is the battery life?"
python -m reviewlens --store chroma --store-path .index ask "do the earbuds hurt?" --max-stars 3
python -m reviewlens --llm local ask "How is the battery life?"
python -m reviewlens topics
python -m reviewlens serve --port 8000             # UI at /, JSON at /api/ask?q=...
python -m pytest -q                                # 24 tests
```

Your own reviews: `--data reviews.csv` with columns `id,text` (optional `rating,title,date`). `--embedder lsa` runs with no model download.

## Evaluation on real reviews

`python scripts/fetch_semeval.py` downloads the SemEval-2014 laptop-review set (human-annotated aspect terms; 1,903 unique review sentences). It is not committed to this repo: check its terms before redistributing. `python scripts/run_eval.py` then runs 12 paraphrased questions such as "How long does it run unplugged?" (battery), "How is the cursor pad?" (touchpad). A sentence counts as relevant if annotators marked one of that aspect's terms in it.

| Retriever | Hit@5 | P@5 | MRR |
|---|---:|---:|---:|
| BM25 | 0.67 | 0.43 | 0.59 |
| Dense (MiniLM) | 0.92 | 0.60 | 0.80 |
| Hybrid (default) | 0.92 | 0.53 | 0.77 |

- Dense retrieval clearly beats keyword search on paraphrased questions. **Hybrid does not beat dense here**; it is kept because exact-term queries (model names, error codes) favour BM25.
- Chroma and the numpy store give identical numbers.
- Abstention: 12/12 in-domain questions answered; 11/12 off-topic questions refused. The miss is "Is the stock market going up?", which matches a real review that says its author's opinion of Sony "has been dropping as fast as the stock market". 
- Local generation (Qwen3-0.6B + NLI support check, first 5 of the 12 questions): 3 produced a generated, verified answer (one of them had a statement removed), 2 fell back entirely to extractive quotes. About 64% of the generated words appear in the retrieved context (crude overlap proxy). Example: "How long does the battery last?" -> "The battery lasts about 6 hours. [L1010] [L455]" (a second statement was removed as unsupported).
- **What the checks do and do not prove.** The citation check proves a cited review exists. The NLI check (`cross-encoder/nli-deberta-v3-xsmall`) is itself a small model: it can keep a wrong statement or drop a right one. Neither proves an answer is true. The UI says so and always shows the retrieved evidence next to the answer.
- 12 questions is a small test. There was no LLM judge or human rating of answer quality. Real generation quality with a strong model is untested.

## Limits

- **Mixed-product corpus.** The SemEval laptop set has no product column, and its sentences name many different laptops (MacBook, Toshiba, HP, Dell, Acer, Sony and others, from keyword counts). So "the battery lasts about 6 hours" means "some reviewers say so", not a spec of one laptop. Run the UI with `--corpus-label "Mixed reviews of many different laptops (SemEval-2014), not one product"`; the label shows under the search box, and the local generator is told to phrase answers as "Reviewers report...". For real use, ingest one product's reviews per index.

- Free local generation is small and often falls back or drops statements; use an OpenAI-compatible model for stronger answers (untested against a real provider).
- The relevance gate drops chunks far weaker than the best match (`rel_cutoff`), which fixes stray off-point quotes but can also hide a valid minority view.
- Answers summarise at most `k=5` retrieved chunks, so they are not a statistical summary of all reviews.
- `data/demo_reviews_synthetic.csv` is generated from templates for a fictional product. Use it for a demo, not for judging quality.
- Retrieval and gate thresholds were set by hand and checked on the same 12 questions.

## Layout

```
reviewlens/  ingest, embed, store, retrieve, generate, pipeline, patterns, server, cli
scripts/     fetch_semeval.py, run_eval.py, make_sample_data.py
tests/       24 tests: chunking, stores (numpy + Chroma), retrieval, filters, citation fidelity,
             abstention, claim-support removal, generator failure/fallback, OpenAI-compatible client (mock server), HTML escaping
```

MIT licensed.

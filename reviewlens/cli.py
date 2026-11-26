import argparse
import sys

from .patterns import mine_topics
from .pipeline import build_pipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="reviewlens", description="RAG over customer reviews, with citations.")
    p.add_argument("--data", default="data/demo_reviews_synthetic.csv")
    p.add_argument("--embedder", default="minilm", help="minilm (default), lsa (offline fallback) or any sentence-transformers id")
    p.add_argument("--store", default="numpy", choices=["numpy", "chroma"])
    p.add_argument("--store-path", help="persist the index here")
    p.add_argument("--llm", choices=["extractive", "local", "openai"], help="default: $REVIEWLENS_LLM or extractive")
    p.add_argument("--corpus-label", default="", help="one line describing the review set, shown in the UI (e.g. which product(s))")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("-k", type=int, default=5)
    a.add_argument("--max-stars", type=int); a.add_argument("--min-stars", type=int)
    t = sub.add_parser("topics"); t.add_argument("-n", type=int, default=6)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    pipe = build_pipeline(args.data, args.embedder, args.store, args.llm, args.store_path, corpus_label=args.corpus_label)
    if args.cmd == "ask":
        r = pipe.ask(args.question, k=args.k, min_rating=args.min_stars, max_rating=args.max_stars)
        print(r.text)
        if not r.abstained:
            print(f"\n[generator: {r.generator}{'' if r.generative else ' - verbatim quotes, no text generation'}]" + (f" ({r.note})" if r.note else ""))
    elif args.cmd == "topics":
        for tp in mine_topics(pipe.store, k=args.n):
            print(f"{tp.size:4d} chunks  avg {tp.avg_rating:.1f}/5  {tp.label}   e.g. {', '.join(tp.chunk_ids[:3])}")
    else:
        from .server import serve
        serve(pipe, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

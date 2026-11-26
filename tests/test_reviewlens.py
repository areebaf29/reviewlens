import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reviewlens.embed import LSAEmbedder
from reviewlens.generate import OpenAICompatLLM, build_prompt
from reviewlens.ingest import Chunk, chunk_text, ingest
from reviewlens.patterns import mine_topics
from reviewlens.pipeline import NO_EVIDENCE, RAGPipeline, build_pipeline
from reviewlens.server import render
from reviewlens.store import NumpyStore

DEMO = os.path.join(ROOT, "data", "demo_reviews_synthetic.csv")


@pytest.fixture(scope="module")
def pipe():
    return build_pipeline(DEMO, embedder="lsa", llm="extractive")


# ---- ingestion / chunking
def test_chunking_keeps_short_reviews_whole_and_splits_long_ones():
    assert chunk_text("Short review.") == ["Short review."]
    long = " ".join(f"Sentence number {i} talks about battery and screen quality here." for i in range(20))
    cs = chunk_text(long, max_words=30, overlap=1)
    assert len(cs) > 5 and all(len(c.split()) <= 45 for c in cs)
    assert cs[0].split(".")[-2].strip() in cs[1]  # overlap sentence carried over


def test_ingest_validation(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("id,rating\n1,5\n")
    with pytest.raises(ValueError, match="text"):
        ingest(str(p))
    p.write_text("id,rating,text\n1,5,good\n1,4,fine\n")
    with pytest.raises(ValueError, match="duplicate"):
        ingest(str(p))
    p.write_text("id,rating,text\n1,x,good\n")
    with pytest.raises(ValueError, match="rating"):
        ingest(str(p))


def test_ingest_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"id": "a", "rating": 5, "text": "Great screen."}) + "\n")
    assert ingest(str(p))[0].review_id == "a"


# ---- vector stores
def test_numpy_store_roundtrip_and_filter(tmp_path):
    cs = [Chunk("a", "a", "x", 5), Chunk("b", "b", "y", 1)]
    v = np.array([[1.0, 0], [0.9, 0.43589]])
    s = NumpyStore(str(tmp_path / "idx"))
    s.add(cs, v)
    s2 = NumpyStore(str(tmp_path / "idx"))  # reload from disk
    assert s2.count() == 2
    assert [c.id for c, _ in s2.query(np.array([1.0, 0]), 2)] == ["a", "b"]
    assert [c.id for c, _ in s2.query(np.array([1.0, 0]), 2, max_rating=2)] == ["b"]


def test_chroma_store_matches_numpy():
    chromadb = pytest.importorskip("chromadb")
    from reviewlens.store import ChromaStore
    cs = [Chunk("a", "a", "x", 5), Chunk("b", "b", "y", 1), Chunk("c", "c", "z", 3)]
    v = np.array([[1.0, 0], [0.0, 1.0], [0.7071, 0.7071]])
    ch, nm = ChromaStore(None, "test_reviews"), NumpyStore()
    ch.add(cs, v), nm.add(cs, v)
    q = np.array([1.0, 0.0])
    assert [c.id for c, _ in ch.query(q, 3)] == [c.id for c, _ in nm.query(q, 3)]
    assert [c.id for c, _ in ch.query(q, 3, min_rating=3, max_rating=3)] == ["c"]


# ---- retrieval + pipeline
def test_retrieval_finds_topical_chunks(pipe):
    hits = pipe.retriever.search("microphone on calls", k=3)
    assert hits and any("mic" in h.chunk.text.lower() or "call" in h.chunk.text.lower() for h in hits)


def test_rating_filter(pipe):
    hits = pipe.retriever.search("battery", k=10, max_rating=2)
    assert hits and all(h.chunk.rating <= 2 for h in hits)


def test_answer_quotes_are_verbatim_and_cited(pipe):
    a = pipe.ask("How is the battery life?")
    assert not a.abstained and a.citations and not a.generative
    by_id = {h.chunk.id: h.chunk.text for h in a.sources}
    for line in a.text.splitlines():
        sent, cid = re.match(r'"(.+)" \[(.+)\]', line).groups()
        assert sent in by_id[cid]


def test_abstains_offtopic_and_empty(pipe):
    for q in ["what do people say about the stock market", "zzzz qqqq", "   "]:
        assert pipe.ask(q).text == NO_EVIDENCE


def test_topics(pipe):
    t = mine_topics(pipe.store, k=5)
    assert t and all(x.avg_rating <= 3 for x in t)


# ---- generation: fake generators exercise the citation contract
class Fake:
    generative, name = True, "fake"
    def __init__(self, reply): self.reply = reply
    def generate(self, q, chunks): return self.reply(chunks) if callable(self.reply) else self.reply


def _with(pipe, llm):
    return RAGPipeline(pipe.store, pipe.embedder, llm, min_dense=pipe.min_dense)


def test_valid_generated_answer_kept(pipe):
    p = _with(pipe, Fake(lambda cs: f"Reviewers say the battery life is mixed overall [{cs[0].id}]."))
    a = p.ask("How is the battery life?")
    assert a.generative and a.citations and not a.note


def test_invented_citation_is_stripped(pipe):
    p = _with(pipe, Fake(lambda cs: f"Battery life is mixed [{cs[0].id}] and always great [R999999]."))
    a = p.ask("How is the battery life?")
    assert "R999999" not in a.text and a.generative


def test_uncited_generation_falls_back_and_says_so(pipe):
    a = _with(pipe, Fake("The battery is wonderful and everyone loves it.")).ask("How is the battery life?")
    assert not a.generative and "no usable cited answer" in a.note and a.citations


def test_generator_can_abstain(pipe):
    assert _with(pipe, Fake("INSUFFICIENT EVIDENCE")).ask("How is the battery life?").abstained


def test_generator_failure_degrades_gracefully(pipe):
    def boom(cs): raise TimeoutError()
    a = _with(pipe, Fake(boom)).ask("How is the battery life?")
    assert not a.generative and "TimeoutError" in a.note


def test_openai_compatible_client_against_mock_server(pipe):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(path=self.path, auth=self.headers["Authorization"], body=body)
            cid = re.search(r"\[(R\d+)\]", body["messages"][1]["content"]).group(1)
            out = json.dumps({"choices": [{"message": {"content": f"Battery life is mixed according to reviewers [{cid}]."}}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
        def log_message(self, *a): pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    llm = OpenAICompatLLM(model="m", base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k")
    a = _with(pipe, llm).ask("How is the battery life?")
    srv.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert "ONLY" in seen["body"]["messages"][0]["content"] and "Excerpts:" in seen["body"]["messages"][1]["content"]
    assert a.generative and a.citations


def test_prompt_contains_ids_and_ratings():
    p = build_prompt("q?", [Chunk("R1", "R1", "text", 4)])
    assert "[R1] (4/5) text" in p


def test_web_ui_escapes_html(pipe):
    page = render(pipe, "<script>alert(1)</script> battery")
    assert "<script>alert(1)</script>" not in page and "ReviewLens" in page


# ---- claim support check
def test_split_claims_attaches_trailing_cites_and_cleans_attribution():
    from reviewlens.verify import atomic_claims, split_claims
    c = split_claims("The battery lasts about 6 hours as per [L1], and it is light as per [L2]. [L1][L2]")
    assert c == [("The battery lasts about 6 hours, and it is light.", ["L1", "L2", "L1", "L2"])]
    assert [x for x, _ in atomic_claims("It is fast, and the screen is dim. [R1]")] == ["It is fast.", "The screen is dim."]


class FakeNLI:
    def supported(self, premise, claim):
        return "wonderful" not in claim


def test_unsupported_claims_are_removed_and_reported(pipe):
    p = _with(pipe, Fake(lambda cs: f"Battery life is mixed [{cs[0].id}]. The battery is wonderful [{cs[0].id}]."))
    p.verifier = FakeNLI()
    a = p.ask("How is the battery life?")
    assert a.generative and "wonderful" not in a.text and "removed" in a.note


def test_all_claims_unsupported_falls_back_to_extractive(pipe):
    p = _with(pipe, Fake(lambda cs: f"The battery is wonderful [{cs[0].id}]."))
    p.verifier = FakeNLI()
    a = p.ask("How is the battery life?")
    assert not a.generative and "support check" in a.note


def test_hedge_inside_answer_is_dropped_not_abstained(pipe):
    p = _with(pipe, Fake(lambda cs: f"Battery life is mixed [{cs[0].id}]. Insufficient evidence."))
    a = p.ask("How is the battery life?")
    assert a.generative and "nsufficient" not in a.text


def test_ui_has_no_default_blue_and_labels_mode(pipe):
    page = render(pipe, "How is the battery life?")
    assert "extractive quotes, not generated" in page and "#3b5bdb" not in page


def test_ui_shows_corpus_label(pipe):
    pipe.corpus_label = "Mixed laptop reviews <b>"
    page = render(pipe, "")
    pipe.corpus_label = ""
    assert "Review set:" in page and "Mixed laptop reviews &lt;b&gt;" in page

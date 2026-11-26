"""Answer generators. All take (question, context chunks) and return text with [chunk id] citations.

  OpenAICompatLLM  any OpenAI-compatible /chat/completions endpoint (OpenAI, Ollama, vLLM, LM Studio...).
                   Needs REVIEWLENS_BASE_URL (default https://api.openai.com/v1), REVIEWLENS_MODEL, OPENAI_API_KEY.
                   Calls cost money on paid providers; nothing is sent unless you select it.
  LocalChatLLM     Qwen3-0.6B run locally via transformers (free, tiny, weak).
  ExtractiveLLM    NOT generation: stitches verbatim review sentences. Always labelled as such.
"""
import json
import os
import urllib.request
from typing import Protocol

from .ingest import Chunk
from .text import sentences, tokenize

SYSTEM = ("You answer questions about a product using ONLY the numbered customer-review excerpts provided. "
          "Cite the excerpt id in square brackets after every claim, like [R12]. If the excerpts do not contain the "
          "answer, reply exactly: INSUFFICIENT EVIDENCE. Do not use outside knowledge.")


SYSTEM_SHORT = ("You answer questions about a product from customer-review excerpts. Use ONLY the excerpts. "
                "The excerpts may come from different products, so never state a specification as a fact about one product: write as 'Reviewers report...' or 'One reviewer says...'. Write 1-2 sentences in your own words that summarise what reviewers say, and end each sentence with "
                "the ids of the excerpts that support it in square brackets, like [L12][L40]. Do not copy excerpts "
                "word for word and do not write phrases like 'as per' or 'according to'. If the excerpts do not answer the question, write: INSUFFICIENT EVIDENCE.")


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    ctx = "\n".join(f"[{c.id}] ({c.rating}/5) {c.text}" for c in chunks)
    return f"Excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer (with [id] citations):"


class LLM(Protocol):
    name: str
    generative: bool
    def generate(self, question: str, chunks: list[Chunk]) -> str: ...


class ExtractiveLLM:
    name, generative = "extractive (not generative)", False

    def generate(self, question, chunks):
        q = set(tokenize(question))
        lines = []
        for c in chunks:
            ss = sentences(c.text) or [c.text]
            best = max(ss, key=lambda s: len(q & set(tokenize(s))))
            lines.append(f'"{best}" [{c.id}]')
        return "\n".join(lines)


class OpenAICompatLLM:
    generative = True

    def __init__(self, model=None, base_url=None, api_key=None, timeout=60):
        self.model = model or os.environ.get("REVIEWLENS_MODEL", "gpt-4o-mini")
        self.base = (base_url or os.environ.get("REVIEWLENS_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout, self.name = timeout, f"openai-compatible:{self.model}"

    def generate(self, question, chunks):
        body = json.dumps({"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(question, chunks)}]}).encode()
        req = urllib.request.Request(self.base + "/chat/completions", body,
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()


class LocalChatLLM:
    """Small instruct model run locally with transformers (free, offline after download).
    Default Qwen3-0.6B (thinking off): fits in ~1.2 GB, writes real paraphrases, but is small:
    it can overstate or mis-cite, which is why the pipeline runs a citation check and an NLI support check."""
    generative = True

    def __init__(self, model="Qwen/Qwen3-0.6B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
        self.name = f"local:{model}"

    def generate(self, question, chunks):
        ctx = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
        msgs = [{"role": "system", "content": SYSTEM_SHORT},
                {"role": "user", "content": f"Excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer:"}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                            enable_thinking=False)
        out = self.m.generate(**ids, max_new_tokens=110, do_sample=False)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def get_llm(name: str | None = None) -> LLM:
    name = name or os.environ.get("REVIEWLENS_LLM", "extractive")
    if name == "openai":
        return OpenAICompatLLM()
    if name == "local":
        return LocalChatLLM()
    if name == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"unknown LLM '{name}' (use extractive, local or openai)")

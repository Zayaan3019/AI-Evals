"""A small, real question-answering agent, used as the system under test.

Retrieval-augmented QA over a slice of the SQuAD v1.1 dev set: BM25 retrieves
passages from a 400-paragraph corpus, and a hosted LLM answers from them. Every
step is traced with the agentlens SDK, so one call produces a workflow span, a
retrieval span carrying the retrieved documents and their scores, and an LLM
span carrying the provider's own token counts.

It exists because an evaluation platform has to be exercised by something that
can actually be wrong. This agent fails in the ordinary ways: BM25 misses the
passage, the model answers from the wrong one, the model pads a short answer,
the provider rate-limits. None of that is scripted.

The model call goes through :func:`~..evaluators.judges.build_judge`, which is
just the platform's hosted-LLM client; that also gives the agent the same
record/replay cassette the judges use (``AGENTLENS_JUDGE_CASSETTE``).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import agentlens

from ..evaluators.judges import JudgeBackend, build_judge

DATA = Path(__file__).parent / "data" / "squad_sample.json"

SYSTEM_PROMPT = (
    "You answer questions using only the passages provided. Reply with the shortest "
    "phrase from the passages that answers the question: no full sentence, no "
    "explanation, no quotation marks. If the passages do not contain the answer, "
    "reply exactly: unknown"
)
CLOSED_BOOK_PROMPT = (
    "Answer the question from your own knowledge. Reply with the shortest phrase "
    "that answers it: no full sentence, no explanation, no quotation marks. If you "
    "do not know, reply exactly: unknown"
)

_TOKEN = re.compile(r"[a-z0-9]+")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True, slots=True)
class Passage:
    id: str
    title: str
    text: str


class BM25:
    """Okapi BM25 over a fixed corpus. Small enough to own rather than import."""

    def __init__(self, passages: list[Passage], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(f"{p.title} {p.text}")) for p in passages]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(1, len(self._len))
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(passages)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[Passage, float]]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, tf in enumerate(self._tf):
            score = 0.0
            norm = self._k1 * (1 - self._b + self._b * self._len[index] / self._avg)
            for term in terms:
                f = tf.get(term)
                if f:
                    score += self._idf[term] * f * (self._k1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(self.passages[i], s) for s, i in scored[:top_k]]


@lru_cache(maxsize=1)
def load_data() -> dict[str, Any]:
    return json.loads(DATA.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def retriever() -> BM25:
    return BM25([Passage(p["id"], p["title"], p["text"]) for p in load_data()["corpus"]])


def clean_answer(text: str) -> str:
    """Strip reasoning tags and wrapping punctuation some models add."""
    text = _THINK.sub("", text).strip()
    text = text.strip().strip("\"'`").strip()
    return text.rstrip(".").strip() if len(text.split()) <= 12 else text


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """One version of the agent. Two configs differing in any field are two systems."""

    model: str
    provider: str = "groq"
    retrieve: bool = True
    top_k: int = 2
    max_tokens: int = 200

    @property
    def label(self) -> str:
        mode = f"BM25 top-{self.top_k}" if self.retrieve else "no retrieval"
        return f"qa-agent | {self.model} | {mode}"


class QAAgent:
    def __init__(self, config: AgentConfig, *, llm: JudgeBackend | None = None) -> None:
        self.config = config
        self._llm = llm or build_judge(config.provider, config.model)

    def answer(self, question: str) -> str:
        """Answer one question. Raises on a provider failure; the caller decides."""
        cfg = self.config
        with agentlens.workflow("answer_question") as wf:
            wf.set_input(question)
            wf.set_attribute("agentlens.agent.config", cfg.label)

            passages: list[tuple[Passage, float]] = []
            if cfg.retrieve:
                with agentlens.retrieval("squad_corpus_bm25", query=question, top_k=cfg.top_k) as span:
                    passages = retriever().search(question, cfg.top_k)
                    span.set_documents(
                        [
                            agentlens.Document(
                                id=p.id, score=round(score, 4), content=p.text, source=p.title
                            )
                            for p, score in passages
                        ]
                    )

            if cfg.retrieve:
                context = "\n\n".join(
                    f"[{i + 1}] ({p.title}) {p.text}" for i, (p, _) in enumerate(passages)
                ) or "(no passages retrieved)"
                system, prompt = SYSTEM_PROMPT, f"Passages:\n{context}\n\nQuestion: {question}"
            else:
                system, prompt = CLOSED_BOOK_PROMPT, f"Question: {question}"

            with agentlens.llm_call(
                cfg.model,
                provider=cfg.provider,
                temperature=0.0,
                max_tokens=cfg.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            ) as llm:
                response = self._llm.complete(
                    prompt, system=system, max_tokens=cfg.max_tokens, temperature=0.0
                )
                answer = clean_answer(response.text)
                llm.set_response(answer)
                if response.input_tokens is not None and response.output_tokens is not None:
                    llm.set_usage(response.input_tokens, response.output_tokens)

            wf.set_output(answer)
            return answer

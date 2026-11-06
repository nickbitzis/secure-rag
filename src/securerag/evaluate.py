"""Retrieval eval: recall@k, leakage rate, and cross-tenant leakage per retrieval mode.

Leakage is checked against the corpus ACLs directly, not against what the retriever
claims, so a bug in the filter shows up here as a nonzero number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .corpus import Corpus
from .retrieval import MODES, Retriever, can_read


@dataclass(frozen=True)
class Question:
    id: str
    user: str
    question: str
    expected_docs: tuple[str, ...]
    refuse: bool


def load_questions(path: str | Path) -> list[Question]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            out.append(Question(q["id"], q["user"], q["question"], tuple(q.get("expected_docs", [])), bool(q.get("refuse"))))
    return out


@dataclass
class ModeResult:
    mode: str
    questions: int
    recall: float          # mean recall@k over answerable questions
    leak_rate: float       # share of questions with any chunk the user may not read
    cross_tenant_rate: float  # share of questions with any chunk from another tenant
    leaked_chunks: int
    refusal_acc: float     # share of should-refuse questions that got zero chunks
    details: list[dict]


def evaluate(corpus: Corpus, retriever: Retriever, questions: list[Question], mode: str, k: int = 4) -> ModeResult:
    details = []
    recalls, leaks, cross, leaked, refusals = [], 0, 0, 0, []
    for q in questions:
        user = corpus.user(q.user)
        if user is None:
            raise ValueError(f"{q.id}: unknown user {q.user}")
        r = retriever.retrieve(user, q.question, k=k, mode=mode)
        bad = [h.chunk.id for h in r.hits if not can_read(user, corpus.doc(h.chunk.doc_id))]
        other = [h.chunk.id for h in r.hits if h.chunk.tenant != user.tenant]
        leaks += bool(bad)
        cross += bool(other)
        leaked += len(bad)
        got = set(r.doc_ids)
        rec = None
        if q.expected_docs:
            rec = len(got & set(q.expected_docs)) / len(q.expected_docs)
            recalls.append(rec)
        if q.refuse:
            refusals.append(not r.hits)
        details.append({"id": q.id, "user": q.user, "docs": r.doc_ids, "recall": rec, "leaked": bad, "cross_tenant": other})
    n = len(questions)
    return ModeResult(
        mode=mode,
        questions=n,
        recall=sum(recalls) / len(recalls) if recalls else 1.0,
        leak_rate=leaks / n if n else 0.0,
        cross_tenant_rate=cross / n if n else 0.0,
        leaked_chunks=leaked,
        refusal_acc=sum(refusals) / len(refusals) if refusals else 1.0,
        details=details,
    )


def run_all(corpus: Corpus, retriever: Retriever, questions: list[Question], k: int = 4, modes=MODES) -> list[ModeResult]:
    return [evaluate(corpus, retriever, questions, m, k) for m in modes]


def table(results: list[ModeResult], k: int) -> str:
    head = f"{'mode':<12}{'questions':>10}{f'recall@{k}':>11}{'leak rate':>11}{'cross-tenant':>14}{'leaked chunks':>15}{'correct refusals':>18}"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.mode:<12}{r.questions:>10}{r.recall:>11.1%}{r.leak_rate:>11.1%}{r.cross_tenant_rate:>14.1%}"
            f"{r.leaked_chunks:>15}{r.refusal_acc:>18.1%}"
        )
    return "\n".join(lines)

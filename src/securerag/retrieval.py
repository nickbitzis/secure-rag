"""Chunking, BM25, and permission-aware retrieval.

The one rule this module exists to enforce: the set of chunks a user can see is
decided BEFORE ranking. A chunk the user may not read is never scored, never
ranked, and never handed to the answer model. The model is not asked to hide
anything, because it never receives anything it would need to hide.

Three retrieval modes are kept side by side so the eval can compare designs:

- prefilter:  tenant partition + ACL filter, then BM25 over what is left (the real design)
- postfilter: BM25 over one shared index of every tenant, take top k, then drop what the
              user may not read (a common first attempt; loses recall)
- nofilter:   BM25 over the shared index with no checks (what leaks)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Corpus, Doc, User

MODES = ("prefilter", "postfilter", "nofilter")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its many me much my
    of on or our per should tell that the their them there this to us was we what when where which
    who why will with you your about after all any before did into more most than then these those
    up out over under get got""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        # Light plural folding so "bands" matches "band". No stemmer dependency.
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    tenant: str
    acl: tuple[str, ...]
    title: str
    text: str


def chunk_doc(doc: Doc, max_words: int = 60) -> list[Chunk]:
    """Pack paragraphs into chunks of at most `max_words` words. The title is prepended
    to each chunk so a chunk is understandable on its own and matches title words."""
    paras = [p.strip() for p in doc.body.split("\n\n") if p.strip()]
    groups: list[list[str]] = []
    count = 0
    for p in paras:
        n = len(p.split())
        if groups and count + n <= max_words:
            groups[-1].append(p)
            count += n
        else:
            groups.append([p])
            count = n
    return [
        Chunk(f"{doc.id}#{i}", doc.id, doc.tenant, doc.acl, doc.title, f"{doc.title}. " + "\n\n".join(g))
        for i, g in enumerate(groups)
    ]


def can_read(user: User, chunk: Chunk | Doc) -> bool:
    """Tenant isolation first, then the ACL. Both must pass."""
    return chunk.tenant == user.tenant and bool(set(chunk.acl) & set(user.groups))


def coverage(terms: set[str], tf: Counter) -> float:
    return sum(1 for t in terms if t in tf) / len(terms) if terms else 0.0


class BM25:
    """Okapi BM25 over a fixed list of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, i: int, terms: list[str]) -> float:
        tf, dl, s = self.tfs[i], self.lens[i], 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def search(self, query: str, k: int, candidates: list[int] | None = None) -> list[tuple[int, float]]:
        """Top k (index, score) with score > 0. Only `candidates` are scored when given."""
        terms = tokenize(query)
        pool = range(len(self.chunks)) if candidates is None else candidates
        scored = [(i, self.score(i, terms)) for i in pool]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: (-x[1], self.chunks[x[0]].id))
        return scored[:k]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Retrieval:
    hits: list[Hit]
    filtered_docs: int  # docs in the user's tenant they may not read (query independent)
    mode: str

    @property
    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for h in self.hits:
            if h.chunk.doc_id not in seen:
                seen.append(h.chunk.doc_id)
        return seen


class Retriever:
    """Holds one BM25 index per tenant (the real path) and one shared index across
    tenants (only used by the two comparison modes)."""

    def __init__(self, corpus: Corpus, max_words: int = 60, min_score: float = 1.0, min_coverage: float = 0.5):
        self.corpus = corpus
        self.min_score = min_score
        # Share of distinct query terms a chunk must contain. Weak single-word matches
        # are dropped, so a question about something the user cannot see comes back
        # empty and the answerer refuses instead of quoting a loosely related doc.
        self.min_coverage = min_coverage
        self.chunks = [c for d in corpus.docs for c in chunk_doc(d, max_words)]
        self.by_tenant = {t: BM25([c for c in self.chunks if c.tenant == t]) for t in corpus.tenants}
        self.shared = BM25(self.chunks)

    def allowed_count(self, user: User) -> tuple[int, int]:
        docs = [d for d in self.corpus.docs if d.tenant == user.tenant]
        allowed = sum(1 for d in docs if can_read(user, d))
        return allowed, len(docs)

    def retrieve(self, user: User, query: str, k: int = 4, mode: str = "prefilter") -> Retrieval:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        allowed, total = self.allowed_count(user)
        if mode == "prefilter":
            index = self.by_tenant.get(user.tenant)
            if index is None:
                return Retrieval([], 0, mode)
            candidates = [i for i, c in enumerate(index.chunks) if can_read(user, c)]
            ranked = index.search(query, k, candidates)
        elif mode == "postfilter":
            index = self.shared
            ranked = [(i, s) for i, s in index.search(query, k) if can_read(user, index.chunks[i])]
        else:
            index = self.shared
            ranked = index.search(query, k)
        terms = set(tokenize(query))
        hits = [
            Hit(index.chunks[i], s)
            for i, s in ranked
            if s >= self.min_score and coverage(terms, index.tfs[i]) >= self.min_coverage
        ]
        return Retrieval(hits, total - allowed, mode)

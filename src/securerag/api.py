"""FastAPI service: POST /ask, GET /healthz, GET /metrics, and a small web UI at /.

The API only ever uses the prefilter retrieval mode. The comparison modes exist in
the eval, not here.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from . import __version__
from .answer import Answerer, make_answerer
from .audit import AuditLog, query_hash
from .corpus import Corpus, build_corpus, load_corpus
from .identity import DEV_SECRET, secret_from_env, sign, verify
from .logs import event, get_logger, now_ms
from .retrieval import Retriever

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("securerag_requests_total", "HTTP requests", ["path", "status"], registry=self.registry)
        self.asks = Counter("securerag_asks_total", "Questions answered", ["tenant", "outcome"], registry=self.registry)
        self.auth_failures = Counter("securerag_auth_failures_total", "Requests with a missing or bad token", registry=self.registry)
        self.filtered = Counter("securerag_docs_filtered_total", "Docs excluded by ACL before ranking, summed over asks", registry=self.registry)
        self.latency = Histogram("securerag_ask_seconds", "Time to answer /ask", registry=self.registry,
                                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
        self.tokens = Counter("securerag_model_tokens_total", "Model tokens used", ["kind"], registry=self.registry)


def _default_corpus() -> Corpus:
    path = os.environ.get("SECURERAG_CORPUS")
    return load_corpus(Path(path)) if path else build_corpus()


def create_app(
    corpus: Corpus | None = None,
    answerer: Answerer | None = None,
    audit_path: str | Path | None = None,
    secret: str | None = None,
    demo_tokens: bool | None = None,
) -> FastAPI:
    corpus = corpus or _default_corpus()
    secret = secret or secret_from_env()
    if demo_tokens is None:
        demo_tokens = os.environ.get("SECURERAG_DEMO_TOKENS", "1") == "1"
    answerer = answerer or make_answerer(os.environ.get("SECURERAG_ANSWERER", "extractive"))
    audit = AuditLog(audit_path or os.environ.get("SECURERAG_AUDIT_DB", ":memory:"))
    retriever = Retriever(corpus)
    metrics = Metrics()
    log = get_logger()

    app = FastAPI(title="securerag", version=__version__)
    app.state.audit, app.state.metrics, app.state.retriever = audit, metrics, retriever

    if secret == DEV_SECRET:
        event(log, "using the built-in dev secret; set SECURERAG_SECRET outside local demos")

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path if request.url.path in {"/", "/ask", "/healthz", "/metrics", "/api/users"} else "other"
        metrics.requests.labels(path, str(response.status_code)).inc()
        return response

    def current_user(authorization: str | None):
        token = (authorization or "").removeprefix("Bearer ").strip()
        user_id = verify(token, secret) if token else None
        user = corpus.user(user_id) if user_id else None
        if user is None:
            metrics.auth_failures.inc()
            raise HTTPException(status_code=401, detail="missing or invalid user token")
        return user

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "docs": len(corpus.docs), "chunks": len(retriever.chunks), "answerer": answerer.name}

    @app.get("/metrics")
    def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/users")
    def users() -> list[dict]:
        # Demo only: hands out signed tokens so the UI can switch users. A real
        # deployment gets the identity from SSO and disables this endpoint.
        if not demo_tokens:
            raise HTTPException(status_code=404, detail="not found")
        return [
            {"id": u.id, "name": u.name, "role": u.role, "company": corpus.tenants[u.tenant],
             "groups": list(u.groups), "token": sign(u.id, secret)}
            for u in corpus.users
        ]

    @app.post("/ask")
    def ask(body: AskRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
        start = time.perf_counter()
        user = current_user(authorization)
        request_id = uuid.uuid4().hex[:16]

        # Access is decided here, before ranking and before the model sees anything.
        result = retriever.retrieve(user, body.question, k=body.k, mode="prefilter")
        ans = answerer.answer(body.question, result.hits)

        audit.record(
            request_id=request_id, user_id=user.id, tenant=user.tenant, question=body.question,
            doc_ids=result.doc_ids, chunk_ids=[h.chunk.id for h in result.hits],
            filtered_docs=result.filtered_docs, refused=ans.refused, answerer=answerer.name,
        )
        elapsed = time.perf_counter() - start
        metrics.latency.observe(elapsed)
        metrics.asks.labels(user.tenant, "refused" if ans.refused else "answered").inc()
        metrics.filtered.inc(result.filtered_docs)
        metrics.tokens.labels("input").inc(ans.input_tokens)
        metrics.tokens.labels("output").inc(ans.output_tokens)
        event(log, "ask", request_id=request_id, user=user.id, tenant=user.tenant,
              query_sha256=query_hash(body.question), doc_ids=result.doc_ids,
              filtered_docs=result.filtered_docs, refused=ans.refused, answerer=answerer.name,
              latency_ms=now_ms(start))

        by_id = {h.chunk.id: h for h in result.hits}
        return JSONResponse({
            "request_id": request_id,
            "user": user.id,
            "company": corpus.tenants[user.tenant],
            "answer": ans.text,
            "refused": ans.refused,
            "citations": [
                {"chunk_id": c, "doc_id": by_id[c].chunk.doc_id, "title": by_id[c].chunk.title,
                 "score": round(by_id[c].score, 3)}
                for c in ans.citations
            ],
            "retrieved": [h.chunk.id for h in result.hits],
            # Count only. Titles of docs the user cannot read are never returned.
            "filtered_out_docs": result.filtered_docs,
            "answerer": answerer.name,
        })

    return app


def app_from_env() -> FastAPI:  # pragma: no cover - used by uvicorn --factory
    return create_app()

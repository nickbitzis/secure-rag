"""Append-only audit log in SQLite: who asked, a hash of what they asked, and which
documents were retrieved. The question text itself is not stored, since questions
can contain sensitive details. The hash still lets an investigator confirm a known
question was asked."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    tenant TEXT NOT NULL,
    query_sha256 TEXT NOT NULL,
    doc_ids TEXT NOT NULL,
    chunk_ids TEXT NOT NULL,
    filtered_docs INTEGER NOT NULL,
    refused INTEGER NOT NULL,
    answerer TEXT NOT NULL
)
"""


def query_hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path = ":memory:"):
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(SCHEMA)
        self.db.commit()

    def record(self, *, request_id: str, user_id: str, tenant: str, question: str, doc_ids: list[str],
               chunk_ids: list[str], filtered_docs: int, refused: bool, answerer: str) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO audit (ts, request_id, user_id, tenant, query_sha256, doc_ids, chunk_ids, filtered_docs, refused, answerer)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), request_id, user_id, tenant, query_hash(question), json.dumps(doc_ids),
                 json.dumps(chunk_ids), filtered_docs, int(refused), answerer),
            )
            self.db.commit()

    def rows(self, limit: int = 100) -> list[dict]:
        with self._lock:
            cur = self.db.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))
            cols = [c[0] for c in cur.description]
            out = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
        for r in out:
            r["doc_ids"] = json.loads(r["doc_ids"])
            r["chunk_ids"] = json.loads(r["chunk_ids"])
        return out

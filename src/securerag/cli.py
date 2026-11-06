"""Command line: `securerag gen | eval | ask | serve`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .answer import make_answerer
from .corpus import build_corpus, load_corpus, write_corpus
from .evaluate import load_questions, run_all, table
from .retrieval import MODES, Retriever


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="securerag", description="Permission-aware, multi-tenant document Q&A.")
    p.add_argument("--version", action="version", version=f"securerag {__version__}")
    p.add_argument("--corpus", help="load the corpus from this folder instead of generating it in memory")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="write the synthetic corpus to disk as markdown")
    g.add_argument("--out", default="corpus")

    e = sub.add_parser("eval", help="recall@k and leakage for each retrieval mode")
    e.add_argument("--questions", default="examples/eval.jsonl")
    e.add_argument("-k", type=int, default=4)
    e.add_argument("--mode", choices=(*MODES, "all"), default="all")
    e.add_argument("--json", action="store_true", help="print per-question details as JSON")
    e.add_argument("--gate", action="store_true",
                   help="exit 1 if the prefilter mode leaks anything (used in CI)")
    e.add_argument("--min-recall", type=float, default=0.0, help="with --gate, also require prefilter recall at or above this")

    a = sub.add_parser("ask", help="ask one question as a given user")
    a.add_argument("question")
    a.add_argument("--user", required=True, help="user id, for example hal-frank")
    a.add_argument("-k", type=int, default=4)
    a.add_argument("--mode", choices=MODES, default="prefilter")
    a.add_argument("--answerer", choices=("extractive", "claude"), default="extractive")

    s = sub.add_parser("serve", help="run the HTTP service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_corpus(Path(args.corpus)) if args.corpus else build_corpus()

    if args.command == "gen":
        n = write_corpus(corpus, Path(args.out))
        print(f"wrote {n} docs and users.json to {args.out}")
        return 0

    if args.command == "eval":
        questions = load_questions(args.questions)
        modes = MODES if args.mode == "all" else (args.mode,)
        results = run_all(corpus, Retriever(corpus), questions, k=args.k, modes=modes)
        if args.json:
            print(json.dumps([r.__dict__ for r in results], indent=2))
        else:
            print(table(results, args.k))
        if args.gate:
            pre = next((r for r in results if r.mode == "prefilter"), None)
            if pre is None:
                print("gate needs the prefilter mode", file=sys.stderr)
                return 2
            problems = []
            if pre.leaked_chunks or pre.leak_rate > 0:
                problems.append(f"leakage {pre.leak_rate:.1%} ({pre.leaked_chunks} chunks)")
            if pre.cross_tenant_rate > 0:
                problems.append(f"cross-tenant leakage {pre.cross_tenant_rate:.1%}")
            if pre.recall < args.min_recall:
                problems.append(f"recall {pre.recall:.1%} below {args.min_recall:.1%}")
            if problems:
                print("SECURITY GATE FAILED: " + "; ".join(problems), file=sys.stderr)
                return 1
            print("security gate passed: prefilter leakage 0, cross-tenant leakage 0")
        return 0

    if args.command == "ask":
        user = corpus.user(args.user)
        if user is None:
            print(f"unknown user {args.user}. Known: {', '.join(u.id for u in corpus.users)}", file=sys.stderr)
            return 2
        r = Retriever(corpus).retrieve(user, args.question, k=args.k, mode=args.mode)
        ans = make_answerer(args.answerer).answer(args.question, r.hits)
        print(ans.text)
        print(f"\nretrieved: {', '.join(h.chunk.id for h in r.hits) or 'nothing'}")
        print(f"docs filtered out for permission: {r.filtered_docs}")
        return 0

    if args.command == "serve":  # pragma: no cover - starts a server
        import uvicorn

        uvicorn.run("securerag.api:app_from_env", factory=True, host=args.host, port=args.port, log_level="warning")
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

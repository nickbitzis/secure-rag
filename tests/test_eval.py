import pytest

from securerag import cli
from securerag import retrieval as retrieval_mod
from securerag.evaluate import evaluate, load_questions, run_all, table

from .conftest import QUESTIONS


@pytest.fixture(scope="module")
def results(corpus, retriever):
    return {r.mode: r for r in run_all(corpus, retriever, load_questions(QUESTIONS), k=4)}


def test_question_file_size_and_users(corpus):
    qs = load_questions(QUESTIONS)
    assert len(qs) >= 40
    assert all(corpus.user(q.user) for q in qs)
    assert all(q.expected_docs or q.refuse for q in qs)


def test_prefilter_has_zero_leakage_and_full_recall(results):
    pre = results["prefilter"]
    assert pre.leak_rate == 0 and pre.cross_tenant_rate == 0 and pre.leaked_chunks == 0
    assert pre.recall == 1.0
    assert pre.refusal_acc == 1.0


def test_postfilter_is_safe_but_loses_recall(results):
    post = results["postfilter"]
    assert post.leak_rate == 0
    assert post.recall < results["prefilter"].recall


def test_nofilter_leaks_including_across_tenants(results):
    nf = results["nofilter"]
    assert nf.leak_rate > 0.5 and nf.cross_tenant_rate > 0.5 and nf.leaked_chunks > 0


def test_table_has_a_row_per_mode(results):
    out = table(list(results.values()), 4)
    assert "recall@4" in out
    for m in ("prefilter", "postfilter", "nofilter"):
        assert m in out


def test_eval_detects_a_broken_acl_filter(corpus, monkeypatch):
    # Simulate a bug: the ACL check always says yes. The eval checks against the
    # corpus ACLs directly, so it must report the leak.
    monkeypatch.setattr(retrieval_mod, "can_read", lambda user, chunk: True)
    r = evaluate(corpus, retrieval_mod.Retriever(corpus), load_questions(QUESTIONS), "prefilter")
    assert r.leak_rate > 0


def test_tenant_partition_holds_even_if_acl_check_forgets_tenant(corpus, monkeypatch):
    # Defense in depth: each tenant has its own index, so a filter bug that drops
    # the tenant comparison still cannot return another company's chunks.
    monkeypatch.setattr(retrieval_mod, "can_read", lambda user, chunk: bool(set(chunk.acl) & set(user.groups)))
    r = evaluate(corpus, retrieval_mod.Retriever(corpus), load_questions(QUESTIONS), "prefilter")
    assert r.cross_tenant_rate == 0


def test_cli_gate_passes(capsys):
    assert cli.main(["eval", "--questions", str(QUESTIONS), "--gate", "--min-recall", "0.95"]) == 0
    assert "security gate passed" in capsys.readouterr().out


def test_cli_gate_fails_on_leak(monkeypatch, capsys):
    monkeypatch.setattr(retrieval_mod, "can_read", lambda user, chunk: True)
    assert cli.main(["eval", "--questions", str(QUESTIONS), "--gate"]) == 1
    assert "SECURITY GATE FAILED" in capsys.readouterr().err

import json

from securerag.corpus import SENSITIVITY, build_corpus, from_markdown, load_corpus, to_markdown, write_corpus
from securerag.retrieval import can_read


def test_two_tenants_with_about_forty_docs_each(corpus):
    for t in ("hal", "bra"):
        n = sum(1 for d in corpus.docs if d.tenant == t)
        assert 38 <= n <= 46
    assert len({d.id for d in corpus.docs}) == len(corpus.docs)


def test_every_doc_has_acl_and_label(corpus):
    for d in corpus.docs:
        assert d.acl and d.sensitivity in SENSITIVITY
        assert d.id.startswith(d.tenant + "-")


def test_generation_is_deterministic():
    a, b = build_corpus(), build_corpus()
    assert a.docs == b.docs and a.users == b.users


def test_planted_secrets_only_visible_to_hr_legal_exec(corpus):
    secrets = {"salary-bands": {"hr", "exec"}, "acquisition": {"legal", "exec"}, "layoff-memo": {"hr", "exec"}}
    for t in ("hal", "bra"):
        for slug, groups in secrets.items():
            doc = corpus.doc(f"{t}-{slug}")
            assert doc.sensitivity == "restricted"
            readers = {u.id for u in corpus.users if can_read(u, doc)}
            expected = {u.id for u in corpus.users if u.tenant == t and set(u.groups) & groups}
            assert readers == expected and readers


def test_markdown_round_trip(corpus):
    for d in corpus.docs[:10]:
        assert from_markdown(to_markdown(d)) == d


def test_write_and_load_round_trip(corpus, tmp_path):
    n = write_corpus(corpus, tmp_path)
    assert n == len(corpus.docs)
    loaded = load_corpus(tmp_path)
    assert sorted(loaded.docs, key=lambda d: d.id) == sorted(corpus.docs, key=lambda d: d.id)
    assert loaded.users == corpus.users
    assert json.loads((tmp_path / "users.json").read_text())["tenants"] == corpus.tenants


def test_committed_corpus_matches_generator(corpus):
    from .conftest import ROOT

    on_disk = load_corpus(ROOT / "corpus")
    assert sorted(on_disk.docs, key=lambda d: d.id) == sorted(corpus.docs, key=lambda d: d.id)

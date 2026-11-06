import pytest

from securerag.corpus import Doc, User
from securerag.evaluate import load_questions
from securerag.retrieval import BM25, Chunk, Retriever, can_read, chunk_doc, tokenize

from .conftest import QUESTIONS


def _chunk(cid, text, tenant="t", acl=("all",)):
    return Chunk(cid, cid, tenant, acl, "", text)


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("What are the salary bands?") == ["salary", "band"]
    assert tokenize("Process access") == ["process", "access"]


def test_bm25_ranks_the_matching_chunk_first():
    idx = BM25([
        _chunk("a", "The cafeteria serves soup on Fridays."),
        _chunk("b", "Database credentials rotate every 30 days."),
        _chunk("c", "Rotate tires on the delivery vans every winter."),
    ])
    top = idx.search("How often do database credentials rotate?", 3)
    assert idx.chunks[top[0][0]].id == "b"
    assert all(s > 0 for _, s in top)


def test_bm25_rare_terms_weigh_more():
    idx = BM25([_chunk(str(i), "policy update") for i in range(5)] + [_chunk("x", "policy codename")])
    top = idx.search("policy codename", 1)
    assert idx.chunks[top[0][0]].id == "x"


def test_bm25_scores_only_candidates():
    idx = BM25([_chunk("a", "secret codename"), _chunk("b", "lunch menu")])
    assert idx.search("codename", 5, candidates=[1]) == []


def test_chunking_keeps_title_and_respects_size():
    doc = Doc("t-x", "t", "Title", "hr", ("all",), "internal", "\n\n".join(["word " * 40] * 3))
    chunks = chunk_doc(doc, max_words=60)
    assert len(chunks) == 3
    assert all(c.text.startswith("Title. ") and c.acl == ("all",) for c in chunks)
    assert [c.id for c in chunks] == ["t-x#0", "t-x#1", "t-x#2"]


def test_can_read_needs_same_tenant_and_shared_group():
    doc = _chunk("d", "x", tenant="hal", acl=("hr",))
    assert can_read(User("u1", "U", "hal", ("all", "hr")), doc)
    assert not can_read(User("u2", "U", "hal", ("all", "finance")), doc)
    assert not can_read(User("u3", "U", "bra", ("all", "hr")), doc)  # same group name, other tenant


def test_engineer_cannot_retrieve_salary_bands(corpus, retriever):
    alice = corpus.user("hal-alice")
    r = retriever.retrieve(alice, "What is the salary band for L4?")
    assert "hal-salary-bands" not in r.doc_ids
    assert r.hits == []


def test_hr_can_retrieve_salary_bands(corpus, retriever):
    r = retriever.retrieve(corpus.user("hal-ben"), "What is the salary band for L4?")
    assert r.doc_ids[0] == "hal-salary-bands"


def test_tenant_isolation_codename(corpus, retriever):
    # Bramblecote's exec asks about Halvorsen's deal by its codename.
    r = retriever.retrieve(corpus.user("bra-kim"), "What is Project Blue Heron?")
    assert all(h.chunk.tenant == "bra" for h in r.hits)
    assert "hal-acquisition" not in r.doc_ids


def test_prefilter_never_returns_unreadable_chunks_for_any_user(corpus, retriever):
    questions = [q.question for q in load_questions(QUESTIONS)]
    for user in corpus.users:
        for q in questions:
            for h in retriever.retrieve(user, q, k=10).hits:
                assert can_read(user, corpus.doc(h.chunk.doc_id)), (user.id, q, h.chunk.id)


def test_filtered_count_matches_acl(corpus, retriever):
    frank, erin = corpus.user("hal-frank"), corpus.user("hal-erin")
    r_frank = retriever.retrieve(frank, "pto")
    r_erin = retriever.retrieve(erin, "pto")
    restricted_for_frank = sum(1 for d in corpus.docs if d.tenant == "hal" and "all" not in d.acl)
    assert r_frank.filtered_docs == restricted_for_frank
    assert 0 < r_erin.filtered_docs < r_frank.filtered_docs


def test_nofilter_mode_leaks(corpus, retriever):
    r = retriever.retrieve(corpus.user("hal-alice"), "What is the salary band for L4?", mode="nofilter")
    assert any(not can_read(corpus.user("hal-alice"), h.chunk) for h in r.hits)


def test_postfilter_mode_loses_an_allowed_doc(corpus, retriever):
    frank = corpus.user("hal-frank")
    q = "How many unused PTO days carry over to next year?"
    assert "hal-pto-policy" in retriever.retrieve(frank, q, mode="prefilter").doc_ids
    assert "hal-pto-policy" not in retriever.retrieve(frank, q, mode="postfilter").doc_ids


def test_unknown_mode_rejected(corpus, retriever):
    with pytest.raises(ValueError):
        retriever.retrieve(corpus.user("hal-frank"), "x", mode="trust-the-model")


def test_unknown_tenant_gets_nothing(corpus):
    r = Retriever(corpus).retrieve(User("x", "X", "zzz", ("all",)), "PTO carry over")
    assert r.hits == []

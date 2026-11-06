import json

import pytest
from fastapi.testclient import TestClient

from securerag.answer import ClaudeAnswerer
from securerag.api import create_app
from securerag.audit import query_hash
from securerag.identity import sign

from .conftest import FakeClient

SECRET = "test-secret"


@pytest.fixture
def app(corpus):
    return create_app(corpus=corpus, secret=SECRET, demo_tokens=True)


@pytest.fixture
def client(app):
    return TestClient(app)


def auth(user_id):
    return {"Authorization": f"Bearer {sign(user_id, SECRET)}"}


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["docs"] == 88


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Ask as" in r.text


def test_ask_requires_token(client):
    assert client.post("/ask", json={"question": "PTO?"}).status_code == 401
    forged = "hal-erin." + sign("hal-frank", SECRET).split(".", 1)[1]
    r = client.post("/ask", json={"question": "PTO?"}, headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


def test_ask_answers_with_citations(client):
    r = client.post("/ask", json={"question": "How many unused PTO days carry over to next year?"}, headers=auth("hal-frank"))
    body = r.json()
    assert r.status_code == 200 and not body["refused"]
    assert body["citations"][0]["doc_id"] == "hal-pto-policy"
    assert body["filtered_out_docs"] > 0
    assert body["company"] == "Halvorsen Robotics"


def test_ask_refuses_restricted_question(client):
    r = client.post("/ask", json={"question": "What is the salary band for L4?"}, headers=auth("hal-alice"))
    body = r.json()
    assert body["refused"] and body["citations"] == [] and body["retrieved"] == []
    assert "118,000" not in json.dumps(body)


def test_same_question_different_users(client):
    q = {"question": "What is the codename for the planned acquisition?"}
    exec_ = client.post("/ask", json=q, headers=auth("hal-erin")).json()
    eng = client.post("/ask", json=q, headers=auth("hal-alice")).json()
    assert "BLUE HERON" in exec_["answer"]
    assert eng["refused"] and "BLUE HERON" not in json.dumps(eng)


def test_response_never_lists_filtered_titles(client):
    body = client.post("/ask", json={"question": "PTO carry over"}, headers=auth("hal-frank")).json()
    text = json.dumps(body)
    assert "Salary bands" not in text and "Workforce reduction" not in text
    assert isinstance(body["filtered_out_docs"], int)


def test_audit_log_stores_hash_and_doc_ids_not_question(client, app):
    q = "How many weeks of paid parental leave do parents get?"
    client.post("/ask", json={"question": q}, headers=auth("bra-leo"))
    row = app.state.audit.rows(1)[0]
    assert row["user_id"] == "bra-leo" and row["tenant"] == "bra"
    assert row["query_sha256"] == query_hash(q)
    assert "bra-parental-leave" in row["doc_ids"]
    assert q not in json.dumps(row)


def test_metrics_endpoint(client):
    client.post("/ask", json={"question": "PTO carry over"}, headers=auth("hal-frank"))
    client.post("/ask", json={"question": "PTO"})
    text = client.get("/metrics").text
    assert 'securerag_asks_total{outcome="answered",tenant="hal"} 1.0' in text
    assert "securerag_auth_failures_total 1.0" in text
    assert "securerag_ask_seconds_bucket" in text


def test_users_endpoint_and_demo_switch(corpus):
    on = TestClient(create_app(corpus=corpus, secret=SECRET, demo_tokens=True)).get("/api/users").json()
    assert len(on) == 12 and on[0]["token"] == sign(on[0]["id"], SECRET)
    off = TestClient(create_app(corpus=corpus, secret=SECRET, demo_tokens=False)).get("/api/users")
    assert off.status_code == 404


def test_claude_answerer_through_api_only_sees_allowed_chunks(corpus):
    fake = FakeClient("Five days [hal-pto-policy#0].")
    c = TestClient(create_app(corpus=corpus, secret=SECRET, answerer=ClaudeAnswerer(fake)))
    body = c.post("/ask", json={"question": "How many unused PTO days carry over to next year?"}, headers=auth("hal-frank")).json()
    assert body["answerer"] == "claude" and body["citations"][0]["chunk_id"] == "hal-pto-policy#0"
    sent = fake.requests[0]["messages"][0]["content"]
    assert "pto-liability" not in sent  # finance-only doc never reaches the model


def test_bad_request_rejected(client):
    assert client.post("/ask", json={"question": ""}, headers=auth("hal-frank")).status_code == 422

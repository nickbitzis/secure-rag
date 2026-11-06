from pathlib import Path
from types import SimpleNamespace

import pytest

from securerag.corpus import build_corpus
from securerag.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "examples" / "eval.jsonl"


@pytest.fixture(scope="session")
def corpus():
    return build_corpus()


@pytest.fixture(scope="session")
def retriever(corpus):
    return Retriever(corpus)


class FakeClient:
    """Stands in for anthropic.Anthropic(). Records every request, returns a fixed reply."""

    def __init__(self, reply="Five days carry over [hal-pto-policy#0]."):
        self.reply = reply
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=120, output_tokens=30),
        )


@pytest.fixture
def fake_client():
    return FakeClient()

"""Shared fixtures. Tests use an invented RBI-style page, a fake embedder and a fake LLM,
so they need no real documents, no model download and no API key."""

import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grounded.chunking import build_chunks
from grounded.ingest import parse_html

FIXTURE = Path(__file__).parent / "fixtures" / "sample_rbi.html"


@pytest.fixture(scope="session")
def blocks():
    return [asdict(b) for b in parse_html(FIXTURE.read_bytes(), "sample")]


@pytest.fixture(scope="session")
def clause_chunks(blocks):
    return build_chunks("clause", blocks)


class FakeEmbedder:
    """Bag-of-words hashed into 64 dimensions: crude, but texts sharing words get similar vectors."""

    def _vec(self, text):
        v = np.zeros(64)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


class FakeLLM:
    """Returns pre-scripted tool calls (grounded mode) or text (naive mode), one per call."""

    def __init__(self, scripted):
        self.scripted, self.calls, self.model = list(scripted), [], "fake"

    def create(self, system, messages, tools=None, max_tokens=800):
        self.calls.append(messages)
        item = self.scripted.pop(0)
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        if isinstance(item, str):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=item)], usage=usage)
        block = SimpleNamespace(type="tool_use", id=f"t{len(self.calls)}", input=item,
                                model_dump=lambda **kw: {"type": "tool_use", "id": "t", "name": "submit_answer",
                                                         "input": item})
        return SimpleNamespace(content=[block], usage=usage)

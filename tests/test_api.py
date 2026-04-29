"""Unit tests for the FastAPI backend (M4).

SentenceTransformer is stubbed globally via conftest.py (numpy ABI mismatch
prevents importing the real package).  All Anthropic calls are mocked locally.
TestClient is synchronous; all test functions are plain def.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from memgraph.kg.extractor import Triple


def _make_anthropic_mock():
    """Return an anthropic.Anthropic mock whose messages.create() returns a fake response."""
    fake_content = SimpleNamespace(text="hello")
    fake_response = SimpleNamespace(content=[fake_content])
    client_mock = MagicMock()
    client_mock.messages.create.return_value = fake_response
    return client_mock


@pytest.fixture
def client():
    """TestClient with Anthropic and extract_triples mocked; ST stub is from conftest."""
    anthropic_mock = _make_anthropic_mock()

    with (
        patch("memgraph.api.main.DB_PATH", ":memory:"),
        patch("anthropic.Anthropic", return_value=anthropic_mock),
        patch("memgraph.api.main.extract_triples", return_value=[]),
    ):
        from memgraph.api.main import app
        with TestClient(app) as c:
            yield c


# ── Test 1 ──────────────────────────────────────────────────────────────────

def test_chat_returns_response(client):
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "triples_extracted" in data
    assert data["triples_extracted"] == 0


# ── Test 2 ──────────────────────────────────────────────────────────────────

def test_chat_with_triples_updates_graph():
    anthropic_mock = _make_anthropic_mock()
    triple = Triple(subject="Alice", predicate="likes", object="pizza")

    with (
        patch("memgraph.api.main.DB_PATH", ":memory:"),
        patch("anthropic.Anthropic", return_value=anthropic_mock),
        patch("memgraph.api.main.extract_triples", return_value=[triple]),
    ):
        import memgraph.api.main as _main

        with TestClient(_main.app) as c:
            chat_resp = c.post("/chat", json={"message": "Alice likes pizza"})
            assert chat_resp.status_code == 200, chat_resp.text
            assert chat_resp.json()["triples_extracted"] == 1

            # Verify graph state was updated
            assert len(_main._state["G"].nodes()) >= 2

            graph_resp = c.get("/graph")
            assert graph_resp.status_code == 200
            assert len(graph_resp.json()["nodes"]) >= 2


# ── Test 3 ──────────────────────────────────────────────────────────────────

def test_get_graph_empty(client):
    resp = client.get("/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"nodes": [], "edges": []}


# ── Test 4 ──────────────────────────────────────────────────────────────────

def test_reset_clears_graph():
    anthropic_mock = _make_anthropic_mock()
    triple = Triple(subject="Bob", predicate="eats", object="sushi")

    with (
        patch("memgraph.api.main.DB_PATH", ":memory:"),
        patch("anthropic.Anthropic", return_value=anthropic_mock),
        patch("memgraph.api.main.extract_triples", return_value=[]),
    ):
        from memgraph.api.main import app, _state
        from memgraph.kg import upsert_triple, load_graph

        with TestClient(app) as c:
            # Seed data directly and force graph reload
            upsert_triple(_state["conn"], triple)
            _state["G"] = load_graph(_state["conn"])

            before = c.get("/graph").json()
            assert len(before["nodes"]) >= 2

            reset_resp = c.post("/reset")
            assert reset_resp.status_code == 200

            after = c.get("/graph").json()
            assert after == {"nodes": [], "edges": []}


# ── Test 5 ──────────────────────────────────────────────────────────────────

def test_reset_clears_history():
    anthropic_mock = _make_anthropic_mock()

    with (
        patch("memgraph.api.main.DB_PATH", ":memory:"),
        patch("anthropic.Anthropic", return_value=anthropic_mock),
        patch("memgraph.api.main.extract_triples", return_value=[]),
    ):
        from memgraph.api.main import app, _state

        with TestClient(app) as c:
            c.post("/chat", json={"message": "first"})
            c.post("/chat", json={"message": "second"})
            assert len(_state["history"]) > 0

            c.post("/reset")
            assert _state["history"] == []


# ── Test 6 ──────────────────────────────────────────────────────────────────

def test_chat_history_trimmed():
    from memgraph.api.main import HISTORY_MAX

    anthropic_mock = _make_anthropic_mock()

    with (
        patch("memgraph.api.main.DB_PATH", ":memory:"),
        patch("anthropic.Anthropic", return_value=anthropic_mock),
        patch("memgraph.api.main.extract_triples", return_value=[]),
    ):
        from memgraph.api.main import app, _state

        with TestClient(app) as c:
            for i in range(HISTORY_MAX + 2):
                c.post("/chat", json={"message": f"msg {i}"})

            assert len(_state["history"]) <= HISTORY_MAX

"""Unit tests for M2: kg-core — extraction + SQLite persistence."""

from unittest.mock import MagicMock

import networkx as nx
import pytest

from memgraph.kg import Triple, extract_triples, init_db, load_graph, upsert_triple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_use_block(triples_data: list[dict]):
    """Return a mock content block that looks like an Anthropic ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"triples": triples_data}
    return block


def _make_mock_client(triples_data: list[dict]):
    """Return a mock Anthropic client whose messages.create returns a tool-use block."""
    response = MagicMock()
    response.content = [_make_tool_use_block(triples_data)]
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------

class TestExtractTriples:
    def test_returns_correct_triples(self):
        raw = [
            {
                "subject": "Alice",
                "predicate": "likes",
                "object": "pizza",
                "confidence": 0.9,
                "subject_type": "person",
                "object_type": "food",
            }
        ]
        client = _make_mock_client(raw)
        triples = extract_triples("Alice likes pizza", client)
        assert len(triples) == 1
        t = triples[0]
        assert t.subject == "Alice"
        assert t.predicate == "likes"
        assert t.object == "pizza"
        assert t.confidence == pytest.approx(0.9)
        assert t.subject_type == "person"

    def test_returns_multiple_triples(self):
        raw = [
            {"subject": "Bob", "predicate": "lives_in", "object": "Berlin"},
            {"subject": "Bob", "predicate": "works_at", "object": "ACME"},
        ]
        client = _make_mock_client(raw)
        triples = extract_triples("Bob lives in Berlin and works at ACME", client)
        assert len(triples) == 2

    def test_empty_triples_list(self):
        client = _make_mock_client([])
        triples = extract_triples("nothing to extract", client)
        assert triples == []

    def test_api_failure_returns_empty(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("API error")
        triples = extract_triples("some text", client)
        assert triples == []

    def test_no_tool_use_block_returns_empty(self):
        """If the model returns a text block instead of tool_use, return []."""
        text_block = MagicMock()
        text_block.type = "text"
        response = MagicMock()
        response.content = [text_block]
        client = MagicMock()
        client.messages.create.return_value = response
        triples = extract_triples("some text", client)
        assert triples == []


# ---------------------------------------------------------------------------
# SQLite round-trip tests
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """Fresh in-memory SQLite connection for each test."""
    c = init_db(":memory:")
    yield c
    c.close()


def _sample_triple(**kwargs) -> Triple:
    defaults = {
        "subject": "Alice",
        "predicate": "likes",
        "object": "pizza",
        "confidence": 0.95,
        "subject_type": "person",
        "object_type": "food",
    }
    defaults.update(kwargs)
    return Triple(**defaults)


class TestInitDb:
    def test_creates_tables(self, conn):
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "nodes" in tables
        assert "edges" in tables


class TestUpsertTriple:
    def test_inserts_nodes_and_edge(self, conn):
        upsert_triple(conn, _sample_triple())
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert node_count == 2
        assert edge_count == 1

    def test_idempotent(self, conn):
        t = _sample_triple()
        upsert_triple(conn, t)
        upsert_triple(conn, t)
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1

    def test_node_id_normalisation(self, conn):
        upsert_triple(conn, _sample_triple(subject="Alice Smith"))
        row = conn.execute(
            "SELECT id FROM nodes WHERE label = 'Alice Smith'"
        ).fetchone()
        assert row is not None
        assert row[0] == "alice_smith"


class TestLoadGraph:
    def test_returns_multidigraph(self, conn):
        upsert_triple(conn, _sample_triple())
        G = load_graph(conn)
        assert isinstance(G, nx.MultiDiGraph)

    def test_correct_node_count(self, conn):
        upsert_triple(conn, _sample_triple())
        upsert_triple(conn, _sample_triple(subject="Bob", object="sushi"))
        G = load_graph(conn)
        # nodes: alice, pizza, bob, sushi = 4
        assert G.number_of_nodes() == 4

    def test_edge_predicate_attribute(self, conn):
        upsert_triple(conn, _sample_triple())
        G = load_graph(conn)
        # edge from alice → pizza with key "likes"
        edge_data = G.get_edge_data("alice", "pizza", key="likes")
        assert edge_data is not None
        assert edge_data["predicate"] == "likes"

    def test_multiple_predicates_between_same_nodes(self, conn):
        upsert_triple(conn, _sample_triple(predicate="likes"))
        upsert_triple(conn, _sample_triple(predicate="loves"))
        G = load_graph(conn)
        assert G.number_of_edges() == 2

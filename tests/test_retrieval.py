"""Unit tests for M3: retrieval engine — BFS traversal + embedding fallback."""

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from memgraph.kg.store import init_db, load_embeddings, store_embedding, upsert_triple
from memgraph.kg.extractor import Triple
from memgraph.retrieval import Retriever
from memgraph.retrieval.engine import (
    bfs_retrieve,
    compute_and_store_embeddings,
    embedding_retrieve,
    retrieve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triple(subject, predicate, object_, subject_type="entity", object_type="entity"):
    return Triple(
        subject=subject,
        predicate=predicate,
        object=object_,
        confidence=1.0,
        subject_type=subject_type,
        object_type=object_type,
    )


@pytest.fixture
def conn():
    c = init_db(":memory:")
    yield c
    c.close()


def _build_graph(conn, triples):
    from memgraph.kg.store import load_graph
    for t in triples:
        upsert_triple(conn, t)
    return load_graph(conn)


# ---------------------------------------------------------------------------
# engine.py function tests (legacy; kept for coverage)
# ---------------------------------------------------------------------------

class TestEngineBfsRetrieve:
    def test_bfs_finds_seed_node(self, conn):
        G = _build_graph(conn, [_triple("alice", "likes", "pizza")])
        result = bfs_retrieve("pizza", G)
        assert "likes" in result

    def test_bfs_traverses_depth_2(self, conn):
        G = _build_graph(conn, [
            _triple("alice", "likes", "pizza"),
            _triple("pizza", "origin", "italy"),
        ])
        result = bfs_retrieve("alice", G, depth=2)
        assert "origin" in result
        assert "italy" in result.lower()

    def test_bfs_no_match_returns_empty(self, conn):
        G = _build_graph(conn, [_triple("alice", "likes", "pizza")])
        result = bfs_retrieve("banana", G)
        assert result == ""

    def test_bfs_empty_graph_returns_empty(self):
        G = nx.MultiDiGraph()
        result = bfs_retrieve("anything", G)
        assert result == ""

    def test_bfs_top_k_cap(self, conn):
        triples = [_triple("alice", f"rel{i}", f"obj{i}") for i in range(30)]
        G = _build_graph(conn, triples)
        result = bfs_retrieve("alice", G, top_k=5)
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 5


class TestStoreLoadEmbeddings:
    def test_store_and_load_embeddings(self, conn):
        upsert_triple(conn, _triple("alice", "likes", "pizza"))
        vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        store_embedding(conn, "alice", vec)
        loaded = load_embeddings(conn)
        assert "alice" in loaded
        np.testing.assert_allclose(loaded["alice"], vec, rtol=1e-5)


class TestEngineEmbeddingRetrieve:
    def test_embedding_retrieve_top_k(self, conn):
        from memgraph.kg.store import load_graph
        upsert_triple(conn, _triple("alice", "likes", "pizza"))
        upsert_triple(conn, _triple("bob", "hates", "sushi"))

        store_embedding(conn, "alice", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        store_embedding(conn, "bob", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        query_enc = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = query_enc

        G = load_graph(conn)
        result = embedding_retrieve("query about alice", conn, G, top_k=1, model=mock_model)
        assert "likes" in result
        assert "hates" not in result

    def test_embedding_retrieve_no_embeddings_returns_empty(self, conn):
        from memgraph.kg.store import load_graph
        upsert_triple(conn, _triple("alice", "likes", "pizza"))
        G = load_graph(conn)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        result = embedding_retrieve("pizza", conn, G, model=mock_model)
        assert result == ""


class TestComputeAndStoreEmbeddings:
    def test_compute_and_store_embeddings_fills_nulls(self, conn):
        upsert_triple(conn, _triple("alice", "likes", "pizza"))

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((2, 4), dtype=np.float32)

        compute_and_store_embeddings(conn, model=mock_model)

        loaded = load_embeddings(conn)
        assert "alice" in loaded
        assert "pizza" in loaded

    def test_compute_noop_when_all_filled(self, conn):
        upsert_triple(conn, _triple("alice", "likes", "pizza"))
        store_embedding(conn, "alice", np.ones(4, dtype=np.float32))
        store_embedding(conn, "pizza", np.ones(4, dtype=np.float32))

        mock_model = MagicMock()
        compute_and_store_embeddings(conn, model=mock_model)
        mock_model.encode.assert_not_called()


class TestEngineRetrieve:
    def test_retrieve_uses_bfs_when_match(self, conn):
        from memgraph.kg.store import load_graph
        upsert_triple(conn, _triple("alice", "likes", "pizza"))
        G = load_graph(conn)

        with patch("memgraph.retrieval.engine._get_model") as mock_get:
            result = retrieve("pizza", conn, G)
            mock_get.assert_not_called()

        assert "likes" in result

    def test_retrieve_falls_back_to_embedding(self, conn):
        from memgraph.kg.store import load_graph
        upsert_triple(conn, _triple("alice", "likes", "pizza"))
        store_embedding(conn, "alice", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        store_embedding(conn, "pizza", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        G = load_graph(conn)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        with patch("memgraph.retrieval.engine._get_model", return_value=mock_model):
            result = retrieve("banana smoothie", conn, G)

        assert "likes" in result


# ---------------------------------------------------------------------------
# Retriever class fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph():
    """Alice --likes--> Pizza --originates_from--> Italy."""
    G = nx.MultiDiGraph()
    G.add_node("alice", label="Alice", entity_type="person")
    G.add_node("pizza", label="Pizza", entity_type="food")
    G.add_node("italy", label="Italy", entity_type="place")
    G.add_edge("alice", "pizza", key="likes", predicate="likes", confidence=1.0)
    G.add_edge("pizza", "italy", key="originates_from", predicate="originates_from", confidence=1.0)
    return G


@pytest.fixture
def mock_model():
    """SentenceTransformer mock returning deterministic float32 vectors."""
    model = MagicMock()
    model.encode.side_effect = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
    return model


@pytest.fixture
def retriever(conn, simple_graph, mock_model):
    return Retriever(conn, simple_graph, model=mock_model)


# ---------------------------------------------------------------------------
# Retriever.bfs_retrieve
# ---------------------------------------------------------------------------

class TestBfsRetrieve:
    def test_finds_seed_and_returns_triples(self, retriever):
        result = retriever.bfs_retrieve("alice")
        assert "Alice" in result
        assert "likes" in result
        assert "Pizza" in result

    def test_no_seed_returns_empty_string(self, retriever):
        result = retriever.bfs_retrieve("xyznotingraph")
        assert result == ""

    def test_depth_limits_traversal(self, conn, mock_model):
        """depth=1 from 'source' should not yield far-node (2 hops away)."""
        G = nx.MultiDiGraph()
        G.add_node("source", label="Source", entity_type="entity")
        G.add_node("middle", label="Middle", entity_type="entity")
        G.add_node("far", label="Far", entity_type="entity")
        G.add_edge("source", "middle", key="r1", predicate="r1", confidence=1.0)
        G.add_edge("middle", "far", key="r2", predicate="r2", confidence=1.0)
        r = Retriever(conn, G, model=mock_model)
        result = r.bfs_retrieve("source", depth=1)
        assert "Middle" in result
        assert "Far" not in result

    def test_max_nodes_caps_output(self, conn, mock_model):
        """max_nodes=2 produces fewer triples than max_nodes=20."""
        G = nx.MultiDiGraph()
        G.add_node("hub", label="Hub", entity_type="entity")
        for i in range(10):
            G.add_node(f"leaf_{i}", label=f"Leaf{i}", entity_type="entity")
            G.add_edge("hub", f"leaf_{i}", key=f"r{i}", predicate="connects", confidence=1.0)
        r = Retriever(conn, G, model=mock_model)
        full_lines = [l for l in r.bfs_retrieve("Hub", max_nodes=20).split("\n") if l]
        capped_lines = [l for l in r.bfs_retrieve("Hub", max_nodes=2).split("\n") if l]
        assert len(capped_lines) < len(full_lines)


# ---------------------------------------------------------------------------
# Retriever.embedding_retrieve
# ---------------------------------------------------------------------------

class TestEmbeddingRetrieve:
    def test_returns_top_k_nodes(self, conn, simple_graph, mock_model):
        """After indexing, top_k=3 retrieves all nodes and their edges."""
        r = Retriever(conn, simple_graph, model=mock_model)
        r.index_graph()
        result = r.embedding_retrieve("food", top_k=3)
        assert "likes" in result

    def test_empty_db_returns_empty_string(self, conn, simple_graph, mock_model):
        r = Retriever(conn, simple_graph, model=mock_model)
        result = r.embedding_retrieve("food")
        assert result == ""


# ---------------------------------------------------------------------------
# Retriever.index_graph
# ---------------------------------------------------------------------------

class TestIndexGraph:
    def test_stores_embedding_for_each_node(self, conn, simple_graph, mock_model):
        r = Retriever(conn, simple_graph, model=mock_model)
        r.index_graph()
        row_count = conn.execute("SELECT count(*) FROM node_embeddings").fetchone()[0]
        assert row_count == simple_graph.number_of_nodes()

    def test_skips_already_indexed_nodes(self, conn, simple_graph, mock_model):
        r = Retriever(conn, simple_graph, model=mock_model)
        r.index_graph()
        call_count_after_first = mock_model.encode.call_count
        r.index_graph()
        assert mock_model.encode.call_count == call_count_after_first


# ---------------------------------------------------------------------------
# Retriever.retrieve (unified)
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_bfs_result_returned_when_nonempty(self, retriever):
        result = retriever.retrieve("alice")
        assert result != ""
        assert "Alice" in result

    def test_falls_back_to_embedding_when_bfs_empty(self, conn, simple_graph, mock_model):
        r = Retriever(conn, simple_graph, model=mock_model)
        r.index_graph()
        result = r.retrieve("xyznotingraph")
        # BFS finds nothing; embedding fallback finds all nodes → non-empty
        assert result != ""

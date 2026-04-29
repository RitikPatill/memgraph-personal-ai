"""Unified retrieval engine: BFS graph traversal with embedding fallback."""

import sqlite3

import networkx as nx
import numpy as np

from memgraph.kg.store import load_embeddings, store_embedding
from memgraph.retrieval.traversal import bfs_context

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Module-level singleton. Loaded once, reused."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def bfs_retrieve(query: str, G: nx.MultiDiGraph, depth: int = 2, top_k: int = 20) -> str:
    """Keyword-anchored BFS over the graph; returns edge triples as a context string."""
    lines = bfs_context(query, G, max_hops=depth, max_nodes=max(top_k * 2, 40))
    return "\n".join(lines[:top_k])


def compute_and_store_embeddings(
    conn: sqlite3.Connection,
    model=None,
) -> None:
    """Encode nodes that lack embeddings and persist them to the DB."""
    rows = conn.execute(
        "SELECT id, label, entity_type FROM nodes WHERE embedding IS NULL"
    ).fetchall()
    if not rows:
        return

    if model is None:
        model = _get_model()

    node_ids = [row[0] for row in rows]
    descriptions = [f"{row[1]} ({row[2]})" for row in rows]
    vectors = model.encode(descriptions)

    for node_id, vector in zip(node_ids, vectors):
        store_embedding(conn, node_id, np.array(vector, dtype=np.float32))


def embedding_retrieve(
    query: str,
    conn: sqlite3.Connection,
    G: nx.MultiDiGraph,
    top_k: int = 5,
    model=None,
) -> str:
    """Cosine similarity over stored node embeddings; returns context string."""
    embeddings = load_embeddings(conn)
    if not embeddings:
        return ""

    if model is None:
        model = _get_model()

    query_vec = np.array(model.encode([query])[0], dtype=np.float32)

    node_ids = list(embeddings.keys())
    matrix = np.stack([embeddings[nid] for nid in node_ids])  # (N, D)

    sims = matrix @ query_vec / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec) + 1e-9
    )
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_nodes = {node_ids[i] for i in top_indices}

    lines: list[str] = []
    seen: set[str] = set()
    for src, tgt, data in G.edges(data=True):
        if src in top_nodes:
            src_label = G.nodes[src].get("label", src)
            tgt_label = G.nodes[tgt].get("label", tgt)
            predicate = data.get("predicate", "")
            line = f"{src_label} {predicate} {tgt_label}"
            if line not in seen:
                seen.add(line)
                lines.append(line)

    return "\n".join(lines)


def retrieve(
    query: str,
    conn: sqlite3.Connection,
    G: nx.MultiDiGraph,
    top_k: int = 20,
) -> str:
    """Unified interface: BFS first, fall back to embedding similarity."""
    result = bfs_retrieve(query, G, top_k=top_k)
    if result:
        return result
    return embedding_retrieve(query, conn, G, top_k=min(top_k, 5))

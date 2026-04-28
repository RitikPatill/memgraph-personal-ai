"""SQLite-backed node embedding store and cosine similarity search."""

import sqlite3

import numpy as np

_CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id   TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
"""


def ensure_embeddings_table(conn: sqlite3.Connection) -> None:
    """Create node_embeddings table if it doesn't exist."""
    conn.execute(_CREATE_EMBEDDINGS)
    conn.commit()


def upsert_node_embedding(
    conn: sqlite3.Connection,
    node_id: str,
    embedding: np.ndarray,
) -> None:
    """Insert or replace a node's embedding (serialised as raw float32 bytes)."""
    blob = embedding.astype(np.float32).tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO node_embeddings (node_id, embedding) VALUES (?, ?)",
        (node_id, blob),
    )
    conn.commit()


def get_all_embeddings(
    conn: sqlite3.Connection,
) -> list[tuple[str, np.ndarray]]:
    """Return [(node_id, embedding_array), ...] for all stored rows."""
    rows = conn.execute(
        "SELECT node_id, embedding FROM node_embeddings"
    ).fetchall()
    return [
        (row[0], np.frombuffer(bytes(row[1]), dtype=np.float32))
        for row in rows
    ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def embedding_search(
    query_embedding: np.ndarray,
    conn: sqlite3.Connection,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return [(node_id, score), ...] sorted by cosine similarity descending."""
    rows = get_all_embeddings(conn)
    scored = [
        (node_id, cosine_similarity(query_embedding, emb))
        for node_id, emb in rows
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

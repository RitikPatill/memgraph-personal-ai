"""SQLite-backed persistence and NetworkX graph loader for the knowledge graph."""

import sqlite3
from datetime import datetime

import networkx as nx
import numpy as np

from .extractor import Triple

_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'entity'
);
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL REFERENCES nodes(id),
    predicate   TEXT NOT NULL,
    target      TEXT NOT NULL REFERENCES nodes(id),
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL
);
"""

_CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id   TEXT PRIMARY KEY REFERENCES nodes(id),
    embedding BLOB NOT NULL
);
"""


def _node_id(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _edge_id(src: str, pred: str, tgt: str) -> str:
    return f"{src}|{pred.strip().lower()}|{tgt}"


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite DB, run CREATE TABLE IF NOT EXISTS, return conn."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_NODES)
    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN embedding BLOB")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.execute(_CREATE_EDGES)
    conn.execute(_CREATE_EMBEDDINGS)
    conn.commit()
    return conn


def upsert_triple(conn: sqlite3.Connection, triple: Triple) -> None:
    """Insert subject node, object node, and edge. All use INSERT OR IGNORE."""
    src_id = _node_id(triple.subject)
    tgt_id = _node_id(triple.object)
    edge_id = _edge_id(src_id, triple.predicate, tgt_id)
    created_at = datetime.utcnow().isoformat()

    conn.execute(
        "INSERT OR IGNORE INTO nodes (id, label, entity_type) VALUES (?, ?, ?)",
        (src_id, triple.subject, triple.subject_type),
    )
    conn.execute(
        "INSERT OR IGNORE INTO nodes (id, label, entity_type) VALUES (?, ?, ?)",
        (tgt_id, triple.object, triple.object_type),
    )
    conn.execute(
        "INSERT OR IGNORE INTO edges (id, source, predicate, target, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (edge_id, src_id, triple.predicate, tgt_id, triple.confidence, created_at),
    )
    conn.commit()


def store_embedding(conn: sqlite3.Connection, node_id: str, vector: np.ndarray) -> None:
    """Serialise vector as float32 bytes and store in nodes.embedding."""
    blob = vector.astype(np.float32).tobytes()
    conn.execute("UPDATE nodes SET embedding = ? WHERE id = ?", (blob, node_id))
    conn.commit()


def load_embeddings(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    """Return {node_id: vector} for all nodes with non-null embeddings."""
    rows = conn.execute(
        "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
    ).fetchall()
    return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in rows}


def load_graph(conn: sqlite3.Connection) -> nx.MultiDiGraph:
    """Read all rows from nodes + edges tables, return populated MultiDiGraph."""
    G = nx.MultiDiGraph()

    for row in conn.execute("SELECT id, label, entity_type FROM nodes"):
        G.add_node(row["id"], label=row["label"], entity_type=row["entity_type"])

    for row in conn.execute(
        "SELECT source, predicate, target, confidence, created_at FROM edges"
    ):
        G.add_edge(
            row["source"],
            row["target"],
            key=row["predicate"],
            predicate=row["predicate"],
            confidence=row["confidence"],
            created_at=row["created_at"],
        )

    return G


def upsert_node_embedding(conn: sqlite3.Connection, node_id: str, embedding_bytes: bytes) -> None:
    """INSERT OR REPLACE a serialized numpy embedding for node_id."""
    conn.execute(
        "INSERT OR REPLACE INTO node_embeddings (node_id, embedding) VALUES (?, ?)",
        (node_id, embedding_bytes),
    )
    conn.commit()


def load_all_embeddings(conn: sqlite3.Connection) -> dict[str, bytes]:
    """Return {node_id: embedding_bytes} for all stored embeddings."""
    rows = conn.execute("SELECT node_id, embedding FROM node_embeddings").fetchall()
    return {row[0]: bytes(row[1]) for row in rows}

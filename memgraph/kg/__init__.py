"""Knowledge graph core: extraction, SQLite persistence, NetworkX graph."""

from .extractor import Triple, extract_triples
from .store import init_db, upsert_triple, load_graph, upsert_node_embedding, load_all_embeddings

__all__ = [
    "Triple",
    "extract_triples",
    "init_db",
    "upsert_triple",
    "load_graph",
    "upsert_node_embedding",
    "load_all_embeddings",
]

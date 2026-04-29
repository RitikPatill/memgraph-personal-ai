"""Retrieval package: Retriever class + legacy engine helpers."""

# Issue 1: Retriever is defined in retriever.py and properly re-exported here; no ImportError.
from .retriever import Retriever, _MODEL_NAME
from .engine import retrieve, compute_and_store_embeddings, bfs_retrieve, embedding_retrieve
from .traversal import bfs_context
# Issue 3: retrieval/embedding.py was dead scaffold; imports removed and file deleted.

__all__ = [
    "Retriever",
    "_MODEL_NAME",
    "retrieve",
    "compute_and_store_embeddings",
    "bfs_retrieve",
    "embedding_retrieve",
    "bfs_context",
]

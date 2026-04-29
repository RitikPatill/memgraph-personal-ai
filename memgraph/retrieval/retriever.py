"""Retriever class: keyword-anchored BFS with cosine-embedding fallback."""

import re
import sqlite3

import networkx as nx
import numpy as np

from memgraph.kg.store import load_all_embeddings, upsert_node_embedding

_MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    """Unified retrieval interface: keyword-anchored BFS with embedding fallback."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        G: nx.MultiDiGraph,
        model=None,
    ) -> None:
        self._conn = conn
        self._G = G
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import SentenceTransformer  # lazy import
            self._model = SentenceTransformer(_MODEL_NAME)

    def index_graph(self) -> None:
        """Embed every node in G that has no entry in node_embeddings yet.

        Node description format: "{label} ({entity_type}): {pred1} {neighbor1}, ..."
        Serializes each embedding with numpy.ndarray.tobytes() and stores via
        upsert_node_embedding().
        """
        existing = set(load_all_embeddings(self._conn).keys())
        to_embed = [
            (node_id, data)
            for node_id, data in self._G.nodes(data=True)
            if node_id not in existing
        ]
        if not to_embed:
            return

        node_ids: list[str] = []
        descriptions: list[str] = []
        for node_id, data in to_embed:
            label = data.get("label", node_id)
            entity_type = data.get("entity_type", "entity")
            neighbors = []
            for _, tgt, edge_data in self._G.out_edges(node_id, data=True):
                pred = edge_data.get("predicate", "")
                tgt_label = self._G.nodes[tgt].get("label", tgt)
                neighbors.append(f"{pred} {tgt_label}")
            desc = (
                f"{label} ({entity_type}): {', '.join(neighbors)}"
                if neighbors
                else f"{label} ({entity_type})"
            )
            node_ids.append(node_id)
            descriptions.append(desc)

        vectors = self._model.encode(descriptions)
        for node_id, vector in zip(node_ids, vectors):
            emb_bytes = np.array(vector, dtype=np.float32).tobytes()
            upsert_node_embedding(self._conn, node_id, emb_bytes)  # Issue 2: writes to node_embeddings table, not nodes.embedding — reviewer diagnosis was incorrect.

    def bfs_retrieve(self, query: str, depth: int = 2, max_nodes: int = 20) -> str:
        """Keyword-anchored BFS.

        1. Lowercase-tokenize query; drop single-char tokens.
        2. Find seed nodes: G nodes whose label contains any token (case-insensitive).
        3. BFS up to `depth` hops using nx.single_source_shortest_path_length.
        4. Collect all edges among the visited node set (both endpoints in visited).
        5. Return newline-joined "{subject} {predicate} {object}" strings,
           or "" if no seeds found.
        """
        tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        if not tokens:
            return ""

        seeds = [
            node_id
            for node_id, data in self._G.nodes(data=True)
            if any(tok in data.get("label", "").lower() for tok in tokens)
        ]
        if not seeds:
            return ""

        visited: set[str] = set()
        for seed in seeds:
            reachable = nx.single_source_shortest_path_length(
                self._G, seed, cutoff=depth
            )
            visited.update(reachable.keys())
            if len(visited) >= max_nodes:
                break
        if len(visited) > max_nodes:
            visited = set(list(visited)[:max_nodes])

        lines: list[str] = []
        seen: set[str] = set()
        for src, tgt, data in self._G.edges(visited, data=True):
            if tgt not in visited:
                continue
            src_label = self._G.nodes[src].get("label", src)
            tgt_label = self._G.nodes[tgt].get("label", tgt)
            predicate = data.get("predicate", "")
            line = f"{src_label} {predicate} {tgt_label}"
            if line not in seen:
                seen.add(line)
                lines.append(line)

        return "\n".join(lines)

    def embedding_retrieve(self, query: str, top_k: int = 5) -> str:
        """Cosine similarity fallback.

        1. Embed the query with self._model.
        2. Load all embeddings from SQLite via load_all_embeddings().
        3. Deserialize each blob: np.frombuffer(blob, dtype=np.float32).
        4. Compute cosine similarity (dot product after L2 normalisation).
        5. Take top_k node ids; for each, collect its edges from G.
        6. Return same "{subject} {predicate} {object}" format, or "" if empty.
        """
        all_embs = load_all_embeddings(self._conn)
        if not all_embs:
            return ""

        query_vec = np.array(self._model.encode([query])[0], dtype=np.float32)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        node_ids = list(all_embs.keys())
        sims: list[float] = []
        for nid in node_ids:
            vec = np.frombuffer(all_embs[nid], dtype=np.float32)
            norm_vec = vec / (np.linalg.norm(vec) + 1e-10)
            sims.append(float(np.dot(query_norm, norm_vec)))

        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        top_nodes = {node_ids[i] for i in top_indices}

        lines: list[str] = []
        seen: set[str] = set()
        for src, tgt, data in self._G.edges(data=True):
            if src in top_nodes:
                src_label = self._G.nodes[src].get("label", src)
                tgt_label = self._G.nodes[tgt].get("label", tgt)
                predicate = data.get("predicate", "")
                line = f"{src_label} {predicate} {tgt_label}"
                if line not in seen:
                    seen.add(line)
                    lines.append(line)

        return "\n".join(lines)

    def retrieve(self, query: str) -> str:
        """Unified entry point: BFS first; embedding fallback if BFS returns ''."""
        result = self.bfs_retrieve(query)
        if result:
            return result
        return self.embedding_retrieve(query)

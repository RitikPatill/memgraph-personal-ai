"""Keyword-anchored BFS traversal over the NetworkX knowledge graph."""

import re
from collections import deque

import networkx as nx

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "then", "once", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "what", "which", "who",
    "this", "that", "these", "those", "i", "me", "my", "myself", "we",
    "our", "you", "your", "he", "she", "it", "they", "them", "their",
}


def _extract_keywords(text: str) -> list[str]:
    """Tokenise text and remove stopwords."""
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def bfs_context(
    query: str,
    G: nx.MultiDiGraph,
    max_hops: int = 2,
    max_nodes: int = 20,
) -> list[str]:
    """BFS over G anchored on nodes whose labels contain query keywords.

    Traversal is bidirectional (follows both successors and predecessors).
    Returns edge-formatted lines: ``"src --[pred]--> tgt (conf=X.XX)"``.
    At most *max_nodes* graph nodes are visited; caller slices for top-k lines.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return []

    seeds = [
        node_id
        for node_id, data in G.nodes(data=True)
        if any(kw in data.get("label", "").lower() for kw in keywords)
    ]
    if not seeds:
        return []

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    for seed in seeds:
        if seed not in visited:
            visited.add(seed)
            queue.append((seed, 0))

    while queue and len(visited) < max_nodes:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for nbr in list(G.successors(node)) + list(G.predecessors(node)):
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, depth + 1))
                if len(visited) >= max_nodes:
                    break

    lines: list[str] = []
    seen: set[str] = set()
    for src, tgt, data in G.edges(data=True):
        if src in visited and tgt in visited:
            src_label = G.nodes[src].get("label", src)
            tgt_label = G.nodes[tgt].get("label", tgt)
            predicate = data.get("predicate", "")
            conf = data.get("confidence", 1.0)
            line = f"{src_label} --[{predicate}]--> {tgt_label} (conf={conf:.2f})"
            if line not in seen:
                seen.add(line)
                lines.append(line)

    return lines

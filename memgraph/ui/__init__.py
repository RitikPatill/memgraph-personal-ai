"""Streamlit UI helpers — pyvis knowledge-graph visualization."""

ENTITY_TYPE_COLORS: dict[str, str] = {
    "person":     "#4CAF50",   # green
    "place":      "#2196F3",   # blue
    "preference": "#FF9800",   # orange
    "fact":       "#9C27B0",   # purple
    "event":      "#F44336",   # red
    "entity":     "#607D8B",   # grey (default)
}


def build_pyvis_html(nodes: list[dict], edges: list[dict]) -> str:
    """Return a self-contained pyvis HTML string for the given graph data.

    Args:
        nodes: list of dicts with keys id, label, entity_type
        edges: list of dicts with keys source, target, predicate, confidence
    Returns:
        HTML string suitable for st.components.v1.html().
        Note: the returned HTML loads vis.js from CDN — an internet connection
        is required for the graph to render.
    """
    from pyvis.network import Network

    net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut()
    for node in nodes:
        color = ENTITY_TYPE_COLORS.get(
            node.get("entity_type", "entity").lower(), "#607D8B"
        )
        net.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            color=color,
            title=f"[{node.get('entity_type', 'entity')}] {node.get('label', node['id'])}",
        )
    for edge in edges:
        net.add_edge(
            edge["source"],
            edge["target"],
            label=edge.get("predicate", ""),
            title=edge.get("predicate", ""),
        )
    return net.generate_html()

"""Unit tests for memgraph.ui helpers (no Streamlit runtime, no live server)."""

from memgraph.ui import ENTITY_TYPE_COLORS, build_pyvis_html


def test_build_pyvis_html_empty():
    html = build_pyvis_html([], [])
    assert isinstance(html, str)
    assert "<html" in html.lower()


def test_build_pyvis_html_with_person_node():
    nodes = [{"id": "alice", "label": "Alice", "entity_type": "person"}]
    html = build_pyvis_html(nodes, [])
    assert "#4CAF50" in html


def test_build_pyvis_html_with_edge():
    nodes = [
        {"id": "alice", "label": "Alice", "entity_type": "person"},
        {"id": "italian_food", "label": "Italian Food", "entity_type": "preference"},
    ]
    edges = [
        {
            "source": "alice",
            "target": "italian_food",
            "predicate": "likes",
            "confidence": 0.9,
        }
    ]
    html = build_pyvis_html(nodes, edges)
    assert "likes" in html


def test_entity_type_colors_has_all_types():
    expected = {"person", "place", "preference", "fact", "event", "entity"}
    assert expected == set(ENTITY_TYPE_COLORS.keys())

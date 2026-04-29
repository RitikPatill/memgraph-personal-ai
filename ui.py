"""MemGraph Streamlit UI — single entry point.

Run with:
    streamlit run ui.py

Starts the FastAPI backend as a background subprocess on first launch
(skipped if port 8000 is already open), then presents a two-column layout:
chat panel (left) and live pyvis knowledge-graph visualization (right).
"""

import atexit
import os
import socket
import subprocess
import sys
import time

import httpx
import streamlit as st
import streamlit.components.v1 as components

from memgraph.ui import ENTITY_TYPE_COLORS, build_pyvis_html

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND_URL: str = os.environ.get("MEMGRAPH_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Backend lifecycle helpers
# ---------------------------------------------------------------------------


def _is_port_open(port: int) -> bool:
    """Return True if something is already listening on the given port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(("127.0.0.1", port))
    s.close()
    return result == 0


def _shutdown(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    proc.terminate()


def _start_backend() -> None:
    """Launch the FastAPI backend as a subprocess if port 8000 is closed."""
    if _is_port_open(8000):
        return  # Already up (pre-started manually or by a previous tab)

    if "_backend_proc" in st.session_state:
        proc = st.session_state["_backend_proc"]
        if proc.poll() is None:
            return  # We launched it earlier in this session and it's still running

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "memgraph.api.main:app",
            "--port",
            "8000",
            "--log-level",
            "error",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state["_backend_proc"] = proc
    atexit.register(_shutdown, proc)

    # Poll /graph until the backend is ready (up to 20 × 0.5 s = 10 s)
    for _ in range(20):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{BACKEND_URL}/graph", timeout=2.0)
            if r.status_code == 200:
                break
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def send_message(message: str) -> dict:
    """POST /chat and return the response JSON."""
    r = httpx.post(f"{BACKEND_URL}/chat", json={"message": message}, timeout=60.0)
    r.raise_for_status()
    return r.json()


def get_graph() -> dict:
    """GET /graph and return the response JSON."""
    r = httpx.get(f"{BACKEND_URL}/graph", timeout=5.0)
    r.raise_for_status()
    return r.json()


def do_reset() -> None:
    """POST /reset."""
    httpx.post(f"{BACKEND_URL}/reset", timeout=5.0)


# ---------------------------------------------------------------------------
# Startup — idempotent; the port check prevents double-launching.
# ---------------------------------------------------------------------------

_start_backend()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="MemGraph")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

col_chat, col_graph = st.columns([1, 1])

# ---------------------------------------------------------------------------
# Chat column
# ---------------------------------------------------------------------------

with col_chat:
    st.subheader("Chat")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.button("Reset"):
        try:
            do_reset()
        except Exception:
            st.error("Reset failed — backend unreachable.")
        else:
            st.session_state["messages"] = []
            st.rerun()

# ---------------------------------------------------------------------------
# Graph column
# ---------------------------------------------------------------------------

with col_graph:
    st.subheader("Knowledge Graph")

    # Color legend
    legend_parts = [
        f'<span style="color:{color}">&#9632;</span> {etype}'
        for etype, color in ENTITY_TYPE_COLORS.items()
        if etype != "entity"
    ]
    legend_parts.append(
        f'<span style="color:{ENTITY_TYPE_COLORS[\"entity\"]}\">&#9632;</span> other'
    )
    st.markdown("**Legend:** " + " | ".join(legend_parts), unsafe_allow_html=True)

    try:
        graph_data = get_graph()
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
    except Exception:
        nodes, edges = [], []
        st.warning("Could not fetch graph data.")

    if nodes:
        # Note: pyvis generate_html() loads vis.js from CDN — internet required.
        html = build_pyvis_html(nodes, edges)
        components.html(html, height=620, scrolling=False)
    else:
        st.info("No nodes yet — start chatting!")

# ---------------------------------------------------------------------------
# Chat input — must be at top-level scope for correct bottom anchoring
# ---------------------------------------------------------------------------

prompt = st.chat_input("Type a message…")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    try:
        result = send_message(prompt)
        reply = result["response"]
    except Exception as exc:
        reply = f"[Error: {exc}]"

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.rerun()

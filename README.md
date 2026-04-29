# MemGraph: Personal AI Assistant with Knowledge Graph Memory

A local personal AI assistant that progressively builds a typed knowledge graph from your conversations. MemGraph extracts structured entities (people, places, preferences, facts, events) and relations between them from every message, persists them as a graph, and traverses that graph to gather relevant personal context before generating responses.

## What works

### M1 — Scaffold (complete)
- Package skeleton under the `memgraph` namespace with pinned dependencies.
- Importable version string: `from memgraph import __version__`.

### M2 — KG Core (complete)
- **Triple model** (`memgraph.kg.Triple`): Pydantic model with `subject`, `predicate`, `object`, `confidence`, `subject_type`, and `object_type` fields.
- **LLM extractor** (`extract_triples`): sends conversation text to `claude-haiku-4-5` via the Anthropic tool-use API; returns a validated `list[Triple]` and degrades gracefully to an empty list on any API or parse failure.
- **SQLite persistence** (`init_db`, `upsert_triple`): `nodes` and `edges` tables created on first open; upserts are idempotent via `INSERT OR IGNORE`; node IDs are normalized to lowercase with underscores.
- **Graph loader** (`load_graph`): reconstructs a `networkx.MultiDiGraph` from the stored tables, preserving predicate labels and confidence scores as edge attributes.
- **Unit tests** (`tests/test_kg.py`): 13 tests covering extractor responses, API-failure degradation, SQLite schema creation, upsert idempotency, node-ID normalization, and graph round-trips. No live API key required — the Anthropic client is fully mocked.

### M3 — Retrieval Engine (complete)
- **`Retriever` class** (`memgraph.retrieval.Retriever`): injectable constructor `(conn, G, model=None)` with four methods below. `SentenceTransformer` is lazily imported so injected mocks never trigger a model download.
- **BFS retrieval** (`Retriever.bfs_retrieve`): lowercase-tokenises the query, drops single-char tokens, finds keyword-matching seed nodes by label substring, and BFS-traverses up to `depth` hops (default 2) via `nx.single_source_shortest_path_length`; collects only edges where both endpoints are in the visited set, capped at `max_nodes`.
- **Embedding retrieval** (`Retriever.embedding_retrieve`): embeds the query with `all-MiniLM-L6-v2`, loads raw `float32` BLOBs from `node_embeddings`, computes L2-normalised cosine similarity, and returns top-k nodes' out-edges as a context string.
- **Graph indexing** (`Retriever.index_graph`): embeds each un-indexed node using `"{label} ({type}): {pred} {neighbor}, ..."` and persists via `store.upsert_node_embedding`; no-op for already-indexed nodes.
- **Unified interface** (`Retriever.retrieve`): tries BFS first; falls back to embedding similarity when BFS returns `""`.
- **Persistent embedding table** (`node_embeddings`): new `(node_id TEXT PK, embedding BLOB NOT NULL)` table created by `init_db`; helpers `upsert_node_embedding` and `load_all_embeddings` in `memgraph.kg.store` and re-exported from `memgraph.kg`.
- **Unit tests** (`tests/test_retrieval.py`): 22 tests covering BFS seed finding, depth-limiting, max-nodes capping, embedding round-trips, index idempotency, cosine fallback, and the unified interface. All sentence-transformer calls are mocked — no download required.

### M4 — FastAPI Backend (complete)
- **`POST /chat`**: extracts triples from the user message, upserts them to SQLite, reloads the graph, retrieves context via `Retriever.retrieve`, calls Anthropic `claude-haiku-4-5` for a response, and maintains an in-memory conversation history buffer (capped at 20 entries).
- **`GET /graph`**: serializes the live NetworkX graph to `{"nodes": [...], "edges": [...]}` JSON for visualization.
- **`POST /reset`**: truncates all three SQLite tables in FK-safe order (`node_embeddings → edges → nodes`), reinitializes an empty graph, and clears conversation history.
- **Shared runtime state** lives in a module-level `_state` dict, wired up via FastAPI's `lifespan` context manager. `MEMGRAPH_DB_PATH` env var overrides the default `memgraph.db` path (used by tests with `:memory:`).
- **Unit tests** (`tests/test_api.py`): 6 tests covering response shape, graph updates from extracted triples, empty-graph response, reset clearing graph/history, and history trimming. No API key or model download required — Anthropic client and SentenceTransformer are fully mocked.
- Run the server: `uvicorn memgraph.api.main:app --reload`

## Architecture

```
User message
  └─► Anthropic API (triple extraction)  ──► NetworkX KG (SQLite-backed)
User query
  └─► Graph traversal + embedding cosine search
        └─► context string → Anthropic API (chat response)
Streamlit UI ◄──── FastAPI ◄──── all of the above
```

## Quickstart

```bash
pip install -r requirements.txt
# Copy and fill in your Anthropic API key
cp .env.example .env

# Start the API server
uvicorn memgraph.api.main:app --reload

streamlit run ui.py   # (coming in M5)
```

## Running tests

```bash
pytest tests/
```

No API key is required — extractor tests use a fully mocked Anthropic client.

## Project Layout

```
memgraph-personal-ai/
├── memgraph/
│   ├── __init__.py        # package root, version
│   ├── kg/                # entity/relation extraction, SQLite, NetworkX
│   │   ├── __init__.py
│   │   ├── extractor.py   # Triple model + LLM-based triple extraction
│   │   └── store.py       # SQLite schema, upsert, NetworkX graph loader
│   ├── retrieval/         # BFS graph traversal + embedding cosine search
│   │   ├── __init__.py
│   │   ├── retriever.py   # Retriever class: bfs_retrieve, embedding_retrieve, index_graph, retrieve
│   │   ├── traversal.py   # bfs_context: keyword-anchored BFS over NetworkX graph
│   │   └── engine.py      # module-level wrappers: bfs_retrieve, embedding_retrieve, retrieve, compute_and_store_embeddings
│   ├── api/               # FastAPI backend (/chat, /graph, /reset)
│   │   ├── __init__.py
│   │   └── main.py        # FastAPI app: lifespan, /chat, /graph, /reset
│   └── ui/                # Streamlit UI + pyvis visualization
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # global sentence_transformers stub + per-test module isolation fixture
│   ├── test_import.py
│   ├── test_kg.py         # M2 unit tests (extractor + SQLite round-trip)
│   ├── test_retrieval.py  # M3 unit tests (BFS traversal + embedding fallback)
│   └── test_api.py        # M4 unit tests (FastAPI endpoints)
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Milestones

| # | Name | Description | Status |
|---|------|-------------|--------|
| M1 | scaffold + readme | Repo skeleton, package structure, dependencies | done |
| M2 | knowledge graph core | Entity/relation extraction, SQLite persistence, NetworkX graph | done |
| M3 | retrieval | BFS graph traversal and embedding cosine search | done |
| M4 | FastAPI backend | `/chat`, `/graph`, `/reset` endpoints | done |
| M5 | Streamlit UI | Chat panel + live pyvis knowledge-graph visualization | pending |

# MemGraph Personal AI

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
│   ├── retrieval/         # BFS/DFS traversal + embedding cosine search
│   │   └── __init__.py
│   ├── api/               # FastAPI backend (/chat, /graph, /reset)
│   │   └── __init__.py
│   └── ui/                # Streamlit UI + pyvis visualization
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_import.py
│   └── test_kg.py         # M2 unit tests (extractor + SQLite round-trip)
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
| M3 | retrieval | BFS/DFS traversal and embedding cosine search | pending |
| M4 | FastAPI backend | `/chat`, `/graph`, `/reset` endpoints | pending |
| M5 | Streamlit UI | Chat panel + live pyvis knowledge-graph visualization | pending |

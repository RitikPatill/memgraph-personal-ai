# MemGraph Personal AI

A local personal AI assistant that progressively builds a typed knowledge graph from your conversations. MemGraph extracts structured entities (people, places, preferences, facts, events) and relations between them from every message, persists them as a graph, and traverses that graph to gather relevant personal context before generating responses.

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

## Project Layout

```
memgraph-personal-ai/
├── memgraph/
│   ├── __init__.py        # package root, version
│   ├── kg/                # entity/relation extraction, SQLite, NetworkX
│   │   └── __init__.py
│   ├── retrieval/         # BFS/DFS traversal + embedding cosine search
│   │   └── __init__.py
│   ├── api/               # FastAPI backend (/chat, /graph, /reset)
│   │   └── __init__.py
│   └── ui/                # Streamlit UI + pyvis visualization
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_import.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Milestones

| # | Name | Description |
|---|------|-------------|
| M1 | scaffold + readme | Repo skeleton, package structure, dependencies |
| M2 | knowledge graph core | Entity/relation extraction, SQLite persistence, NetworkX graph |
| M3 | retrieval | BFS/DFS traversal and embedding cosine search |
| M4 | FastAPI backend | `/chat`, `/graph`, `/reset` endpoints |
| M5 | Streamlit UI | Chat panel + live pyvis knowledge-graph visualization |

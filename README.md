# MemGraph: Personal AI Assistant with Knowledge Graph Memory

> A local AI assistant that extracts a personal knowledge graph from your conversations and uses it for context-aware, personalized responses.

<!-- TODO: replace with a 5-10 second demo gif. Record with ScreenToGif on
     Windows or peek on macOS. Save to docs/demo.gif and update path here. -->
![demo](docs/demo.gif)

## What it is

MemGraph is a local personal AI assistant that builds a structured knowledge graph from your conversations. Every message you send is parsed for entities — people, places, preferences, facts, events — and the typed relations between them. Those triples are persisted in SQLite and loaded into an in-memory NetworkX graph. When you ask a follow-up question, the assistant traverses the graph to pull relevant context before generating a response, so it remembers what you told it across the session.

Most production memory systems store raw message chunks and retrieve them by vector similarity. MemGraph uses a typed graph instead, which supports structured queries: answering "what food do I like in Berlin?" requires joining location and preference nodes, not just matching embeddings. Both approaches are implemented — keyword-anchored BFS graph traversal runs first, with embedding cosine similarity as a fallback for paraphrased or abstract queries.

## Quickstart

**Requirements:** Python 3.10+, an Anthropic API key

```bash
git clone https://github.com/RitikPatill/memgraph-personal-ai.git
cd memgraph-personal-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # open .env and paste your key
streamlit run ui.py
```

The browser opens automatically. The FastAPI backend starts as a subprocess on port 8000; no separate terminal is needed.

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. |
| `MEMGRAPH_DB_PATH` | `memgraph.db` | Override the SQLite file path. |
| `MEMGRAPH_API_URL` | `http://localhost:8000` | Point the UI at a remote backend. |

## Usage

Type naturally in the chat panel on the left. After each message, the knowledge graph panel on the right re-renders — nodes are color-coded by entity type (person, place, preference, fact, event). The **Reset** button clears the graph and conversation history.

To call the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I live in Berlin and love Italian food."}'
# {"response": "...", "triples_extracted": 2}
```

To run the backend without the Streamlit UI:

```bash
uvicorn memgraph.api.main:app --reload
```

To run the test suite (no API key required — all external calls are mocked):

```bash
pytest tests/
```

## Architecture

```
User message
  └─► POST /chat
        ├─► claude-haiku-4-5  ──► typed triples (subject, predicate, object)
        │                              └─► SQLite  (nodes · edges · node_embeddings)
        │                                      └─► NetworkX MultiDiGraph (in-memory)
        └─► Retriever
              ├─► BFS  (keyword-anchored, depth=2)          primary
              └─► Embedding cosine sim  (MiniLM-L6-v2)      fallback
                       └─► context + history ──► claude-haiku-4-5 ──► reply

Streamlit UI  ◄────────────────────  FastAPI  (/chat  /graph  /reset)
```

## Project structure

```
memgraph-personal-ai/
├── memgraph/
│   ├── kg/           # triple extraction, SQLite schema, NetworkX graph loader
│   ├── retrieval/    # BFS traversal, embedding cosine search, unified Retriever
│   ├── api/          # FastAPI app — /chat, /graph, /reset endpoints
│   └── ui/           # pyvis HTML builder, entity-type color map
├── tests/            # 45 unit tests; no API key or model download required
├── ui.py             # Streamlit entry point — launches backend as subprocess
├── requirements.txt
├── setup.py
└── .env.example
```

## Roadmap

- [ ] Export and import the knowledge graph as JSON for backup and portability
- [ ] Streaming responses via SSE to reduce perceived latency on long replies
- [ ] Confidence-weighted graph pruning to discard or downgrade stale facts over time
- [ ] CLI interface for headless usage without the browser UI
- [ ] Multi-turn entity resolution — merge aliases ("Alex", "I", "me") into a single node

## License

MIT — see LICENSE.

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.

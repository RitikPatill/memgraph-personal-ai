"""FastAPI application: /chat, /graph, /reset endpoints."""

import os
from contextlib import asynccontextmanager
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from memgraph.kg import extract_triples, init_db, load_graph, upsert_triple
from memgraph.retrieval import Retriever

load_dotenv()

DB_PATH: str = os.getenv("MEMGRAPH_DB_PATH", "memgraph.db")
HISTORY_MAX: int = 20

_state: dict[str, Any] = {}

_BASE_SYSTEM = (
    "You are a helpful personal AI assistant with access to a knowledge graph "
    "built from past conversations. Answer questions using the provided context "
    "when relevant, and be concise."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = init_db(DB_PATH)
    G = load_graph(conn)
    client = anthropic.Anthropic()
    retriever = Retriever(conn, G)
    _state["conn"] = conn
    _state["G"] = G
    _state["client"] = client
    _state["retriever"] = retriever
    _state["history"] = []
    yield
    conn.close()


app = FastAPI(title="MemGraph API", lifespan=lifespan)


# ---------- Pydantic models ----------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    triples_extracted: int


class GraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


# ---------- Endpoints ----------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    conn = _state["conn"]
    client = _state["client"]
    history: list[dict] = _state["history"]

    # 1. Extract triples from the user message
    triples = extract_triples(req.message, client)

    # 2. Upsert each triple into SQLite
    for triple in triples:
        upsert_triple(conn, triple)

    # 3. If new triples arrived, reload graph + rebuild retriever
    if triples:
        new_G = load_graph(conn)
        old_model = _state["retriever"]._model
        retriever = Retriever(conn, new_G, model=old_model)
        retriever.index_graph()
        _state["G"] = new_G
        _state["retriever"] = retriever
    else:
        retriever = _state["retriever"]

    # 4. Retrieve context
    context = retriever.retrieve(req.message)

    # 5. Build system prompt
    system = _BASE_SYSTEM
    if context:
        system += f"\n\nRelevant context:\n{context}"

    # 6. Append user message to history, trim
    history.append({"role": "user", "content": req.message})
    if len(history) > HISTORY_MAX:
        history[:] = history[-HISTORY_MAX:]

    # 7. Call Anthropic
    api_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=history,
    )
    reply_text: str = api_response.content[0].text

    # 8. Append assistant reply to history
    history.append({"role": "assistant", "content": reply_text})
    if len(history) > HISTORY_MAX:
        history[:] = history[-HISTORY_MAX:]

    return ChatResponse(response=reply_text, triples_extracted=len(triples))


@app.get("/graph", response_model=GraphResponse)
def get_graph() -> GraphResponse:
    G = _state["G"]
    nodes = [
        {"id": node_id, "label": data.get("label", node_id), "entity_type": data.get("entity_type", "entity")}
        for node_id, data in G.nodes(data=True)
    ]
    edges = [
        {
            "source": src,
            "target": tgt,
            "predicate": data.get("predicate", ""),
            "confidence": data.get("confidence", 1.0),
        }
        for src, tgt, data in G.edges(data=True)
    ]
    return GraphResponse(nodes=nodes, edges=edges)


@app.post("/reset")
def reset() -> dict:
    conn = _state["conn"]
    # FK-safe delete order: node_embeddings → edges → nodes
    conn.execute("DELETE FROM node_embeddings")
    conn.execute("DELETE FROM edges")
    conn.execute("DELETE FROM nodes")
    conn.commit()

    new_G = load_graph(conn)
    old_model = _state["retriever"]._model
    retriever = Retriever(conn, new_G, model=old_model)
    _state["G"] = new_G
    _state["retriever"] = retriever
    _state["history"].clear()

    return {"status": "ok"}

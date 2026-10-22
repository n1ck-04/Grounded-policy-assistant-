"""HTTP API for the assistant.

    GET  /health      status and configuration
    GET  /documents   the documents the assistant can answer from (with links)
    POST /ask         {"question": "..."} -> answer, refusal flag, verified citations

Run:  uvicorn grounded.api:app --reload      then open http://127.0.0.1:8000/docs
(needs ANTHROPIC_API_KEY in .env)
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from grounded import config


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["How long do I have to report an unauthorised transaction to get zero liability?"])


class Citation(BaseModel):
    chunk_id: str
    quote: str


class AskOut(BaseModel):
    question: str
    answer: str
    refused: bool
    verified: bool
    citations: list[Citation]
    retrieved: list[str]
    seconds: float


def create_app(answerer=None) -> FastAPI:
    app = FastAPI(title="Grounded Policy Assistant", version="0.1.0")
    state = {"a": answerer}

    def get():
        if state["a"] is None:
            from grounded.answer import build_answerer
            state["a"] = build_answerer()
        return state["a"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model": config.LLM_MODEL, "top_k": config.TOP_K}

    @app.get("/documents")
    def documents():
        return {"documents": [{"id": k, **v} for k, v in config.SOURCES.items()]}

    @app.post("/ask", response_model=AskOut)
    def ask(body: AskIn):
        a = get().answer(body.question)
        return AskOut(question=a.question, answer=a.answer, refused=a.refused, verified=a.verified,
                      citations=[Citation(**c) for c in a.citations], retrieved=a.retrieved, seconds=a.seconds)

    return app


app = create_app()

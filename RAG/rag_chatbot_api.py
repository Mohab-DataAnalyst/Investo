"""
Investo RAG Chatbot API.

Fully local inference via Ollama; no data is sent to external LLM APIs.
"""

from functools import lru_cache
from importlib import import_module
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/rag", tags=["rag"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatInput(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    investor_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_history: List[ChatMessage]
    sources: List[str]
    status: str


@lru_cache(maxsize=1)
def _load_query_function():
    try:
        return import_module("rag.rag_query_data").query
    except ModuleNotFoundError as e:
        if e.name != "rag":
            raise
        return import_module("rag_query_data").query


def _query_or_503():
    try:
        return _load_query_function()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG service unavailable: {e}",
        ) from e


@router.post("/chat", response_model=ChatResponse)
def chat(data: ChatInput):
    query = _query_or_503()
    try:
        history = [
            {"role": message.role, "content": message.content}
            for message in data.conversation_history
        ]

        response_text, updated_history, sources = query(
            question=data.message,
            conversation_history=history,
            current_investor_id=data.investor_id,
        )

        return ChatResponse(
            response=response_text,
            conversation_history=[ChatMessage(**message) for message in updated_history],
            sources=sources,
            status="success",
        )
    except Exception as e:
        return ChatResponse(
            response=f"An error occurred: {e}",
            conversation_history=data.conversation_history,
            sources=[],
            status="error",
        )


@router.get("/")
def health():
    try:
        _load_query_function()
        service_loaded = True
        detail = None
    except Exception as e:
        service_loaded = False
        detail = str(e)

    response = {
        "status": "Investo RAG Chatbot API is running"
        if service_loaded
        else "Investo RAG Chatbot API is unavailable",
        "service_loaded": service_loaded,
        "inference": "local (Ollama)",
        "model": "llama3",
        "embeddings": "nomic-embed-text",
    }
    if detail:
        response["detail"] = detail
    return response


def create_app() -> FastAPI:
    app = FastAPI(title="Investo RAG Chatbot API")
    app.include_router(router)
    return app


app = create_app()

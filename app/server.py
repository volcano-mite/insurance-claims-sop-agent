"""HTTP API and the test UI."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import data
from app.agent import Agent
from app.config import ROOT_DIR, settings
from app.llm.client import LLMAuthError, LLMError
from app.memory import Session

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Insurance Claims SOP Agent")
agent = Agent()
SESSIONS: dict[str, Session] = {}
LOCKS: dict[str, threading.Lock] = {}
WEB_DIR = Path(ROOT_DIR) / "web"


class NewSessionIn(BaseModel):
    api_key: str | None = None
    consent_scenario: str | None = None


class ChatIn(BaseModel):
    session_id: str
    message: str
    api_key: str | None = None


def _session(session_id: str) -> Session:
    s = SESSIONS.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found. Start a new chat.")
    return s


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/api/config")
def config() -> dict:
    return {
        "has_server_key": bool(settings.api_key),
        "base_url": settings.base_url,
        "models": {"extractor": settings.model_extractor, "responder": settings.model_responder},
        "consent_scenarios": list(data.consent_scenarios()),
        "today": settings.today.isoformat(),
        "limits": {
            "verify_min_fields": settings.verify_min_fields,
            "max_verify_failures": settings.max_verify_failures,
            "max_off_topic": settings.max_off_topic,
            "persuasion_budget": settings.persuasion_budget,
        },
    }


@app.post("/api/session")
def new_session(body: NewSessionIn) -> dict:
    s = agent.new_session(api_key=body.api_key, consent_scenario=body.consent_scenario)
    SESSIONS[s.id] = s
    LOCKS[s.id] = threading.Lock()
    return {"session_id": s.id, "messages": s.history, "state": s.snapshot()}


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    s = _session(session_id)
    return {"session_id": s.id, "messages": s.history, "state": s.snapshot()}


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    s = _session(body.session_id)
    if body.api_key:
        s.api_key = body.api_key
    if not (s.api_key or settings.api_key):
        raise HTTPException(400, "No API key. Enter your DeepSeek API key at the top of the page.")
    if not body.message.strip():
        raise HTTPException(400, "Empty message.")
    with LOCKS[s.id]:
        try:
            reply = agent.handle(s, body.message)
        except LLMAuthError as e:
            raise HTTPException(401, str(e)) from e
        except LLMError as e:
            raise HTTPException(502, f"LLM error: {e}") from e
    return {"reply": reply, "messages": s.history, "state": s.snapshot()}

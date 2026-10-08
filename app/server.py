"""HTTP API and the test UI."""
from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import data
from app.agent import Agent, TurnError
from app.config import ROOT_DIR, settings
from app.llm.client import LLMAuthError, LLMError
from app.memory import Session

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Insurance Claims SOP Agent")
agent = Agent()
SESSIONS: dict[str, Session] = {}
LOCKS: dict[str, threading.Lock] = {}
LAST_SEEN: dict[str, float] = {}
REQUESTS: dict[str, deque] = {}  # client ip -> request times in the last minute
REGISTRY_LOCK = threading.Lock()
WEB_DIR = Path(ROOT_DIR) / "web"
MAX_API_KEY_CHARS = 300


class NewSessionIn(BaseModel):
    api_key: str | None = None
    consent_scenario: str | None = None


class ChatIn(BaseModel):
    session_id: str
    message: str
    api_key: str | None = None


# ---------------------------------------------------------------- access, rate limit, lifecycle

def require_access(request: Request) -> None:
    """With ACCESS_TOKEN set, every session and chat call needs it (X-Access-Token header)."""
    if not settings.access_token:
        return
    given = request.headers.get("x-access-token", "")
    if not hmac.compare_digest(given.encode(), settings.access_token.encode()):
        raise HTTPException(403, "Access code required. Enter the access code at the top of the page.")


def rate_limit(request: Request) -> None:
    """Per client IP, at most RATE_LIMIT_PER_MINUTE session/chat calls in any 60 seconds.

    Behind a reverse proxy, start uvicorn with FORWARDED_ALLOW_IPS set so the client IP is
    taken from X-Forwarded-For instead of the proxy's address.
    """
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with REGISTRY_LOCK:
        q = REQUESTS.setdefault(ip, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= settings.rate_limit_per_minute:
            raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
        q.append(now)


def _cleanup(now: float) -> None:
    """Drop idle sessions, keep at most MAX_SESSIONS, and forget idle rate-limit entries."""
    ttl = settings.session_ttl_minutes * 60
    for sid in [sid for sid, t in LAST_SEEN.items() if now - t > ttl]:
        _drop(sid)
    if len(SESSIONS) >= settings.max_sessions:
        for sid in sorted(LAST_SEEN, key=LAST_SEEN.get)[:len(SESSIONS) - settings.max_sessions + 1]:
            _drop(sid)
    for ip in [ip for ip, q in REQUESTS.items() if not q or now - q[-1] > 60]:
        del REQUESTS[ip]


def _drop(sid: str) -> None:
    SESSIONS.pop(sid, None)
    LOCKS.pop(sid, None)
    LAST_SEEN.pop(sid, None)


def _session(session_id: str) -> Session:
    with REGISTRY_LOCK:
        s = SESSIONS.get(session_id)
        if s is None:
            raise HTTPException(404, "Session not found or expired. Start a new chat.")
        LAST_SEEN[session_id] = time.monotonic()
        return s


def _state(s: Session) -> dict:
    return {"session_id": s.id, "messages": s.history, "state": s.snapshot(debug=settings.debug_inspector)}


def _check_api_key(key: str | None) -> None:
    if key and len(key) > MAX_API_KEY_CHARS:
        raise HTTPException(400, "That API key is too long.")


# ---------------------------------------------------------------- routes

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
        "access_required": bool(settings.access_token),
        "base_url": settings.base_url,
        "models": {"extractor": settings.model_extractor, "responder": settings.model_responder},
        "consent_scenarios": list(data.consent_scenarios()),
        "today": settings.today.isoformat(),
        "today_mode": settings.today_mode,
        "debug_inspector": settings.debug_inspector,
        "limits": {
            "verify_min_fields": settings.verify_min_fields,
            "max_verify_failures": settings.max_verify_failures,
            "max_off_topic": settings.max_off_topic,
            "persuasion_budget": settings.persuasion_budget,
            "max_message_chars": settings.max_message_chars,
        },
    }


@app.post("/api/session", dependencies=[Depends(require_access), Depends(rate_limit)])
def new_session(body: NewSessionIn) -> dict:
    _check_api_key(body.api_key)
    if body.consent_scenario and body.consent_scenario not in data.consent_scenarios():
        raise HTTPException(400, f"Unknown consent scenario {body.consent_scenario!r}.")
    s = agent.new_session(api_key=body.api_key, consent_scenario=body.consent_scenario)
    with REGISTRY_LOCK:
        now = time.monotonic()
        _cleanup(now)
        SESSIONS[s.id] = s
        LOCKS[s.id] = threading.Lock()
        LAST_SEEN[s.id] = now
    return _state(s)


@app.get("/api/session/{session_id}", dependencies=[Depends(require_access)])
def get_session(session_id: str) -> dict:
    return _state(_session(session_id))


@app.post("/api/chat", dependencies=[Depends(require_access), Depends(rate_limit)])
def chat(body: ChatIn) -> dict:
    s = _session(body.session_id)
    _check_api_key(body.api_key)
    if body.api_key:
        s.api_key = body.api_key
    if not (s.api_key or settings.api_key):
        raise HTTPException(400, "No API key. Enter your DeepSeek API key at the top of the page.")
    if not body.message.strip():
        raise HTTPException(400, "Empty message.")
    if len(body.message) > settings.max_message_chars:
        raise HTTPException(400, f"Message is too long (max {settings.max_message_chars} characters).")
    if len(s.history) // 2 >= settings.max_turns:  # counts closed-phase messages too
        raise HTTPException(400, "This conversation has reached its length limit. Please start a new chat.")
    lock = LOCKS.get(s.id)
    if lock is None:
        raise HTTPException(404, "Session not found or expired. Start a new chat.")
    with lock:
        try:
            reply = agent.handle(s, body.message)
        except LLMAuthError as e:
            raise HTTPException(401, str(e)) from e
        except LLMError as e:
            raise HTTPException(502, f"LLM error: {e}") from e
        except TurnError as e:
            raise HTTPException(500, str(e)) from e
    return {"reply": reply, **_state(s)}

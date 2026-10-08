from fastapi.testclient import TestClient

from app import server
from app.config import settings
from tests.fakes import FakeLLM

client = TestClient(server.app)


def test_index_and_config():
    assert client.get("/").status_code == 200
    cfg = client.get("/api/config").json()
    assert set(cfg["consent_scenarios"]) == {"default", "timeout"}


def test_chat_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    sid = client.post("/api/session", json={}).json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert r.status_code == 400 and "API key" in r.json()["detail"]


def test_chat_round_trip_with_ui_key(monkeypatch):
    seen = {}

    def factory(key):
        seen["key"] = key
        return FakeLLM(extractions=[{"identity": {"full_name": "Margaret Chen"}}], replies=["Thanks!"])

    monkeypatch.setattr(server.agent, "llm_factory", factory)
    sid = client.post("/api/session", json={"consent_scenario": "timeout"}).json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "I'm Margaret Chen", "api_key": "sk-test"})
    body = r.json()
    assert r.status_code == 200 and body["reply"] == "Thanks!"
    assert seen["key"] == "sk-test"
    assert body["state"]["verification"]["fields_provided"] == {"full_name": "Margaret Chen"}
    assert server.SESSIONS[sid].consent_scenario == "timeout"


def test_unknown_session():
    assert client.post("/api/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404

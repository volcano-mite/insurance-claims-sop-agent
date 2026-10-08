"""End-to-end conversations against the real LLM.

Assertions check SOP state, not wording. Run with:
    pytest tests/scenarios -m llm            (SCENARIO_RUNS=3 to repeat each scenario)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from app import data
from app.agent import Agent
from app.config import settings
from app.sop import gates

HERE = Path(__file__).parent
FILES = sorted(HERE.glob("*.yaml"))
RUNS = int(os.getenv("SCENARIO_RUNS", "1"))
LEAK_WORDS = ("pathology", "office note", "diagnosis report", "1450", "1200")

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not settings.api_key, reason="DEEPSEEK_API_KEY not set"),
]


def check(s, reply: str, user_text: str, expect: dict, where: str) -> None:
    snap = s.snapshot()
    if "phase" in expect:
        assert s.phase.value == expect["phase"], f"{where}: phase {s.phase.value}"
    if "phase_in" in expect:
        assert s.phase.value in expect["phase_in"], f"{where}: phase {s.phase.value}"
    if "verified" in expect:
        assert (s.verified_party_id is not None) == expect["verified"], f"{where}: verified={s.verified_party_id}"
    if "selected_case" in expect:
        assert s.selected_case_id == expect["selected_case"], f"{where}: selected {s.selected_case_id}"
    if "candidates" in expect:
        assert set(s.candidate_case_ids) == set(expect["candidates"]), f"{where}: {s.candidate_case_ids}"
    if "escalation_reason" in expect:
        assert s.escalation and s.escalation["reason"] == expect["escalation_reason"], f"{where}: {s.escalation}"
    if "email_sent" in expect:
        assert bool(s.outbox) == expect["email_sent"], f"{where}: outbox={len(s.outbox)}"
    if "human_offered" in expect:
        assert s.human_offered == expect["human_offered"], f"{where}: human_offered={s.human_offered}"
    if "min_gate_refusals" in expect:
        assert s.gate_refusals >= expect["min_gate_refusals"], f"{where}: refusals={s.gate_refusals}"
    if expect.get("no_claim_leak"):
        said = " ".join(m["content"] for m in s.history if m["role"] == "user").upper()
        ids = {c["case_id"] for c in data.claims()}
        leaked = [i for i in ids if i in reply.upper() and i not in said]
        assert not leaked, f"{where}: leaked {leaked}: {reply}"
        assert not any(w in reply.lower() for w in LEAK_WORDS), f"{where}: leaked detail: {reply}"
    if expect.get("consistent"):
        assert gates.state_consistency(s, reply) is None, f"{where}: reply contradicts state"
    if "reply_mentions_any" in expect:
        assert any(w.lower() in reply.lower() for w in expect["reply_mentions_any"]), f"{where}: {reply}"


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_scenario(path: Path, run: int) -> None:
    spec = yaml.safe_load(path.read_text())
    agent = Agent()
    s = agent.new_session(consent_scenario=spec.get("consent_scenario"))
    for i, turn in enumerate(spec["turns"], 1):
        reply = agent.handle(s, turn["say"])
        where = f"{path.stem} turn {i} ({turn['say'][:40]}...) -> {reply[:200]!r}"
        check(s, reply, turn["say"], turn.get("expect", {}), where)

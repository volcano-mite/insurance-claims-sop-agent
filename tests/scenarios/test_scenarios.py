"""End-to-end conversations against the real LLM.

Assertions check SOP state and grounded content (amounts, dates, documents, promises,
email privacy), not exact wording. Every verified reply about a selected claim is also
checked for ungrounded amounts, dates, time spans and promises. Run with:
    pytest tests/scenarios -m llm            (SCENARIO_RUNS=3 to repeat each scenario)
"""
from __future__ import annotations

import os
import re
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from app import data
from app.agent import Agent
from app.config import settings
from app.sop import gates, grounding

HERE = Path(__file__).parent
FILES = sorted(HERE.glob("*.yaml"))
RUNS = int(os.getenv("SCENARIO_RUNS", "1"))
LEAK_WORDS = ("pathology", "office note", "diagnosis report", "1450", "1200")
# Independent of app code: the expected facts are written in each YAML file, and amounts are
# read with this regex rather than the app's own parser.
DOLLARS_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{2})?)")


def dollar_amounts(text: str) -> set[Decimal]:
    return {Decimal(m.replace(",", "")) for m in DOLLARS_RE.findall(text)}

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
        said = gates.caller_text(s)
        ids = {c["case_id"] for c in data.claims()}
        leaked = [i for i in ids if i in reply.upper() and i not in said.upper()]
        assert not leaked, f"{where}: leaked {leaked}: {reply}"
        assert not any(w in reply.lower() for w in LEAK_WORDS), f"{where}: leaked detail: {reply}"
        assert not gates.money_values(reply) - gates.money_values(said), f"{where}: leaked amount: {reply}"
    if expect.get("consistent"):
        assert gates.state_consistency(s, reply) is None, f"{where}: reply contradicts state"
    if "reply_mentions_any" in expect:
        assert any(w.lower() in reply.lower() for w in expect["reply_mentions_any"]), f"{where}: {reply}"
    if "reply_mentions_all" in expect:
        missing = [w for w in expect["reply_mentions_all"] if w.lower() not in reply.lower()]
        assert not missing, f"{where}: missing {missing}: {reply}"
    if "amounts_allowed" in expect:
        allowed = {Decimal(str(a)) for a in expect["amounts_allowed"]}
        extra = dollar_amounts(reply) - allowed
        assert not extra, f"{where}: unexpected amounts {sorted(extra)}: {reply}"
    for pattern in expect.get("reply_not_matching", []):
        assert not re.search(pattern, reply), f"{where}: matches forbidden {pattern!r}: {reply}"
    if "reply_excludes" in expect:
        found = [w for w in expect["reply_excludes"] if w.lower() in reply.lower()]
        assert not found, f"{where}: must not say {found}: {reply}"
    if s.verified_party_id and s.selected_case_id:
        problems = grounding.violations(reply, grounding.for_session(s, include_guidance=True))
        assert not problems, f"{where}: ungrounded {problems}: {reply}"
    if expect.get("email_sent") and any(k.startswith("email_") and k != "email_sent" for k in expect):
        email = s.outbox[-1]["subject"] + "\n" + s.outbox[-1]["body"]
        holder = data.policyholder_by_party(s.verified_party_id)
        assert not gates.email_pii_violations(email, holder), f"{where}: email has identity details: {email}"
        missing = [w for w in expect.get("email_mentions_all", []) if w.lower() not in email.lower()]
        assert not missing, f"{where}: email missing {missing}: {email}"
        if "email_mentions_any" in expect:
            assert any(w.lower() in email.lower() for w in expect["email_mentions_any"]), f"{where}: {email}"
        if "email_amounts_allowed" in expect:
            extra = dollar_amounts(email) - {Decimal(str(a)) for a in expect["email_amounts_allowed"]}
            assert not extra, f"{where}: email has unexpected amounts {sorted(extra)}"


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

"""Regression tests for the safety and recovery fixes from the code review."""
import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import server
from app.agent import Agent, TurnError
from app.config import ConfigError, DEMO_TODAY, Settings, settings
from app.llm import prompts
from app.llm.responder import parse_tool_args
from app.memory import CaseHints, Extraction, Phase, Session
from app.sop import gates, grounding
from app.sop.grounding import Evidence
from app.sop.state_machine import Controller, describe_hints
from tests.fakes import FakeLLM, tool_call

DEMO_EX = {
    "identity": {"full_name": "Margaret Chen", "policy_number": "POL-9921",
                 "dob": "1985-03-15", "id_last4": "4472"},
    "caller_role": "policyholder", "intent": "denial_question",
    "case_hints": {"case_type": "healthcare", "status": "denied", "month": 1},
}
WRONG_ID = {"identity": {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "0000"}}


def make(fake):
    return Agent(llm_factory=lambda key: fake)


def unverified(said: str = "hi") -> Session:
    s = Session()
    s.history.append({"role": "user", "content": said})
    return s


# ---------------------------------------------------------------- 1. no per-field verification results

def test_public_snapshot_hides_which_identity_field_failed():
    agent = make(FakeLLM(extractions=[WRONG_ID], replies=["Sorry, those details don't match."]))
    s = agent.new_session()
    agent.handle(s, "Margaret Chen, 1985-03-15, 0000")
    public = s.snapshot()
    assert public["verification"]["last_check"] == {"verified": False}
    assert not any({"matched", "mismatched"} & e.keys() for e in public["trace"])
    debug = s.snapshot(debug=True)
    assert debug["verification"]["last_check"]["mismatched"] == ["id_last4"]


def test_api_response_hides_field_results_unless_debug(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setattr(server.agent, "llm_factory", lambda k: FakeLLM(extractions=[WRONG_ID], replies=["No match."]))
    for debug, expect_detail in ((False, False), (True, True)):
        monkeypatch.setattr(settings, "debug_inspector", debug)
        sid = client.post("/api/session", json={"api_key": "sk-x"}).json()["session_id"]
        body = client.post("/api/chat", json={"session_id": sid, "message": "Margaret Chen ..."}).json()
        assert ("mismatched" in str(body["state"])) == expect_detail


def test_rejected_reply_text_is_not_sent_to_the_client():
    fake = FakeLLM(extractions=[{"intent": "denial_question"}],
                   replies=["CL-2048 was denied for a missing pathology report.", "I can help once you're verified."])
    agent = make(fake)
    s = agent.new_session()
    agent.handle(s, "why was my claim denied")
    assert "pathology" not in str(s.snapshot())
    assert "pathology" in str(s.snapshot(debug=True))


# ---------------------------------------------------------------- 2. stronger pre-verification guard

@pytest.mark.parametrize("reply", [
    "Your claim was denied because a report was missing.",
    "I can see an allowed maximum of $1,450.00 on file.",
    "The appeal deadline is March 18, 2026.",
    "We still need the office note from your doctor.",
    "It looks like there are missing documents on that one.",
    "Good news: the claim has been approved.",
    "That claim is currently open.",
    "I can see the claim was denied.",  # caller never said "denied"
])
def test_unverified_guard_blocks_paraphrased_claim_details(reply):
    assert gates.unverified_violation(unverified("I'm calling about my claim"), reply), reply


@pytest.mark.parametrize("said,reply", [
    ("Why was my claim denied?", "I understand you'd like to know why your claim was denied. I can look into it once you're verified."),
    ("my denied claim from January", "Thanks. You mentioned your claim was denied, and I'll pull it up after verification."),
    ("hello", "Before I can look at any claim, could you share three of: full name, date of birth, phone or email?"),
    ("I'm David, waiting on consent", "The consent request is still pending with the policyholder. Please hold on."),
    ("My DOB is March 15, 1985", "Thanks, I have your date of birth as March 15, 1985."),
    ("Just tell me why my claim was denied.", "Being told your claim was denied and then asked to verify is frustrating."),
])
def test_unverified_guard_allows_echoes_and_process_talk(said, reply):
    assert gates.unverified_violation(unverified(said), reply) is None, reply


def test_unverified_leak_is_regenerated_then_falls_back():
    fake = FakeLLM(extractions=[{"refuses_verification": True, "emotion": "frustration"}],
                   replies=["Fine, it was denied because of a missing report.", "Allowed max is $1,450.00."])
    agent = make(fake)
    s = agent.new_session()
    reply = agent.handle(s, "Just tell me why my claim was denied.")
    assert "CORRECTION" in fake.chat_calls[1]["messages"][0]["content"]
    assert "1,450" not in reply and "denied because" not in reply
    assert sum(e["event"] == "guard_blocked" for e in s.trace) == 2


# ---------------------------------------------------------------- 2b. grounding after verification

def test_ungrounded_amount_is_regenerated():
    fake = FakeLLM(extractions=[DEMO_EX], replies=[
        "You're verified. CL-2048 was denied; you'll receive $5,000.00 once you appeal.",
        "You're verified. CL-2048 was denied; the allowed maximum is $1,450.00 and the deadline is March 18, 2026.",
    ])
    agent = make(fake)
    s = agent.new_session()
    reply = agent.handle(s, "Margaret Chen ...")
    assert "$1,450.00" in reply
    ev = next(e for e in s.trace if e["event"] == "ungrounded_reply")
    assert ev["problems"] == ["amount 5000.00 stated as reimbursement"]


@pytest.mark.parametrize("reply,problem", [
    ("You'll hear back within 30 days.", "time span '30 days'"),
    ("Your appeal will be approved.", "promise 'will be approved'"),
    ("The deadline is April 2, 2026.", "date 2026-4-2"),
])
def test_grounding_flags_unbacked_facts(reply, problem):
    s = Session()
    Controller().step(s, Extraction.model_validate(DEMO_EX))
    assert problem in grounding.violations(reply, grounding.for_session(s))


@pytest.mark.parametrize("reply", [
    "I can't guarantee you'll get the money back, but submitting both documents gives the review what it needs.",
    "Nobody can promise it, and I'm not able to say it will be approved.",
    "There's no guarantee the denial will be reversed.",
])
def test_negated_promises_are_allowed(reply):
    assert grounding.violations(reply, Evidence()) == []


def test_grounded_reply_passes():
    s = Session()
    Controller().step(s, Extraction.model_validate(DEMO_EX))
    reply = ("CL-2048 was denied. Expected reimbursement is $0.00 against an allowed maximum of $1,450.00. "
             "You have until March 18, 2026, which is 17 days from today, to appeal.")
    assert grounding.violations(reply, grounding.for_session(s, include_guidance=True)) == []


# ---------------------------------------------------------------- 3. malformed tool calls and recovery

@pytest.mark.parametrize("raw,error", [
    ("[]", "arguments must be a JSON object"),
    ("not json", "arguments are not valid JSON"),
    ('{"case_id": 2048}', None),
    ('{"case_id": "CL-2048", "topic": "made_up"}', None),
])
def test_tool_arguments_are_validated(raw, error):
    name = "get_followup_guidance" if "topic" in raw else "get_claim"
    args, err = parse_tool_args(name, raw)
    assert err == error
    if not err and name == "get_followup_guidance":
        assert args["topic"] is None


def test_bad_tool_arguments_go_back_to_the_model():
    fake = FakeLLM(extractions=[DEMO_EX],
                   replies=[[tool_call("get_claim", [])], "You're verified. CL-2048 was denied."])
    agent = make(fake)
    s = agent.new_session()
    reply = agent.handle(s, "Margaret Chen ...")
    assert reply.startswith("You're verified")
    tool_msg = next(m for m in fake.chat_calls[1]["messages"] if m["role"] == "tool")
    assert "arguments must be a JSON object" in tool_msg["content"]


def test_unexpected_failure_rolls_back_the_whole_turn():
    fake = FakeLLM(extractions=[DEMO_EX])
    fake.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug"))
    agent = make(fake)
    s = agent.new_session()
    with pytest.raises(TurnError):
        agent.handle(s, "Margaret Chen ...")
    assert s.phase == Phase.VERIFY_ID and s.verified_party_id is None and s.turn == 0
    assert len(s.history) == 1 and s.trace[-1]["event"] == "turn_error"
    # the same message works once the problem is gone
    fake.chat = FakeLLM(replies=["You're verified."]).chat
    fake.extractions = [DEMO_EX]
    agent.handle(s, "Margaret Chen ...")
    assert s.phase == Phase.PROCESS_CASE and s.turn == 1


def test_server_reports_turn_failure_and_session_stays_usable(monkeypatch):
    client = TestClient(server.app)
    fake = FakeLLM(extractions=[DEMO_EX, DEMO_EX], replies=["You're verified."])
    real_chat = fake.chat
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("bug")
        return real_chat(*a, **k)
    fake.chat = flaky
    monkeypatch.setattr(server.agent, "llm_factory", lambda k: fake)
    sid = client.post("/api/session", json={"api_key": "sk-x"}).json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "Margaret Chen ..."})
    assert r.status_code == 500 and "not applied" in r.json()["detail"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "Margaret Chen ..."})
    assert r.status_code == 200 and r.json()["state"]["phase"] == "PROCESS_CASE"


# ---------------------------------------------------------------- 4. value ranges and config

@pytest.mark.parametrize("raw,month", [("13", None), (13, None), ("0", None), ("1", 1), ("Jan", 1),
                                       ("septembre", 9), (1.5, None), (True, None), ("xyz", None), ([1], None)])
def test_month_is_range_checked(raw, month):
    assert CaseHints.model_validate({"month": raw}).month == month


def test_garbage_extraction_shapes_do_not_crash():
    ex = Extraction.model_validate({"identity": [], "case_hints": "jan", "declined_fields": "dob"})
    assert ex.identity.full_name is None and ex.declined_fields == []
    assert CaseHints.model_validate({"year": "99999"}).year is None
    assert describe_hints({"month": 13, "case_type": "auto"}) == "auto claim"


@pytest.mark.parametrize("env,value", [("VERIFY_MIN_FIELDS", "2"), ("VERIFY_MIN_FIELDS", "abc"),
                                       ("MAX_OFF_TOPIC", "0"), ("SOP_TODAY", "2026-13-01"),
                                       ("CONSENT_SCENARIO", "sometimes"), ("DEBUG_INSPECTOR", "maybe")])
def test_invalid_config_fails_at_startup(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ConfigError):
        Settings()


@pytest.mark.parametrize("raw,mode,today", [("", "demo", DEMO_TODAY), ("demo", "demo", DEMO_TODAY),
                                            ("2026-04-01", "fixed", date(2026, 4, 1)), ("real", "real", None)])
def test_sop_today_modes(monkeypatch, raw, mode, today):
    monkeypatch.setenv("SOP_TODAY", raw)
    cfg = Settings()
    assert cfg.today_mode == mode
    assert cfg.today == (today or date.today())


# ---------------------------------------------------------------- 5. email summary privacy and grounding

def _email_flow(email: dict) -> tuple[FakeLLM, Session]:
    fake = FakeLLM(extractions=[DEMO_EX, {"wants_to_end": True}, {"email_decision": "send"}],
                   replies=["You're verified. CL-2048 was denied.", "Anything else?", "Sent. Goodbye!"],
                   email=email)
    agent = make(fake)
    s = agent.new_session()
    for text in ("I'm Margaret Chen, DOB 1985-03-15, SSN last four 4472, phone 650-521-2836.",
                 "That's all", "Yes send it"):
        agent.handle(s, text)
    return fake, s


def test_email_model_never_sees_identity_details():
    fake, _ = _email_flow({"subject": "Summary", "body": "CL-2048 was denied."})
    prompt = next(c["user"] for c in fake.json_calls if c["system"] == prompts.EMAIL_SYSTEM)
    for secret in ("1985-03-15", "4472", "650-521-2836"):
        assert secret not in prompt
    assert "[date of birth]" in prompt and "[id]" in prompt


@pytest.mark.parametrize("body", [
    "Hi Margaret, we verified you with DOB March 15, 1985. CL-2048 was denied.",
    "Hi Margaret, your SSN ending 4472 was confirmed. CL-2048 was denied.",
    "Hi Margaret, CL-2048 will be paid $5,000.00 next week.",
])
def test_unsafe_or_ungrounded_email_falls_back_to_template(body):
    _, s = _email_flow({"subject": "Summary", "body": body})
    sent = s.outbox[0]
    assert sent["body"] != body and sent["subject"].startswith("Summary of your Harbor Mutual claims conversation")
    assert any(e["event"] == "email_check_failed" for e in s.trace)


def test_grounded_email_is_kept():
    body = ("Hi Margaret,\n\nClaim CL-2048 was denied because the pathology report and office note were missing. "
            "The allowed maximum is $1,450.00. Please appeal by March 18, 2026.\n\nAlex, Harbor Mutual Claims Support")
    _, s = _email_flow({"subject": "Your CL-2048 summary", "body": body})
    assert s.outbox[0]["body"] == body


# ---------------------------------------------------------------- 6. verify, ask and end in one message

def test_verify_question_and_goodbye_in_one_turn():
    s = Session()
    plan = Controller().step(s, Extraction.model_validate({
        "identity": DEMO_EX["identity"], "intent": "status_inquiry",
        "case_hints": {"case_type": "auto"}, "wants_to_end": True}))
    assert [d["kind"] for d in plan.directives] == ["VERIFIED", "CASE_SELECTED", "ANSWER_CASE_QUESTION",
                                                    "OFFER_EMAIL_SUMMARY"]
    assert s.phase == Phase.POST_PROCESS and s.selected_case_id == "CL-2102" and plan.tools_enabled


def test_one_turn_answer_uses_case_guidance_and_tools():
    fake = FakeLLM(extractions=[{**DEMO_EX, "wants_to_end": True}], replies=["CL-2048 ... Want an email summary?"])
    agent = make(fake)
    s = agent.new_session()
    agent.handle(s, "Margaret Chen ... why was it denied? That's all I need.")
    call = fake.chat_calls[0]
    assert "Current phase: PROCESS_CASE" in call["messages"][0]["content"] and call["tools"]
    assert s.phase == Phase.POST_PROCESS


def test_question_and_goodbye_mid_case_answers_before_wrapping_up():
    s = Session()
    ctl = Controller()
    ctl.step(s, Extraction.model_validate(DEMO_EX))
    plan = ctl.step(s, Extraction.model_validate({"intent": "next_steps", "wants_to_end": True}))
    assert [d["kind"] for d in plan.directives] == ["ANSWER_CASE_QUESTION", "OFFER_EMAIL_SUMMARY"]
    assert s.phase == Phase.POST_PROCESS and plan.tools_enabled
    # a plain goodbye still goes straight to the email offer
    s2 = Session()
    ctl.step(s2, Extraction.model_validate(DEMO_EX))
    plan = ctl.step(s2, Extraction.model_validate({"wants_to_end": True}))
    assert [d["kind"] for d in plan.directives] == ["OFFER_EMAIL_SUMMARY"]


def test_verify_question_and_email_in_one_turn_sends_and_ends():
    s = Session()
    plan = Controller().step(s, Extraction.model_validate({**DEMO_EX, "email_decision": "send"}))
    assert "SEND_EMAIL" in plan.actions and s.phase == Phase.ENDED
    assert "ANSWER_CASE_QUESTION" in [d["kind"] for d in plan.directives]


# ---------------------------------------------------------------- 7. public deployment limits

@pytest.fixture
def client(monkeypatch):
    server.REQUESTS.clear()
    monkeypatch.setattr(server.agent, "llm_factory", lambda k: FakeLLM(replies=["Hello."]))
    return TestClient(server.app)


def test_access_token(client, monkeypatch):
    monkeypatch.setattr(settings, "access_token", "letmein")
    assert client.get("/api/config").json()["access_required"] is True
    assert client.post("/api/session", json={}).status_code == 403
    assert client.post("/api/session", json={}, headers={"X-Access-Token": "wrong"}).status_code == 403
    r = client.post("/api/session", json={"api_key": "sk-x"}, headers={"X-Access-Token": "letmein"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert client.get(f"/api/session/{sid}").status_code == 403
    assert client.post("/api/chat", json={"session_id": sid, "message": "hi"}).status_code == 403


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    sid = client.post("/api/session", json={"api_key": "sk-x"}).json()["session_id"]
    codes = [client.post("/api/chat", json={"session_id": sid, "message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_message_length_and_conversation_length(client, monkeypatch):
    monkeypatch.setattr(settings, "max_message_chars", 10)
    monkeypatch.setattr(settings, "max_turns", 1)
    sid = client.post("/api/session", json={"api_key": "sk-x"}).json()["session_id"]
    assert client.post("/api/chat", json={"session_id": sid, "message": "x" * 11}).status_code == 400
    assert client.post("/api/chat", json={"session_id": sid, "message": "hi"}).status_code == 200
    r = client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert r.status_code == 400 and "length limit" in r.json()["detail"]


def test_idle_sessions_expire_and_count_is_capped(client, monkeypatch):
    old = client.post("/api/session", json={"api_key": "sk-x"}).json()["session_id"]
    server.LAST_SEEN[old] = time.monotonic() - settings.session_ttl_minutes * 60 - 1
    client.post("/api/session", json={"api_key": "sk-x"})
    assert client.get(f"/api/session/{old}").status_code == 404

    monkeypatch.setattr(settings, "max_sessions", 2)
    ids = [client.post("/api/session", json={"api_key": "sk-x"}).json()["session_id"] for _ in range(3)]
    assert len(server.SESSIONS) <= 2 and ids[-1] in server.SESSIONS


def test_unknown_consent_scenario_is_rejected(client):
    assert client.post("/api/session", json={"consent_scenario": "sometimes"}).status_code == 400

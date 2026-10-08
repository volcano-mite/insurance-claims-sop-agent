from app.agent import Agent
from app.llm import prompts
from app.memory import Phase
from tests.fakes import FakeLLM, tool_call

DEMO_EX = {
    "identity": {"full_name": "Margaret Chen", "policy_number": "POL-9921",
                 "dob": "1985-03-15", "id_last4": "4472"},
    "caller_role": "policyholder", "intent": "denial_question",
    "case_hints": {"case_type": "healthcare", "status": "denied", "month": 1},
}
DEMO_TEXT = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
             "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")


def make(fake):
    return Agent(llm_factory=lambda key: fake)


def test_demo_turn_runs_tool_loop_and_grounds_reply():
    fake = FakeLLM(extractions=[DEMO_EX],
                   replies=[[tool_call("get_document_guidance", {"case_id": "CL-2048"})],
                            "You're verified. CL-2048 was denied because..."])
    agent = make(fake)
    s = agent.new_session()
    reply = agent.handle(s, DEMO_TEXT)
    assert "CL-2048" in reply and s.phase == Phase.PROCESS_CASE
    assert any(e["event"] == "tool_call" and e["ok"] for e in s.trace)
    # tool result was fed back to the model
    tool_msgs = [m for m in fake.chat_calls[1]["messages"] if m["role"] == "tool"]
    assert "pathology report" in tool_msgs[0]["content"] and "requirements" in tool_msgs[0]["content"]


def test_unverified_prompt_has_no_claim_data_and_guard_blocks_leaks():
    fake = FakeLLM(extractions=[{"intent": "denial_question", "refuses_verification": True,
                                 "emotion": "frustration"}],
                   replies=["Fine: CL-2048 was denied for missing documents."])
    agent = make(fake)
    s = agent.new_session()
    reply = agent.handle(s, "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.")
    system_prompt = fake.chat_calls[0]["messages"][0]["content"]
    assert "CL-2048" not in system_prompt and "pathology" not in system_prompt
    assert fake.chat_calls[0]["tools"] is None
    assert "CL-2048" not in reply
    assert any(e["event"] == "guard_blocked" for e in s.trace)


def test_email_is_sent_to_email_on_file():
    fake = FakeLLM(extractions=[DEMO_EX, {"wants_to_end": True}, {"email_decision": "send"}])
    agent = make(fake)
    s = agent.new_session()
    for text in (DEMO_TEXT, "That's all, thanks", "Yes please send it"):
        agent.handle(s, text)
    assert s.phase == Phase.ENDED
    assert s.outbox and s.outbox[0]["to"] == "margaret@email.com"
    assert agent.handle(s, "hello?").startswith("This conversation has ended")


def test_llm_failures_degrade_to_safe_fallbacks():
    fake = FakeLLM(fail_json=True)
    fake.chat = lambda *a, **k: (_ for _ in ()).throw(__import__("app.llm.client", fromlist=["LLMError"]).LLMError("x"))
    agent = make(fake)
    s = agent.new_session()
    reply = agent.handle(s, "hello")
    assert s.phase == Phase.VERIFY_ID and "could you share 3 more" in reply


def test_regex_safety_net_fills_missed_fields():
    fake = FakeLLM(extractions=[{}])
    agent = make(fake)
    s = agent.new_session()
    agent.handle(s, "my email is margaret@email.com and policy POL-9921")
    assert s.identity["email"] == "margaret@email.com" and s.identity["policy_number"] == "POL-9921"


def test_greeting_is_first_message():
    s = make(FakeLLM()).new_session()
    assert s.history[0] == {"role": "assistant", "content": prompts.GREETING}


def test_rejected_api_key_rolls_back_the_turn():
    from app.llm.client import LLMAuthError
    import pytest

    fake = FakeLLM(extractions=[DEMO_EX])
    def bad_chat(*a, **k):
        raise LLMAuthError("API key rejected")
    fake.chat = bad_chat
    agent = make(fake)
    s = agent.new_session()
    with pytest.raises(LLMAuthError):
        agent.handle(s, DEMO_TEXT)
    assert s.phase == Phase.VERIFY_ID and s.verified_party_id is None and len(s.history) == 1


def test_process_case_prefetches_guidance_into_facts():
    fake = FakeLLM(extractions=[DEMO_EX, {"intent": "document_submission"}], replies=["A", "B"])
    agent = make(fake)
    s = agent.new_session()
    agent.handle(s, DEMO_TEXT)
    agent.handle(s, "How soon do I need to submit them?")
    system = fake.chat_calls[-1]["messages"][0]["content"]
    assert "document_guidance" in system and "followup_guidance_for_this_question" in system
    assert "within a week" in system
    assert any(e["event"] == "tool_call" and e.get("prefetched") for e in s.trace)

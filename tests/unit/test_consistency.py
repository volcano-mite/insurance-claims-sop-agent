from app.agent import Agent
from app.memory import Session
from app.sop import gates
from tests.fakes import FakeLLM

DEMO_EX = {
    "identity": {"full_name": "Margaret Chen", "policy_number": "POL-9921",
                 "dob": "1985-03-15", "id_last4": "4472"},
    "intent": "denial_question", "case_hints": {"case_type": "healthcare", "status": "denied", "month": 1},
}
BAD = "I'm sorry, but those details still don't match what we have on file, so I'm not able to verify your identity."


def verified() -> Session:
    s = Session()
    s.verified_party_id = "P9"
    return s


def test_verified_state_rejects_not_verified_claims():
    s = verified()
    assert gates.state_consistency(s, BAD)
    assert gates.state_consistency(s, "Unfortunately you're not verified yet.")
    assert gates.state_consistency(
        s, "Could you share your full name, date of birth and the last 4 digits of your SSN?")


def test_verified_state_allows_normal_claim_talk():
    s = verified()
    for ok in ("You're verified. CL-2048 was denied because two documents were missing.",
               "If the dates on the office note don't match the claim, ask the clinic for a corrected copy.",
               "I'm not able to verify whether the provider already sent it, but you can check the portal.",
               "I'll send the summary to your email address on file."):
        assert gates.state_consistency(s, ok) is None, ok


def test_unverified_state_rejects_verified_claims_but_allows_conditionals():
    s = Session()
    assert gates.state_consistency(s, "Great, you're verified now. Let me look.")
    assert gates.state_consistency(s, "Thanks, verification is complete.")
    for ok in ("Once you're verified, I'll pull up the claim.",
               "As soon as your identity is verified I can help.",
               "I'll look into it right after verification is complete."):
        assert gates.state_consistency(s, ok) is None, ok


def test_contradicting_reply_is_regenerated_once():
    fake = FakeLLM(extractions=[DEMO_EX], replies=[BAD, "Thanks, Margaret, you're now verified. I've pulled up CL-2048."])
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    reply = agent.handle(s, "Margaret Chen, POL-9921, DOB 1985-03-15, SSN 4472, denied claim from January")
    assert reply.startswith("Thanks, Margaret, you're now verified")
    assert any(e["event"] == "state_mismatch" for e in s.trace)
    assert "CORRECTION" in fake.chat_calls[1]["messages"][0]["content"]


def test_persistent_contradiction_falls_back_to_fixed_reply():
    fake = FakeLLM(extractions=[DEMO_EX], replies=[BAD, BAD])
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    reply = agent.handle(s, "Margaret Chen, POL-9921, DOB 1985-03-15, SSN 4472, denied claim from January")
    assert "you're now verified" in reply and "CL-2048" in reply
    assert gates.state_consistency(s, reply) is None


def test_prompt_states_verification_status():
    fake = FakeLLM(extractions=[DEMO_EX], replies=["ok"])
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    agent.handle(s, "Margaret Chen ...")
    system = fake.chat_calls[0]["messages"][0]["content"]
    assert "VERIFICATION STATUS: VERIFIED" in system and system.rstrip().endswith("even if earlier replies said otherwise.")

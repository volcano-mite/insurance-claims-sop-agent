"""Field-aware grounding, pre-verification confirmations, email ordering and the template email.

Regression tests for the second code review: each wrong reply below passed the earlier
"the number appears somewhere in the context" check.
"""
from datetime import date

import pytest

from app.agent import Agent
from app.config import settings
from app.llm import prompts
from app.llm.responder import template_email
from app.memory import Extraction, Session
from app.sop import gates, grounding
from app.sop.state_machine import Controller
from tests.fakes import FakeLLM, tool_call

IDENTITY = {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}
DENIED_JAN = {"identity": IDENTITY, "case_hints": {"case_type": "healthcare", "status": "denied", "month": 1}}


def verified(said: str = "", ex: dict = DENIED_JAN) -> Session:
    s = Session()
    if said:
        s.history.append({"role": "user", "content": said})
    Controller().step(s, Extraction.model_validate(ex))
    return s


def evidence(s: Session) -> grounding.Evidence:
    return grounding.for_session(s, include_guidance=True)


# ---------------------------------------------------------------- 1. field-aware grounding

@pytest.mark.parametrize("reply,problem", [
    # $1,450 is CL-2048's allowed maximum; expected reimbursement and net pay are $0
    ("Your reimbursement amount is $1,450.", "amount 1450 stated as reimbursement"),
    ("You'll be paid $1,450.00 for this claim.", "amount 1450.00 stated as payment"),
    # 18 only appears in the deadline date
    ("Review will take 18 months.", "time span '18 months'"),
    ("Processing takes 17 business days.", "time span '17 business days'"),
    # the recorded reason is missing documents
    ("The claim was denied because the policy doesn't cover this treatment.", "denial reason"),
    ("The reason for the denial is that the policy lapsed.", "denial reason"),
    ("CL-2048 is approved.", "status 'approved' for CL-2048 (recorded: denied)"),
    ("Your claim has been paid.", "status 'paid' for CL-2048 (recorded: denied)"),
    ("Your appeal deadline is January 12, 2026.", "date 2026-1-12"),
])
def test_values_must_match_the_field_they_are_described_as(reply, problem):
    found = grounding.violations(reply, evidence(verified()))
    assert any(p.startswith(problem) for p in found), found


def test_caller_statements_are_not_evidence():
    s = verified("I should get $9999 back for this")
    ev = evidence(s)
    assert grounding.violations("Yes, you're owed $9,999.", ev) == ["amount 9999 stated as reimbursement"]
    assert grounding.violations("That's right, the $9,999 is correct.", ev)
    # acknowledging without confirming is fine
    assert grounding.violations("You mentioned $9,999, but I don't see that amount on this claim.", ev) == []


def test_disputing_a_callers_figure_is_allowed():
    ev = evidence(verified("My doctor said I'm owed $9,999"))
    for reply in ("I don't have anything in the claim showing $9,999, so I can't confirm that figure.",
                  "I'm not able to confirm $9,999; the file shows an allowed maximum of $1,450.00.",
                  "There's no record of a $9,999 amount on this claim.",
                  "If your doctor believes $9,999 should be paid, a representative can review it.",
                  "You asked whether you're owed $9,999; we could not confirm that amount.",
                  "There's no $9,999 amount associated with this claim in our records."):
        assert grounding.violations(reply, ev) == [], reply


def test_each_date_gets_its_own_role():
    ev = evidence(verified())
    reply = "Your claim from January 12, 2026 was denied, and the appeal deadline is March 18, 2026."
    assert grounding.violations(reply, ev) == []
    assert grounding.violations("Filed March 18, 2026; the appeal deadline is January 12, 2026.", ev) == [
        "date 2026-3-18", "date 2026-1-12"]


def test_denial_reason_from_claim_list_summary():
    s = Session()
    Controller().step(s, Extraction.model_validate({"identity": IDENTITY, "case_hints": {"case_type": "healthcare", "month": 1}}))
    ev = grounding.for_session(s)  # disambiguation: only claim summaries are visible
    assert grounding.violations("CL-2048 (denied due to a missing pathology report and office note).", ev) == []
    assert grounding.violations("CL-2048 was denied because the policy lapsed.", ev)


@pytest.mark.parametrize("reply", [
    "CL-2048 was denied because the review file didn't include the pathology report and the treating provider office note.",
    "It was denied due to missing documents.",
    "Expected reimbursement is $0.00 against an allowed maximum of $1,450.00.",
    "If the denial is reversed, you could receive up to $1,450.00.",
    "Your appeal deadline is March 18, 2026, which is 17 days from today.",
    "The claim was filed on January 12, 2026 and is currently denied.",
    "Once the files are received, processing usually takes less than a week.",
    "Please submit both documents within a week.",
    "If it still looks unchanged after a few business days, support can check.",
    "Once it's approved, payment follows.",
    "We reviewed the reason for the denial and the documents needed to move forward.",
    "Reason for the denial: the review file did not include the pathology report.",
    "I can't guarantee it will be approved.",
    "Nobody can guarantee the outcome of an appeal.",
    "The most you could be paid on this claim is the allowed maximum of $1,450.00.",
])
def test_correct_statements_pass(reply):
    assert grounding.violations(reply, evidence(verified())) == [], reply


def test_other_claims_are_checked_against_their_own_record():
    s = verified()
    ev = evidence(s)
    ev.add({"case_id": "CL-2011", "status": "closed", "expected_reimbursement_amount": "780.00",
            "allowed_max_amount": "800.00", "net_pay": "780.00", "net_fee": "800.00"})
    assert grounding.violations("CL-2011 is closed and paid $780.00.", ev) == []
    assert grounding.violations("CL-2011 was denied.", ev)
    assert grounding.violations("CL-2011 paid $800.00.", ev)  # $800 is CL-2011's allowed max, not its payment


def test_agent_regenerates_a_misattributed_amount():
    fake = FakeLLM(extractions=[DENIED_JAN], replies=[
        "You're verified. CL-2048 was denied, and your reimbursement is $1,450.00.",
        "You're verified. CL-2048 was denied; expected reimbursement is $0.00 against an allowed maximum of $1,450.00.",
    ])
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    reply = agent.handle(s, "Margaret Chen ...")
    assert "expected reimbursement is $0.00" in reply
    correction = fake.chat_calls[1]["messages"][0]["content"]
    assert "amount 1450.00 stated as reimbursement" in correction and "what its field means" in correction


# ---------------------------------------------------------------- 2. no confirming the caller's guess before verification

def unverified(said: str) -> Session:
    s = Session()
    s.history.append({"role": "user", "content": said})
    return s


@pytest.mark.parametrize("said,reply", [
    ("Is CL-2048 denied?", "CL-2048 was denied."),
    ("Is CL-2048 denied?", "Yes, it was."),
    ("Is CL-2048 denied?", "Yes — denied, I'm afraid."),
    ("Was my claim approved?", "Your claim was approved."),
    ("Was my claim approved?", "Unfortunately, the claim wasn't approved."),
    ("my claim is denied right?", "Correct. Let's get you verified so we can talk about next steps."),
    ("Is CL-2048 still open?", "It's still open."),
    ("Is CL-2048 denied?", "After checking, CL-2048 was denied."),
    ("Is CL-2048 denied?", "When I looked, the claim was denied."),
])
def test_unverified_reply_cannot_confirm_a_guess(said, reply):
    assert gates.guard_reply(unverified(said), reply), reply


@pytest.mark.parametrize("said,reply", [
    ("Is CL-2048 denied?", "I can check the status of CL-2048 as soon as you're verified."),
    ("Is CL-2048 denied?", "I can't confirm whether CL-2048 was denied until you're verified."),
    ("Is CL-2048 denied?", "Yes, I can look into that once you're verified. Could you share your full name?"),
    ("Why was my claim denied?", "You mentioned your claim was denied; I'll go through it right after verification."),
    ("Just tell me why my claim was denied.", "Being told your claim was denied and then asked to verify is frustrating."),
    ("Is it approved yet?", "Not yet, David. Consent is still pending, and I'll help as soon as it's approved."),
    ("my denied claim from January", "I'm sorry to hear it was denied. I'll look into it after verification."),
    ("Anything yet?", "Consent hasn't come through yet. I'll pick right back up the moment it's approved."),
])
def test_unverified_reply_may_acknowledge_the_question(said, reply):
    assert gates.guard_reply(unverified(said), reply) is None, reply


# ---------------------------------------------------------------- 3. email is written after the answer

def test_email_summary_sees_this_turns_answer_and_tool_results():
    answer = "CL-2048 was denied because the review file didn't include the pathology report. I've emailed you a summary."
    fake = FakeLLM(extractions=[{**DENIED_JAN, "intent": "denial_question", "email_decision": "send"}],
                   replies=[[tool_call("get_claim", {"case_id": "CL-2048"})], answer],
                   email={"subject": "CL-2048", "body": "CL-2048 was denied: the pathology report and office note were missing."})
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    agent.handle(s, "Margaret Chen, 1985-03-15, 4472. Why was it denied? Email me the summary.")
    email_prompt = next(c["user"] for c in fake.json_calls if c["system"] == prompts.EMAIL_SYSTEM)
    assert answer in email_prompt
    assert s.outbox[0]["body"] == "CL-2048 was denied: the pathology report and office note were missing."
    # the email went out after the reply was generated
    events = [e["event"] for e in s.trace]
    assert events.index("send_email") > max(i for i, e in enumerate(events) if e == "tool_call")


# ---------------------------------------------------------------- 4. template email is a complete summary

def test_template_email_covers_discussion_outcome_and_next_steps():
    s = verified(ex={**DENIED_JAN, "intent": "denial_question"})
    subject, body = template_email(s)
    assert "CL-2048" in subject
    for part in ("What we discussed", "Claim status", "Next steps", "why the claim was denied",
                 "is denied", "did not include the pathology report", "Appeal deadline: March 18, 2026",
                 "17 days away", "Send the pathology report", "Send the office note",
                 "If it is not available", "member portal", "usually less than a week",
                 "Send everything before March 18, 2026"):
        assert part in body, part
    holder = {"id_last4": "4472", "dob": "1985-03-15", "phone": "+16505212836"}
    assert gates.email_pii_violations(body, holder) == []
    assert grounding.violations(subject + "\n" + body, evidence(s)) == []


def test_template_email_for_a_claim_without_documents_and_a_passed_deadline(monkeypatch):
    s = verified(ex={"identity": IDENTITY, "case_hints": {"case_type": "auto"}})
    _, body = template_email(s)
    assert "CL-2102 is open" in body and "no documents are currently requested" in body
    monkeypatch.setattr(settings, "_fixed_today", date(2026, 4, 1))
    _, body = template_email(verified())
    assert "This deadline has passed" in body and "Send everything before" not in body


def test_failed_email_draft_falls_back_to_the_full_template():
    fake = FakeLLM(extractions=[DENIED_JAN, {"wants_to_end": True}, {"email_decision": "send"}],
                   replies=["You're verified. CL-2048 was denied.", "Anything else?", "Sent."],
                   email={"subject": "Summary", "body": "Your reimbursement of $1,450.00 is on its way."})
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    for text in ("Margaret Chen ...", "That's all", "Yes"):
        agent.handle(s, text)
    body = s.outbox[0]["body"]
    assert "What we discussed" in body and "Next steps" in body and "pathology report" in body


# ---------------------------------------------------------------- required directives

def test_reply_that_drops_the_email_offer_is_regenerated():
    fake = FakeLLM(extractions=[{"identity": IDENTITY, "case_hints": {"case_type": "auto"}, "wants_to_end": True}],
                   replies=["You're verified. CL-2102 is open.",
                            "You're verified. CL-2102 is open. Would you like an email summary of this conversation?"])
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    reply = agent.handle(s, "Margaret Chen ... status of my auto claim? That's all.")
    assert "email summary" in reply
    assert any(e["event"] == "directive_missing" for e in s.trace)


def test_escalation_reply_must_include_reference():
    from app.memory import TurnPlan
    plan = TurnPlan()
    plan.add("ESCALATE_TO_HUMAN", reason="caller_requested_human", reference="HX-12345", caller_verified=False)
    assert gates.missing_directives(plan, "I'll transfer you to a person now.")
    assert gates.missing_directives(plan, "Transferring you now; your reference is HX-12345.") is None


def test_caller_figure_presented_as_a_field_is_still_flagged():
    ev = evidence(verified("I'm owed $9,999"))
    assert grounding.violations("Your reimbursement is $9,999.", ev)


def test_unsupported_sentences_are_pruned_after_two_failed_drafts():
    bad = ("You're verified. CL-2048 was denied because the pathology report and office note were missing. "
           "Review will take 18 months.\n\nYour appeal deadline is March 18, 2026.")
    fake = FakeLLM(extractions=[DENIED_JAN], replies=[bad, bad])
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    reply = agent.handle(s, "Margaret Chen ...")
    assert "18 months" not in reply and "March 18, 2026" in reply and "pathology report" in reply
    assert any(e["event"] == "pruned_reply" and e["sentences_removed"] == 1 for e in s.trace)


def test_leaks_are_never_pruned_only_replaced():
    fake = FakeLLM(extractions=[{"intent": "denial_question"}],
                   replies=["Hi! CL-2048 was denied. Please verify.", "Hi! CL-2048 was denied. Please verify."])
    agent = Agent(llm_factory=lambda k: fake)
    s = agent.new_session()
    reply = agent.handle(s, "why was it denied")
    assert "CL-2048" not in reply and not any(e["event"] == "pruned_reply" for e in s.trace)

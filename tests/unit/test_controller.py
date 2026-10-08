from app.config import Settings
from app.memory import Extraction, Phase, Session
from app.sop import gates
from app.sop.state_machine import Controller

ctl = Controller(Settings())

DEMO = {
    "identity": {"full_name": "Margaret Chen", "policy_number": "POL-9921",
                 "dob": "1985-03-15", "id_last4": "4472"},
    "caller_role": "policyholder",
    "intent": "denial_question",
    "case_hints": {"case_type": "healthcare", "status": "denied", "month": 1},
}


def ex(**kw) -> Extraction:
    return Extraction.model_validate(kw)


def kinds(plan) -> list[str]:
    return [d["kind"] for d in plan.directives]


def test_official_demo_verifies_and_uses_remembered_hint_in_one_turn():
    s = Session()
    plan = ctl.step(s, ex(**DEMO))
    assert s.verified_party_id == "P9"
    assert s.selected_case_id == "CL-2048"
    assert s.phase == Phase.PROCESS_CASE
    assert kinds(plan) == ["VERIFIED", "CASE_SELECTED", "ANSWER_CASE_QUESTION"]
    assert plan.tools_enabled


def test_intent_said_before_verification_is_remembered_not_disclosed():
    s = Session()
    plan = ctl.step(s, ex(intent="denial_question",
                          case_hints={"case_type": "healthcare", "status": "denied", "month": 1}))
    assert s.phase == Phase.VERIFY_ID
    assert "ACK_REMEMBERED_REQUEST" in kinds(plan) and "ASK_FOR_IDENTITY" in kinds(plan)
    facts = gates.visible_facts(s)
    assert "selected_claim" not in facts and "claims_on_file" not in facts

    ctl.step(s, ex(identity={"full_name": "Margaret Chen", "dob": "1985-03-15"}))
    assert s.phase == Phase.VERIFY_ID
    plan = ctl.step(s, ex(identity={"email": "margaret@email.com"}))
    assert s.selected_case_id == "CL-2048" and s.phase == Phase.PROCESS_CASE
    assert "CASE_SELECTED" in kinds(plan)


def test_ask_for_identity_lists_alternatives_without_declined_fields():
    s = Session()
    plan = ctl.step(s, ex(identity={"full_name": "Margaret Chen"}, declined_fields=["id_last4"]))
    ask = next(d for d in plan.directives if d["kind"] == "ASK_FOR_IDENTITY")
    assert ask["still_needed"] == 2
    assert "last 4 digits of SSN or national ID" not in ask["options"]
    assert "email address" in ask["options"]


def test_mismatch_clears_fields_and_escalates_after_max_failures():
    s = Session()
    bad = {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "0000"}
    plan = ctl.step(s, ex(identity=bad))
    assert "VERIFICATION_FAILED" in kinds(plan) and s.identity == {}
    ctl.step(s, ex(identity=bad))
    plan = ctl.step(s, ex(identity=bad))
    assert s.phase == Phase.ESCALATED and s.escalation["reason"] == "verification_failed"
    assert s.verified_party_id is None


def test_frustrated_caller_gets_explanation_then_escalation_without_bypass():
    s = Session()
    angry = dict(refuses_verification=True, emotion="frustration", emotion_intensity="medium",
                 intent="denial_question")
    plan = ctl.step(s, ex(**angry))
    assert "EXPLAIN_WHY_VERIFICATION" in kinds(plan) and "ASK_FOR_IDENTITY" in kinds(plan)
    assert s.phase == Phase.VERIFY_ID
    ctl.step(s, ex(**angry))
    plan = ctl.step(s, ex(**angry))
    assert s.phase == Phase.ESCALATED and s.escalation["reason"] == "verification_refused"
    assert s.verified_party_id is None


def test_high_anger_escalates_sooner():
    s = Session()
    angry = dict(refuses_verification=True, emotion="anger", emotion_intensity="high")
    ctl.step(s, ex(**angry))
    assert s.phase == Phase.VERIFY_ID
    ctl.step(s, ex(**angry))
    assert s.phase == Phase.ESCALATED


def test_off_topic_streak_offers_human_and_acceptance_escalates():
    s = Session()
    for _ in range(2):
        plan = ctl.step(s, ex(in_scope=False, off_topic_summary="what is RL"))
        assert "DECLINE_OFF_TOPIC" in kinds(plan) and "OFFER_HUMAN" not in kinds(plan)
    plan = ctl.step(s, ex(in_scope=False, off_topic_summary="what is RL"))
    assert "OFFER_HUMAN" in kinds(plan) and s.human_offered
    ctl.step(s, ex(accepts_human_offer=True))
    assert s.phase == Phase.ESCALATED


def test_in_scope_message_resets_off_topic_streak():
    s = Session()
    ctl.step(s, ex(in_scope=False))
    ctl.step(s, ex(in_scope=False))
    ctl.step(s, ex(identity={"full_name": "Margaret Chen"}))
    assert s.off_topic_streak == 0


def test_explicit_human_request_escalates_from_any_phase():
    s = Session()
    ctl.step(s, ex(**DEMO))
    ctl.step(s, ex(wants_human=True))
    assert s.phase == Phase.ESCALATED and s.escalation["reason"] == "caller_requested_human"


def test_ambiguous_hint_triggers_disambiguation_then_selection():
    s = Session()
    plan = ctl.step(s, ex(identity=DEMO["identity"], case_hints={"case_type": "healthcare", "month": 1}))
    assert s.phase == Phase.RESOLVE_INTENT
    assert "DISAMBIGUATE" in kinds(plan)
    assert set(s.candidate_case_ids) == {"CL-2048", "CL-2011"}
    plan = ctl.step(s, ex(selected_case_id="CL-2011"))
    assert s.selected_case_id == "CL-2011" and s.phase == Phase.PROCESS_CASE


def test_single_claim_is_selected_automatically():
    s = Session()
    ctl.step(s, ex(identity={"full_name": "Ma Tian", "id_last4": "6688", "dob": "1964-09-10"}))
    assert s.selected_case_id == "CL-3001" and s.phase == Phase.PROCESS_CASE


def test_no_claims_on_file():
    s = Session()
    plan = ctl.step(s, ex(identity={"full_name": "Ava Lopez", "id_last4": "9180", "dob": "1990-08-21"}))
    assert "NO_CLAIMS_ON_FILE" in kinds(plan) and s.selected_case_id is None


def test_switching_to_another_claim():
    s = Session()
    ctl.step(s, ex(**DEMO))
    ctl.step(s, ex(case_hints={"case_id": "CL-2102"}, intent="status_inquiry"))
    assert s.selected_case_id == "CL-2102" and s.phase == Phase.PROCESS_CASE


def test_cannot_select_another_partys_claim():
    s = Session()
    ctl.step(s, ex(**DEMO))
    plan = ctl.step(s, ex(case_hints={"case_id": "CL-3001"}))
    assert s.selected_case_id != "CL-3001"
    assert "NO_MATCHING_CLAIM" in kinds(plan)


def test_post_process_send_email():
    s = Session()
    ctl.step(s, ex(**DEMO))
    plan = ctl.step(s, ex(wants_to_end=True))
    assert s.phase == Phase.POST_PROCESS and "OFFER_EMAIL_SUMMARY" in kinds(plan)
    plan = ctl.step(s, ex(email_decision="send"))
    assert "SEND_EMAIL" in plan.actions and s.phase == Phase.ENDED


def test_post_process_skip_email_and_question_before_deciding():
    s = Session()
    ctl.step(s, ex(**DEMO))
    ctl.step(s, ex(wants_to_end=True))
    plan = ctl.step(s, ex(intent="document_submission"))
    assert s.phase == Phase.PROCESS_CASE and "ANSWER_CASE_QUESTION" in kinds(plan)
    ctl.step(s, ex(wants_to_end=True))
    plan = ctl.step(s, ex(email_decision="skip"))
    assert "SEND_EMAIL" not in plan.actions and s.phase == Phase.ENDED and s.email_decision == "skip"


def test_representative_consent_approved():
    s = Session(consent_scenario="default")
    plan = ctl.step(s, ex(caller_role="representative", representative_name="David Chen",
                          relationship="son", identity=DEMO["identity"]))
    assert "CONSENT_PENDING" in kinds(plan) and s.verified_party_id is None
    plan = ctl.step(s, ex())
    assert "CONSENT_APPROVED" in kinds(plan) and s.verified_party_id == "P9"
    assert s.phase in (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE)


def test_representative_consent_timeout_escalates():
    s = Session(consent_scenario="timeout")
    ctl.step(s, ex(caller_role="representative", representative_name="David Chen", identity=DEMO["identity"]))
    for _ in range(4):
        ctl.step(s, ex())
    assert s.phase == Phase.ESCALATED and s.escalation["reason"] == "consent_timeout"
    assert s.verified_party_id is None


def test_unknown_representative_is_not_authorized():
    s = Session()
    plan = ctl.step(s, ex(caller_role="representative", representative_name="John Smith",
                          identity=DEMO["identity"]))
    assert "REPRESENTATIVE_NOT_AUTHORIZED" in kinds(plan) and s.verified_party_id is None


def test_guard_blocks_claim_ids_before_verification():
    s = Session()
    s.history.append({"role": "user", "content": "hi"})
    _, violation = gates.guard_reply(s, "Your claim CL-2048 was denied.")
    assert violation
    s.history.append({"role": "user", "content": "it's about CL-2048"})
    _, violation = gates.guard_reply(s, "Thanks, I've noted CL-2048 for after verification.")
    assert violation is None


def test_guard_blocks_other_partys_claims_after_verification():
    s = Session()
    ctl.step(s, ex(**DEMO))
    _, violation = gates.guard_reply(s, "Claim CL-3001 is also denied.")
    assert violation
    _, violation = gates.guard_reply(s, "Claim CL-2048 was denied.")
    assert violation is None

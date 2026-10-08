from datetime import date

from app.tools import claims, consent, guidance


def test_denied_healthcare_january_resolves_to_cl2048():
    found, exact = claims.match_claims("P9", {"case_type": "healthcare", "status": "denied", "month": 1})
    assert exact and [c["case_id"] for c in found] == ["CL-2048"]


def test_january_healthcare_without_status_is_ambiguous():
    found, exact = claims.match_claims("P9", {"case_type": "medical", "month": 1})
    assert exact and {c["case_id"] for c in found} == {"CL-2048", "CL-2011"}


def test_partial_match_returns_best_candidates():
    found, exact = claims.match_claims("P9", {"status": "denied", "month": 2})
    assert not exact and {c["case_id"] for c in found} == {"CL-2048", "CL-2102"}


def test_claim_lookup_is_scoped_to_party():
    assert claims.get_claim("P9", "CL-2048")
    assert claims.get_claim("P9", "CL-3001") is None
    assert claims.get_claim("P9", "2048")["case_id"] == "CL-2048"


def test_deadline_facts():
    facts = claims.claim_facts(claims.get_claim("P9", "CL-2048"), date(2026, 3, 1))
    assert facts["appeal_deadline_passed"] is False and facts["days_until_appeal_deadline"] == 17
    assert "party_id" not in facts


def test_document_name_mapping():
    assert guidance.resolve_document_key("pathology report") == "original pathology report"
    assert guidance.resolve_document_key("office note") == "treating provider office note"
    assert guidance.resolve_document_key("diagnosis report") is None


def test_document_guidance_falls_back_for_unknown_document():
    g = guidance.get_document_guidance(claims.get_claim("P12", "CL-3001"))
    doc = g["documents"][0]
    assert doc["requirements"] is None and not doc["specific_guidance_available"]
    assert "replacement copy" in doc["if_unavailable"]
    assert "provider or facility" in g["case_type_guidance"]


def test_followup_templates_are_filled():
    claim = claims.get_claim("P9", "CL-2048")
    g = guidance.get_followup_guidance(claim, "How long does it take after I submit?")
    text = " ".join(t["guidance"] for t in g["matched_topics"])
    assert "CL-2048" in text and "the pathology report and the office note" in text
    assert "{" not in text


def test_followup_fallback_and_explicit_topic():
    claim = claims.get_claim("P9", "CL-2048")
    assert "fallback" in guidance.get_followup_guidance(claim, "can my dog help?")
    g = guidance.get_followup_guidance(claim, "", topic="submission_timing")
    assert g["matched_topics"][0]["guidance"].endswith("within a week.")


def test_followup_rules_needing_documents_skip_claims_without_documents():
    g = guidance.get_followup_guidance(claims.get_claim("P9", "CL-2102"), "how long does it take")
    assert g["matched_topics"] == []


def test_consent_default_and_timeout():
    req = consent.request_consent("P9", "David Chen", "default")
    assert consent.poll(req) == "pending" and consent.poll(req) == "approved"
    req = consent.request_consent("P9", "David Chen", "timeout")
    statuses = [consent.poll(req) for _ in range(5)]
    assert statuses[:4] == ["pending"] * 4 and statuses[4] == "timeout"

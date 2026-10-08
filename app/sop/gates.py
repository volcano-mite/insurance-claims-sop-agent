"""Information gates.

visible_facts() decides what data the responder LLM may see in the current
phase. Before verification it contains no claim data at all, so the model
cannot leak what it never received. guard_reply() is a second line of defense
on the model's output.
"""
from __future__ import annotations

import re

from app import data
from app.config import settings
from app.memory import Phase, Session
from app.tools import claims
from app.tools.email import mask_email

CASE_ID_RE = re.compile(r"\bCL-?\d{3,}\b", re.I)


def visible_facts(s: Session) -> dict:
    facts: dict = {"phase": s.phase.value}
    if s.verified_party_id is None:
        facts["identity_fields_received"] = s.identity_labels()
        facts["policy_number_received"] = bool(s.identity.get("policy_number"))
        facts["caller_said_they_are"] = s.caller_role or "unknown"
        if s.rep_name:
            facts["caller_name"] = s.rep_name
        facts["claim_data_access"] = "NONE - caller not verified"
        return facts

    holder = data.policyholder_by_party(s.verified_party_id)
    facts["caller"] = {
        "role": s.caller_role or "policyholder",
        "policyholder_name": holder["name"],
        "representative_name": s.rep_name if s.caller_role == "representative" else None,
        "relationship": s.relationship if s.caller_role == "representative" else None,
    }
    facts["today"] = settings.today.isoformat()
    if s.phase == Phase.RESOLVE_INTENT or s.candidate_case_ids:
        facts["claims_on_file"] = [claims.claim_summary(c) for c in claims.list_claims(s.verified_party_id)]
    if s.selected_case_id:
        claim = claims.get_claim(s.verified_party_id, s.selected_case_id)
        facts["selected_claim"] = claims.claim_facts(claim, settings.today)
        facts["amount_field_definitions"] = {
            k: v["description"] for k, v in data.claim_schema()["field_descriptions"].items()
        }
    if s.phase in (Phase.POST_PROCESS, Phase.ENDED):
        facts["email_on_file_masked"] = mask_email(holder["email"])
    return facts


def _mentioned_by_caller(s: Session) -> str:
    return " ".join(m["content"] for m in s.history if m["role"] == "user").upper()


def guard_reply(s: Session, reply: str) -> tuple[str, str | None]:
    """Return (reply, violation). violation is None when the reply is safe."""
    said = _mentioned_by_caller(s)
    ids_in_reply = {claims.normalize_case_id(m) for m in CASE_ID_RE.findall(reply)}

    if s.verified_party_id is None:
        leaked = [i for i in ids_in_reply if i and i not in said]
        if leaked:
            return reply, f"unverified reply mentions claim ids {leaked}"
        for c in data.claims():
            for k in ("allowed_max_amount", "expected_reimbursement_amount", "net_pay", "net_fee"):
                amt = c.get(k)
                if amt and amt != "0.00" and amt in reply and amt not in said:
                    return reply, "unverified reply mentions a claim amount"
            if c.get("denial_reason") and c["denial_reason"].lower() in reply.lower():
                return reply, "unverified reply mentions a denial reason"
        return reply, None

    own = {c["case_id"] for c in claims.list_claims(s.verified_party_id)}
    foreign = [i for i in ids_in_reply if i and i not in own and i not in said]
    if foreign:
        return reply, f"reply mentions claims of another party {foreign}"
    return reply, None


# ---------------------------------------------------------------- state consistency
# The LLM sometimes follows the conversation's earlier pattern instead of the current
# directives, e.g. repeating "those details don't match" after verification succeeded.
# These checks catch replies that contradict the controller's state.

_SAYS_NOT_VERIFIED = [
    # "not able to verify you / your identity", "you're not verified", "couldn't be verified"
    re.compile(r"\b(not|n't|unable to|cannot)\s+(yet\s+)?(be\s+)?(able\s+to\s+)?"
               r"(verify\s+(you\b|your\b|who you are|it's you)|verified\b)", re.I),
    # "details don't match our records / what we have on file"
    re.compile(r"\b(not|n't)\s+(quite\s+|still\s+)?match(ing)?\s+(what we have|our records|the records|"
               r"our files|what's on file|the information on file)", re.I),
]
_ID_FIELD_WORDS = ("full name", "date of birth", "phone number", "email address", "last 4", "last four", "ssn")
_SAYS_VERIFIED = re.compile(
    r"\b(you('re| are)|you've been|you have been|your identity (is|has been))\s+(now\s+)?(fully\s+|successfully\s+)?verified\b"
    r"|\bverification (is|has been) (now\s+)?(complete|successful|done)\b", re.I)
_CONDITIONAL = re.compile(r"(once|after|until|when|as soon as|before|if)\s+\S*\s*$", re.I)


def state_consistency(s: Session, reply: str) -> str | None:
    """Return a description of how the reply contradicts the SOP state, or None."""
    if s.verified_party_id is not None:
        for rx in _SAYS_NOT_VERIFIED:
            if rx.search(reply):
                return "caller IS verified but the reply says verification failed or details don't match"
        if sum(w in reply.lower() for w in _ID_FIELD_WORDS) >= 3:
            return "caller IS verified but the reply asks for identity details again"
        return None
    for m in _SAYS_VERIFIED.finditer(reply):
        if not _CONDITIONAL.search(reply[max(0, m.start() - 20):m.start()]):
            return "caller is NOT verified but the reply says they are verified"
    return None

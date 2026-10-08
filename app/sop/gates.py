"""Information gates.

visible_facts() decides what data the responder LLM may see in the current
phase. Before verification it contains no claim data at all, so the model
cannot leak what it never received. The checks below are a second line of
defense on the model's output:

- guard_reply(): before verification, nothing that reads as claim information
  (ids, amounts, claim dates, document names, denial explanations, status
  statements, or confirming the caller's own guess); after it, no other party's claims.
- grounding.violations() (grounding.py): after verification, each stated fact must
  match the trusted claim data and guidance, field by field.
- state_consistency(): the reply must not contradict the verification state.
"""
from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation

from app import data
from app.config import settings
from app.memory import Phase, Session
from app.tools import claims
from app.tools.email import mask_email
from app.tools.verification import normalize_dob, normalize_phone

CASE_ID_RE = re.compile(r"\bCL-?\d{3,}\b", re.I)
MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")
_MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"


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


def caller_text(s: Session) -> str:
    return " ".join(m["content"] for m in s.history if m["role"] == "user")


# ---------------------------------------------------------------- value extraction

_MONEY_RE = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)"
    r"|\b(\d[\d,]*\.\d{2})\b"
    r"|\b(\d[\d,]*(?:\.\d{1,2})?)\s?(?:dollars|usd)\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_US_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_DAY_RE = re.compile(rf"\b{_MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_RE}(?:,?\s+(\d{{4}}))?", re.I)


def _decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def money_values(text: str) -> set[Decimal]:
    out = set()
    for m in _MONEY_RE.finditer(text):
        raw = next(g for g in m.groups() if g)
        if (d := _decimal(raw)) is not None:
            out.add(d)
    return out


def _month_num(token: str) -> int:
    return next(i for i, name in enumerate(MONTHS, 1) if name.startswith(token.lower().rstrip(".")[:3]))


def date_values(text: str) -> set[tuple[int | None, int, int]]:
    """(year or None, month, day) for every date written in the text."""
    out = set()
    for y, mo, d in _ISO_DATE_RE.findall(text):
        out.add((int(y), int(mo), int(d)))
    for mo, d, y in _US_DATE_RE.findall(text):
        out.add((int(y), int(mo), int(d)))
    for mon, d, y in _MONTH_DAY_RE.findall(text):
        out.add((int(y) if y else None, _month_num(mon), int(d)))
    for d, mon, y in _DAY_MONTH_RE.findall(text):
        out.add((int(y) if y else None, _month_num(mon), int(d)))
    return {v for v in out if 1 <= v[1] <= 12 and 1 <= v[2] <= 31}


def _date_backed(value: tuple, known: set[tuple]) -> bool:
    y, mo, d = value
    return any(k[1] == mo and k[2] == d and (y is None or k[0] is None or k[0] == y) for k in known)


# ---------------------------------------------------------------- before verification

_CLAIM_EXPLANATION_RE = re.compile(
    r"\b(denied|rejected|declined|approved|paid|closed|settled)\b[^.?!\n]{0,40}?"
    r"\b(because|due to|since|as a result of|for (?:missing|lack|incomplete))\b", re.I)
_STATUS_WORDS = (r"denied|rejected|declined|approved|paid|closed|settled|open|pending|reopened|under review|in review")
_STATUS_ASSERTION_RE = re.compile(
    rf"\b(?:claim|case|CL-?\d{{3,}})(?:\s+(?:is|was|has been|had been|isn't|wasn't|hasn't been|is not|was not|"
    rf"has not been)|'s been|'s)\s+(?:currently\s+|already\s+|now\s+|still\s+)?({_STATUS_WORDS})\b"
    r"|\bit(?:\s+(?:is|was|has been|isn't|wasn't|hasn't been)|'s been|'s)\s+(?:currently\s+|already\s+|still\s+)?"
    r"(denied|rejected|declined|approved|paid|settled|closed|reopened|open)\b", re.I)  # not "pending": consent waits
# Acknowledging what the caller said or asked is fine ("you mentioned your claim was denied",
# "I can check whether it was denied"); stating it as fact is a confirmation, even when the
# caller suggested it ("Is CL-2048 denied?" -> "CL-2048 was denied.").
_ATTRIBUTION_RE = re.compile(
    r"\b(you(?:'ve| have)? (?:said|mentioned|told me|noted|believe|think|were told|heard|received)|"
    r"(?:you've|you have|you've been|you have been|being) told|(?:sorry )?to hear|hearing|you're calling about|you are calling about|"
    r"you(?:'d| would) like to know|asking (?:about|why|whether|if)|why|whether|if|once|as soon as|the moment|until|"
    r"(?:can't|cannot|can not|unable to|not able to) (?:confirm|say|share|tell|check|discuss|see))\b", re.I)
_STATUS_QUESTION_RE = re.compile(rf"\b({_STATUS_WORDS})\b[^?]*\?|\b(status|was it|is it|did it)\b[^?]*\?", re.I)
_DEFERRAL_RE = re.compile(r"verif|\b(once|after|before|can't|cannot|unable|first)\b", re.I)
_AFFIRMATIVE_OPENING_RE = re.compile(r"^\W*(yes|yeah|yep|correct|that's (?:right|correct)|right|indeed|"
                                     r"(?:unfortunately|sadly),? (?:yes|it (?:is|was)))\b", re.I)
_MISSING_DOCS_RE = re.compile(r"\bmissing\s+(documents?|paperwork|records?|files?|reports?)\b", re.I)


def _claim_vocabulary() -> list[str]:
    """Document names and denial-reason wording that only come from claim records."""
    words = set()
    for c in data.claims():
        words.update(d.lower() for d in c.get("documents_needed", []))
    words.update(k.lower() for k in data.document_guideline().get("document_guidance", {}))
    return sorted(words)


def _claim_dates() -> set[tuple]:
    out = set()
    for c in data.claims():
        for k in ("created_at", "appeal_deadline"):
            if c.get(k):
                y, mo, d = map(int, c[k].split("-"))
                out.add((y, mo, d))
    return out


def unverified_violation(s: Session, reply: str) -> str | None:
    """What in a pre-verification reply reads as claim information, or None."""
    said = caller_text(s)
    said_up, said_low, low = said.upper(), said.lower(), reply.lower()

    ids = {claims.normalize_case_id(m) for m in CASE_ID_RE.findall(reply)}
    leaked = sorted(i for i in ids if i and i not in said_up)
    if leaked:
        return f"mentions claim ids {leaked}"
    if money_values(reply) - money_values(said):
        return "mentions a money amount"
    caller_dates = date_values(said)
    for v in date_values(reply):
        if _date_backed(v, _claim_dates()) and not _date_backed(v, caller_dates):
            return "mentions a date from a claim record"
    for word in _claim_vocabulary():
        if word in low and word not in said_low:
            return f"mentions claim document '{word}'"
    for c in data.claims():
        if c.get("denial_reason") and c["denial_reason"].lower() in low:
            return "mentions a denial reason"
    if (m := _MISSING_DOCS_RE.search(reply)) and m.group(0).lower() not in said_low:
        return "says documents are missing"
    if _CLAIM_EXPLANATION_RE.search(reply):
        return "explains a claim decision"
    for sentence in re.split(r"(?<=[.?!])\s+", reply):
        m = _STATUS_ASSERTION_RE.search(sentence)
        if m and not _ATTRIBUTION_RE.search(sentence[:m.end()]):
            return "states or confirms a claim status"
    last_question = next((m["content"] for m in reversed(s.history) if m["role"] == "user"), "")
    first_sentence = re.split(r"(?<=[.?!])\s+", reply.strip(), maxsplit=1)[0]
    if (_STATUS_QUESTION_RE.search(last_question) and _AFFIRMATIVE_OPENING_RE.search(first_sentence)
            and not _DEFERRAL_RE.search(first_sentence)):
        return "confirms the caller's guess about a claim"
    return None


def guard_reply(s: Session, reply: str) -> str | None:
    """Return a description of the disclosure, or None when the reply is safe to show."""
    if s.verified_party_id is None:
        problem = unverified_violation(s, reply)
        return f"unverified reply {problem}" if problem else None

    said = caller_text(s).upper()
    own = {c["case_id"] for c in claims.list_claims(s.verified_party_id)}
    ids = {claims.normalize_case_id(m) for m in CASE_ID_RE.findall(reply)}
    foreign = sorted(i for i in ids if i and i not in own and i not in said)
    if foreign:
        return f"reply mentions claims of another party {foreign}"
    return None


# ---------------------------------------------------------------- shared by grounding.py

_PROMISE_RE = re.compile(
    r"\b(guarantee[sd]?|definitely (?:be |get )?(?:approved|paid|reimbursed)|"
    r"will (?:be|get) (?:approved|paid|reimbursed|overturned|reversed)|"
    r"(?:is|are) (?:certain|sure) to be (?:approved|paid))\b", re.I)
_NEGATION_RE = re.compile(r"\b(not|no|never|nobody|no one|without|neither|nor|cannot|can't|can not|couldn't|won't|"
                          r"unable to|isn't|aren't|don't|doesn't)\b[^.?!]*$", re.I)


# ---------------------------------------------------------------- email

def email_pii_violations(body: str, holder: dict) -> list[str]:
    """Identity details that must never appear in an email summary."""
    problems = []
    if re.search(rf"\b{re.escape(holder['id_last4'])}\b", body):
        problems.append("ID last 4")
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", body):
        problems.append("SSN-like number")
    dob = date.fromisoformat(holder["dob"])
    if (dob.year, dob.month, dob.day) in date_values(body) or holder["dob"] in body:
        problems.append("date of birth")
    phones = {normalize_phone(p) for p in [holder["phone"], *holder.get("phone_aliases", [])]}
    for m in re.finditer(r"\+?\d[\d\s().-]{8,}\d", body):
        if normalize_phone(m.group(0)) in phones:
            problems.append("phone number")
    return problems


def redact_identity(text: str, holder: dict | None) -> str:
    """Mask identity details in a transcript before it goes to the summary model."""
    def _dob(m: re.Match) -> str:
        return "[date of birth]" if holder and normalize_dob(m.group(0)) == holder["dob"] else m.group(0)
    dates = rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b|\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    text = re.sub(dates, _dob, text, flags=re.I)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\+?\d[\d\s().-]{8,}\d", lambda m: "[phone]" if normalize_phone(m.group(0)) else m.group(0), text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[id]", text)
    if holder:
        text = re.sub(rf"\b{re.escape(holder['id_last4'])}\b", "[id]", text)
    return text


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


def missing_directives(plan, reply: str) -> str | None:
    """Directives with a must-say element the reply left out (the model sometimes drops one)."""
    low = reply.lower()
    for d in plan.directives:
        if d["kind"] == "OFFER_EMAIL_SUMMARY" and "email" not in low and "summary" not in low:
            return "left out the offer to email a summary, which the DIRECTIVES require"
        if d["kind"] == "ESCALATE_TO_HUMAN" and d["reference"].lower() not in low:
            return f"left out the transfer reference number {d['reference']}"
        if d["kind"] == "EMAIL_SENT" and not re.search(r"\b(sent|emailed|on its way|in your inbox)\b", low):
            return "did not confirm that the summary email was sent"
    return None


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

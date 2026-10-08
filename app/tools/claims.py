"""Claim lookup and matching. Every lookup is scoped to the verified party."""
from __future__ import annotations

import re
from datetime import date

from app import data

CASE_TYPE_SYNONYMS = {
    "healthcare": "healthcare", "health": "healthcare", "medical": "healthcare", "hospital": "healthcare",
    "dental": "dental", "dentist": "dental",
    "auto": "auto", "car": "auto", "vehicle": "auto", "motor": "auto",
}
STATUS_SYNONYMS = {
    "denied": "denied", "rejected": "denied", "declined": "denied",
    "open": "open", "pending": "open", "in progress": "open", "processing": "open",
    "closed": "closed", "settled": "closed", "paid": "closed", "completed": "closed", "approved": "closed",
}


def normalize_case_type(value: str | None) -> str | None:
    return CASE_TYPE_SYNONYMS.get(value.strip().lower()) if value else None


def normalize_status(value: str | None) -> str | None:
    return STATUS_SYNONYMS.get(value.strip().lower()) if value else None


def normalize_case_id(value: str | None) -> str | None:
    if not value:
        return None
    m = re.search(r"(\d{3,})", value)
    return f"CL-{m.group(1)}" if m else None


def list_claims(party_id: str) -> list[dict]:
    return [c for c in data.claims() if c["party_id"] == party_id]


def get_claim(party_id: str, case_id: str) -> dict | None:
    case_id = normalize_case_id(case_id)
    return next((c for c in list_claims(party_id) if c["case_id"] == case_id), None)


def claim_summary(claim: dict) -> dict:
    return {k: claim[k] for k in ("case_id", "case_type", "created_at", "status", "summary")}


def claim_facts(claim: dict, today: date) -> dict:
    """The full claim record plus computed facts the agent may state."""
    facts = dict(claim)
    facts.pop("party_id", None)
    deadline = claim.get("appeal_deadline")
    if deadline:
        days = (date.fromisoformat(deadline) - today).days
        facts["appeal_deadline_passed"] = days < 0
        facts["days_until_appeal_deadline"] = days
    facts["today"] = today.isoformat()
    return facts


def _hint_checks(claim: dict, hints: dict) -> list[bool]:
    created = date.fromisoformat(claim["created_at"])
    checks = []
    if hints.get("case_type"):
        checks.append(normalize_case_type(hints["case_type"]) == claim["case_type"])
    if hints.get("status"):
        checks.append(normalize_status(hints["status"]) == claim["status"])
    if hints.get("month"):
        checks.append(int(hints["month"]) == created.month)
    if hints.get("year"):
        checks.append(int(hints["year"]) == created.year)
    return checks


def match_claims(party_id: str, hints: dict) -> tuple[list[dict], bool]:
    """Return (candidates, exact).

    exact=True means every candidate satisfies every hint. If nothing satisfies
    all hints, the best partial matches are returned with exact=False so the
    agent can ask the caller to choose.
    """
    claims = list_claims(party_id)
    if hints.get("case_id"):
        hit = get_claim(party_id, hints["case_id"])
        if hit:
            return [hit], True
    scored = []
    for c in claims:
        checks = _hint_checks(c, hints)
        if checks:
            scored.append((sum(checks), len(checks), c))
    if not scored:
        return [], False
    full = [c for s, n, c in scored if s == n]
    if full:
        return full, True
    best = max(s for s, _, _ in scored)
    if best == 0:
        return [], False
    return [c for s, _, c in scored if s == best], False

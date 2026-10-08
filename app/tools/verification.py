"""Deterministic identity verification against policyholders.json.

The LLM only extracts what the caller said. Whether the caller is verified is
decided here, in code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from app import data

# Fields that count toward the verification threshold. policy_number is used
# only to look up the account and never counts.
COUNTED_FIELDS = ("full_name", "dob", "phone", "email", "id_last4")
FIELD_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "id_last4": "last 4 digits of SSN or national ID",
    "policy_number": "policy number",
}

_DOB_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%B %d %Y",
    "%b %d, %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%Y.%m.%d",
)


def normalize_name(value: str) -> str:
    return " ".join(re.sub(r"[^a-z\s]", " ", value.lower()).split())


def normalize_dob(value: str) -> str | None:
    value = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value.strip(), flags=re.I)
    for fmt in _DOB_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_phone(value: str) -> str | None:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_last4(value: str) -> str | None:
    digits = re.sub(r"\D", "", value)
    return digits[-4:] if len(digits) >= 4 else None


def normalize_policy(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _names_equal(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    # Accept reordered tokens ("Chen Margaret") and joined tokens ("Yawen Li").
    return sorted(na.split()) == sorted(nb.split()) or na.replace(" ", "") == nb.replace(" ", "")


def field_matches(holder: dict, field_name: str, value: str) -> bool:
    if not value:
        return False
    if field_name == "full_name":
        names = [holder["name"], *holder.get("name_aliases", [])]
        return any(_names_equal(value, n) for n in names)
    if field_name == "dob":
        return normalize_dob(value) == holder["dob"]
    if field_name == "phone":
        given = normalize_phone(value)
        phones = [holder["phone"], *holder.get("phone_aliases", [])]
        return given is not None and any(normalize_phone(p) == given for p in phones)
    if field_name == "email":
        emails = [holder["email"], *holder.get("email_aliases", [])]
        return normalize_email(value) in {normalize_email(e) for e in emails}
    if field_name == "id_last4":
        return normalize_last4(value) == holder["id_last4"]
    if field_name == "policy_number":
        return normalize_policy(value) == normalize_policy(holder["policy_number"])
    return False


@dataclass
class VerificationResult:
    party_id: str | None = None
    matched: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    provided_count: int = 0
    min_fields: int = 3

    @property
    def verified(self) -> bool:
        return (
            self.party_id is not None
            and len(self.matched) >= self.min_fields
            and not self.mismatched
        )

    @property
    def ready(self) -> bool:
        """Enough counted fields were given to make a pass/fail decision."""
        return self.provided_count >= self.min_fields

    @property
    def failed(self) -> bool:
        return self.ready and not self.verified


def evaluate(provided: dict[str, str], min_fields: int = 3) -> VerificationResult:
    counted = {k: v for k, v in provided.items() if k in COUNTED_FIELDS and v}
    result = VerificationResult(provided_count=len(counted), min_fields=min_fields)

    candidate = None
    policy = provided.get("policy_number")
    if policy:
        candidate = next((h for h in data.policyholders() if field_matches(h, "policy_number", policy)), None)
    if candidate is None:
        scored = [(sum(field_matches(h, k, v) for k, v in counted.items()), h) for h in data.policyholders()]
        scored.sort(key=lambda s: s[0], reverse=True)
        top = [h for score, h in scored if score == scored[0][0] and score > 0] if scored else []
        candidate = top[0] if len(top) == 1 else None

    if candidate is None:
        if policy:
            result.mismatched.append("policy_number")
        return result

    result.party_id = candidate["party_id"]
    if policy and not field_matches(candidate, "policy_number", policy):
        result.mismatched.append("policy_number")
    for k, v in counted.items():
        (result.matched if field_matches(candidate, k, v) else result.mismatched).append(k)
    return result


def is_authorized_representative(rep_name: str, party_id: str) -> dict | None:
    for rep in data.representatives():
        if rep["buyer_party_id"] == party_id and _names_equal(rep_name, rep["rep_name"]):
            return rep
    return None

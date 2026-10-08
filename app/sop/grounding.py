"""Grounding checks: does each factual claim in a reply match what the trusted data says?

Earlier versions only checked that a number appeared somewhere in the context, which let
"your reimbursement is $1,450" through when $1,450 is the allowed maximum, or "review takes
18 months" because 18 appears in a date. These checks are field-aware instead:

- amounts: the role the reply gives an amount (reimbursement, allowed maximum, payment, fee)
  must match a claim field with that value;
- durations: the length (in days) must match a duration in the guidance or the computed
  days until a deadline;
- dates: a date described as a deadline must be an appeal deadline, a filing date a created date;
- status: "CL-2048 is approved" must match the claim's recorded status;
- denial reasons: an explanation of a denial must share content with the recorded reason;
- promises of an outcome are never grounded.

Trusted evidence is claim records, guidance and tool results. What the caller said is kept
separately: it can be acknowledged ("you mentioned $9,999") but never confirmed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from app.config import settings
from app.tools import claims as claims_tool
from app.tools import guidance
from app.tools.claims import normalize_status
from app.sop.gates import (
    CASE_ID_RE, caller_text, date_values, money_values, visible_facts, _DAY_MONTH_RE, _ISO_DATE_RE,
    _MONEY_RE, _MONTH_DAY_RE, _NEGATION_RE, _PROMISE_RE, _US_DATE_RE, _decimal,
)

AMOUNT_FIELDS = ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee")

# What a reply calls an amount -> the claim fields that amount may come from.
AMOUNT_ROLES = {
    "allowed_max": (re.compile(r"\b(allowed|maximum|max|up to|cap|limit|ceiling|eligible|the most|at most|highest)\b", re.I),
                    ("allowed_max_amount",)),
    "reimbursement": (re.compile(r"\b(reimburs\w*|expected|payout|pay out|receive|get back|refund\w*|owed?)\b", re.I),
                      ("expected_reimbursement_amount", "net_pay")),
    "payment": (re.compile(r"\b(net pay|paid|payment|pay)\b", re.I), ("net_pay", "expected_reimbursement_amount")),
    "fee": (re.compile(r"\b(fee|rate)\b", re.I), ("net_fee",)),
}
DATE_ROLES = {
    "deadline": (re.compile(r"\b(deadline|appeal|due|no later than|by)\b", re.I), ("appeal_deadline",)),
    "created": (re.compile(r"\b(filed|submitted|created|opened|received|started|service)\b", re.I), ("created_at",)),
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_DURATION_RE = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|ten)\s+(business\s+|calendar\s+|working\s+)?(day|week|month|year)s?\b", re.I)
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}

_STATUS_WORDS = r"denied|rejected|declined|approved|paid|closed|settled|open|pending|reopened|under review|in review"
_STATUS_RE = re.compile(
    rf"\b(?:claim|case|CL-?\d{{3,}}|it)(?:\s+(?:is|was|has been|had been)|'s been|'s)\s+"
    rf"(?:currently\s+|already\s+|now\s+|still\s+)?({_STATUS_WORDS})\b", re.I)
_HYPOTHETICAL_RE = re.compile(r"\b(if|once|whether|until|unless|would|could|might|hope)\b", re.I)
_REASON_RE = re.compile(
    r"\b(?:denied|denial|rejected|declined)\b[^.?!]{0,60}?\b(?:because|due to|as a result of|"
    r"for (?=lack|missing|incomplete|not|no\b))\s*(?P<a>[^.?!]{3,})"
    r"|\b(?:denial reason|reason for (?:the |this |your )?denial|"
    r"reason (?:it|the claim|your claim|this claim|CL-?\d+) was (?:denied|rejected|declined))"
    r"(?:\s+(?:is|was)\b|\s*:)\s*(?:that\s+)?(?P<b>[^.?!]{3,})", re.I)  # a statement, not "we discussed the reason for the denial"
_STOPWORDS = set("""the a an and or of to in on for with that this was were is are be been being not did does didn
doesn don your you our we it its claim claims denied denial because due file files from have has had any all which
their there they them what when who will would can could should may might also just only so as at by into than""".split())
# A caller's figure may be repeated to acknowledge or dispute it, never to confirm it.
_CALLER_ATTRIBUTION_RE = re.compile(
    r"\byou(?:'ve| have)?\s+(?:said|mentioned|asked|expected|believe|think|told|heard)\b"
    r"|\b(?:doctor|provider|clinic|they|someone)\s+(?:said|told|believes?|mentioned|quoted|estimated|thinks?)\b"
    r"|\b(?:if|whether)\b"
    r"|\b(?:don't|do not|doesn't|didn't|can't|cannot|unable to|not able to|no record|nothing|isn't|aren't|not)\b", re.I)
_AFFIRMATION_RE = re.compile(r"\b(correct|right|yes|confirm\w*|that's what|is accurate|you'?re owed|you will (?:get|receive))\b", re.I)


@dataclass
class Evidence:
    """Trusted facts for one reply, kept apart from what the caller said."""
    claims: dict[str, dict] = field(default_factory=dict)  # case_id -> merged claim fields
    texts: list[str] = field(default_factory=list)          # trusted text (facts, guidance, tool results)
    caller: str = ""                                         # caller statements: acknowledge, never confirm
    primary_case_id: str | None = None                      # the claim "it" / "the claim" refers to
    today: date = field(default_factory=lambda: settings.today)

    def add(self, obj) -> None:
        """Add a facts dict, tool result or guidance text, collecting claim records found inside it."""
        if obj is None:
            return
        self.texts.append(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))
        self._collect(obj)

    def _collect(self, obj) -> None:
        if isinstance(obj, dict):
            cid = obj.get("case_id")
            if cid and ("status" in obj or any(k in obj for k in AMOUNT_FIELDS)):
                self.claims.setdefault(cid, {}).update({k: v for k, v in obj.items() if not isinstance(v, (dict, list))})
                if obj.get("documents_needed"):
                    self.claims[cid]["documents_needed"] = obj["documents_needed"]
            for v in obj.values():
                self._collect(v)
        elif isinstance(obj, list):
            for v in obj:
                self._collect(v)

    def merge(self, other: "Evidence") -> None:
        self.texts += other.texts
        for cid, fields in other.claims.items():
            self.claims.setdefault(cid, {}).update(fields)

    @property
    def text(self) -> str:
        return "\n".join(self.texts)

    def durations(self) -> set[tuple[int, bool]]:
        out = {d for t in self.texts for d in _durations(t)}
        for c in self.claims.values():
            if c.get("appeal_deadline"):
                out.add(((date.fromisoformat(c["appeal_deadline"]) - self.today).days, False))
        return out


def for_session(s, *extra, include_guidance: bool = False) -> Evidence:
    """Evidence from what the session may see now, plus any extra tool results."""
    ev = Evidence(caller=caller_text(s), primary_case_id=s.selected_case_id)
    ev.add(visible_facts(s))
    for obj in extra:
        ev.add(obj)
    if include_guidance and s.verified_party_id:
        for cid in dict.fromkeys([*s.cases_discussed, s.selected_case_id]):
            claim = claims_tool.get_claim(s.verified_party_id, cid) if cid else None
            if claim:
                ev.add(claims_tool.claim_facts(claim, settings.today))
                ev.add(guidance.get_document_guidance(claim))
                ev.add([guidance.get_followup_guidance(claim, topic=t) for t in guidance.followup_topics()])
    return ev


# ---------------------------------------------------------------- helpers

def _durations(text: str) -> set[tuple[int, bool]]:
    out = set()
    for n, business, unit in _DURATION_RE.findall(text):
        count = int(n) if n.isdigit() else _WORD_NUMBERS[n.lower()]
        out.add((count * _UNIT_DAYS[unit.lower()], bool(business.strip())))
    return out


def _sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.?!])\s+|\n+", text) if p.strip()]


def _clause_before(sentence: str, pos: int, width: int = 60) -> str:
    """Text leading up to pos, cut at the previous amount so each amount gets its own label."""
    start = max(0, pos - width)
    prev = [m.end() for m in _MONEY_RE.finditer(sentence, start, pos)]
    return sentence[(prev[-1] if prev else start):pos]


def _roles(text: str, table: dict) -> list[tuple]:
    return [fields for rx, fields in table.values() if rx.search(text)]


def _tokens(text: str) -> set[str]:
    words = {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}
    return words - _STOPWORDS


def _affirms(sentence: str) -> bool:
    """True when the sentence agrees with something ("that's right", "I can confirm"), not "I can't confirm"."""
    def agrees(m: re.Match) -> bool:
        before = sentence[max(0, m.start() - 25):m.start()]
        return not (_NEGATION_RE.search(before) or re.search(r"\b(whether|if)\b", before, re.I))
    return any(agrees(m) for m in _AFFIRMATION_RE.finditer(sentence))


def _claim_for(sentence: str, ev: Evidence) -> dict | None:
    ids = [claims_tool.normalize_case_id(m) for m in CASE_ID_RE.findall(sentence)]
    for cid in ids:
        if cid in ev.claims:
            return ev.claims[cid]
    if ids:
        return None
    return ev.claims.get(ev.primary_case_id) if ev.primary_case_id else None


# ---------------------------------------------------------------- checks

def _check_amounts(sentence: str, ev: Evidence) -> list[str]:
    problems = []
    caller_money = money_values(ev.caller)
    for m in _MONEY_RE.finditer(sentence):
        amount = _decimal(next(g for g in m.groups() if g))
        if amount is None:
            continue
        roles = _roles(_clause_before(sentence, m.start()), AMOUNT_ROLES)
        fields = {f for r in roles for f in r} or set(AMOUNT_FIELDS)
        allowed = {_decimal(c[f]) for c in ev.claims.values() for f in fields if c.get(f)}
        if not roles:
            allowed |= money_values("\n".join(ev.texts))
        if amount in allowed:
            continue
        # The caller's own figure may be mentioned or disputed ("there's no $9,999 on file"),
        # but not agreed with or presented as a claim field ("your reimbursement is $9,999").
        if amount in caller_money and (not roles or _CALLER_ATTRIBUTION_RE.search(sentence)) and not _affirms(sentence):
            continue
        label = " / ".join(sorted({k for k, (_, f) in AMOUNT_ROLES.items() if f in roles})) or "an amount"
        problems.append(f"amount {amount} stated as {label}")
    return problems


_DATE_PATTERNS = (_ISO_DATE_RE, _US_DATE_RE, _MONTH_DAY_RE, _DAY_MONTH_RE)


def _dates_with_position(sentence: str) -> list[tuple[int, tuple]]:
    found = {}
    for rx in _DATE_PATTERNS:
        for m in rx.finditer(sentence):
            for v in date_values(m.group(0)):
                found.setdefault(m.start(), v)
    return sorted(found.items())


def _check_dates(sentence: str, ev: Evidence) -> list[str]:
    problems = []
    known_all = date_values("\n".join(ev.texts))
    dated = _dates_with_position(sentence)
    for i, (pos, v) in enumerate(dated):
        label = "-".join(str(x) for x in v if x is not None)
        # The words just before a date say what it is; stop at the previous date.
        start = max(dated[i - 1][0] + 1 if i else 0, pos - 50)
        roles = _roles(sentence[start:pos], DATE_ROLES)
        if roles:
            known = set()
            for c in ev.claims.values():
                for f in {f for r in roles for f in r}:
                    if c.get(f):
                        y, mo, d = map(int, c[f].split("-"))
                        known.add((y, mo, d))
            if any(k[1:] == v[1:] and (v[0] is None or k[0] == v[0]) for k in known):
                continue
            if not known and any(k[1:] == v[1:] for k in known_all):
                continue
        elif any(k[1:] == v[1:] and (v[0] is None or k[0] is None or k[0] == v[0]) for k in known_all):
            continue
        problems.append(f"date {label}")
    return problems


def _check_durations(sentence: str, ev: Evidence) -> list[str]:
    known = ev.durations()
    known_days = {d for d, _ in known}
    problems = []
    for m in _DURATION_RE.finditer(sentence):
        if not m.group(1).isdigit():
            continue  # "a week", "one day" come straight from guidance wording; numbers are what get invented
        days, business = next(iter(_durations(m.group(0))))
        if (days, business) in known or (not business and days in known_days):
            continue
        problems.append(f"time span '{m.group(0)}'")
    return problems


def _check_status(sentence: str, ev: Evidence) -> list[str]:
    if sentence.rstrip().endswith("?"):
        return []
    problems = []
    for m in _STATUS_RE.finditer(sentence):
        if _HYPOTHETICAL_RE.search(sentence[:m.start()]):
            continue  # "once it's approved", "if the claim is reopened"
        claim = _claim_for(sentence, ev)
        said = normalize_status(m.group(1)) or ("open" if "review" in m.group(1).lower() else None)
        if claim and claim.get("status") and said and said != claim["status"]:
            problems.append(f"status '{m.group(1)}' for {claim.get('case_id')} (recorded: {claim['status']})")
    return problems


def _check_reason(sentence: str, ev: Evidence) -> list[str]:
    problems = []
    if sentence.rstrip().endswith("?"):
        return []
    for m in _REASON_RE.finditer(sentence):
        if _HYPOTHETICAL_RE.search(sentence[:m.start()]):
            continue
        clause = m.group("a") or m.group("b") or ""
        claim = _claim_for(sentence, ev)
        candidates = [claim] if claim else list(ev.claims.values())
        recorded = " ".join(f"{c.get('denial_reason', '')} {c.get('summary', '')} {' '.join(c.get('documents_needed', []))}"
                            for c in candidates if c and (c.get("denial_reason") or c.get("status") == "denied"))
        if not recorded.strip() or not (_tokens(clause) & _tokens(recorded)):
            problems.append(f"denial reason '{clause.strip()[:60]}'")
    return problems


def _check_promises(sentence: str, ev: Evidence) -> list[str]:
    problems = []
    low = ev.text.lower()
    for m in _PROMISE_RE.finditer(sentence):
        # "I can't guarantee it will be approved" is the honest answer, not a promise.
        if _NEGATION_RE.search(sentence[max(0, m.start() - 30):m.start()]):
            continue
        if m.group(0).lower() not in low:
            problems.append(f"promise '{m.group(0)}'")
    return problems


CHECKS = (_check_amounts, _check_dates, _check_durations, _check_status, _check_reason, _check_promises)


def _sentence_problems(sentence: str, ev: Evidence) -> list[str]:
    return [p for check in CHECKS for p in check(sentence, ev)]


def violations(reply: str, ev: Evidence) -> list[str]:
    """Every statement in the reply that the trusted evidence does not support."""
    problems = [p for sentence in _sentences(reply) for p in _sentence_problems(sentence, ev)]
    return list(dict.fromkeys(problems))


def prune(reply: str, ev: Evidence) -> tuple[str, int]:
    """Drop the sentences the evidence doesn't support, keeping the rest and the layout.

    Returns (pruned reply, number of sentences removed).
    """
    removed = 0
    lines = []
    for line in reply.split("\n"):
        kept = []
        for sentence in re.split(r"(?<=[.?!])\s+", line):
            if sentence.strip() and _sentence_problems(sentence, ev):
                removed += 1
            else:
                kept.append(sentence)
        lines.append(" ".join(kept).rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text, removed

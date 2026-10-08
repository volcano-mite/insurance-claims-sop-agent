"""Grounding checks: does each factual claim in a reply match what the trusted data says?

Checking that a number appears somewhere in the context is not enough ("review takes 17 days"
would pass because 17 days is the time left to appeal). Each statement is checked for what it
claims, against the claim it is about:

- every amount, date and status is bound to a claim: the nearest claim id in the sentence,
  else the claim the reply was last talking about, else the selected claim;
- amounts: the role the reply gives an amount (reimbursement, allowed maximum, payment, fee)
  must match that claim's field with that value;
- dates: a deadline must be that claim's appeal deadline, a filing date its created date;
- durations (digits or words) are checked by purpose: time left to appeal must match the days
  until that claim's deadline; review, submission and status-update times must match a duration
  the guidance gives for the same purpose. Each is compared as a range with its kind of days
  and hedge: "usually less than a week" supports neither "7 business days" nor "takes 7 days";
- status must match the claim's recorded status;
- denial reasons are a controlled fact: when the record says documents were missing, the reply
  must say they were missing (not "were not missing" or "were submitted") and may only name
  documents on the claim's list; a document name the record can't confirm ("police report")
  is not grounded; reasons the record doesn't give (not covered, excluded, lapsed policy...)
  are never grounded;
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
    _MONEY_RE, _MONTH_DAY_RE, _NEGATION_RE, _PROMISE_RE, _US_DATE_RE, _claim_vocabulary, _decimal,
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
    # not "submitted"/"received": those usually describe the caller's documents, not the claim
    "created": (re.compile(r"\b(filed|created|opened|service)\b", re.I), ("created_at",)),
}
_DEADLINE_NOUN_AFTER_RE = re.compile(r"(?:,?\s+\d{4})?\s+(?:appeal\s+)?(deadline|cutoff)\b", re.I)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15,
                 "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90}
_NUMBER = r"\d+|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True))
_DURATION_RE = re.compile(
    rf"\b({_NUMBER})\s+(business\s+|calendar\s+|working\s+)?(day|week|month|year)s?\b", re.I)
# What a duration is about, from the words before it ("review takes", "submit within") or after it ("left").
DURATION_PURPOSES = {
    "deadline": re.compile(r"\b(deadline|appeal|expires?|to respond)\b", re.I),
    "review": re.compile(r"\b(review\w*|process\w*|takes?|taking|decision|hear back|turnaround|reconsider\w*)\b", re.I),
    "submit": re.compile(r"\b(submit\w*|send|upload\w*|provide)\b", re.I),
    "update": re.compile(r"\b(status|update\w*|reflect\w*|show up|intake)\b", re.I),
}
_DEADLINE_AFTER_RE = re.compile(r"\s*(left|remaining|away|until|to (?:appeal|respond)|before the deadline|from today)\b", re.I)
# A duration's bound comes from the words right before the number: "less than a week" is 0-6 days.
_BOUND_RES = (
    ("lt", re.compile(r"\b(?:less than|under|fewer than|shorter than)\s+$", re.I)),
    ("le", re.compile(r"\b(?:within|up to|at most|no more than|not more than|no longer than|inside of)\s+$", re.I)),
    ("gt", re.compile(r"\b(?:more than|over|longer than)\s+$", re.I)),
    ("ge", re.compile(r"\b(?:at least|no less than|not less than|a minimum of)\s+$", re.I)),
    ("approx", re.compile(r"\b(?:about|around|roughly|nearly|almost|approximately|some)\s+$", re.I)),
)
_APPROX_AFTER_RE = re.compile(r"\s+or so\b", re.I)
_TOLERANCE_DAYS = {"week": 3, "month": 15, "year": 60}
# "usually less than a week" is a typical time, not a promise; a reply may not drop the hedge.
_HEDGE_RE = re.compile(r"\b(?:usually|typically|generally|normally|often|average|most|tends? to|"
                       r"may|might|can|could|expect)\b", re.I)
_HEDGE_AFTER_RE = re.compile(r"\s+(?:on average|usually|typically|in most cases)\b", re.I)
_CLAUSE_BREAK_RE = re.compile(r"[,;:]|\b(?:and|but|although|though|while|which|so)\b", re.I)


@dataclass(frozen=True)
class Span:
    """A stated duration as a range of days (hi None = no upper end), in calendar or business days."""
    lo: int
    hi: int | None
    business: bool = False
    hedged: bool = False

    def within(self, other: "Span") -> bool:
        """True when every duration this span allows is also allowed by other."""
        return (self.lo >= other.lo
                and (other.hi is None or (self.hi is not None and self.hi <= other.hi)))

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

    def durations_by_purpose(self) -> dict[str | None, set[Span]]:
        """Durations stated in trusted text, keyed by what they are about (None = unclear)."""
        out: dict[str | None, set[Span]] = {}
        for t in self.texts:
            for m in _DURATION_RE.finditer(t):
                out.setdefault(_purpose(t, m.start(), m.end()), set()).add(_span(t, m))
        return out

    def days_to_deadline(self, claim: dict | None) -> set[Span]:
        """Calendar days left to appeal, computed from each claim's deadline."""
        claims = [claim] if claim else list(self.claims.values())
        days = {(date.fromisoformat(c["appeal_deadline"]) - self.today).days
                for c in claims if c and c.get("appeal_deadline")}
        return {Span(d, d) for d in days}


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

def _days(m: re.Match) -> int:
    n = m.group(1).lower()
    return (int(n) if n.isdigit() else _WORD_NUMBERS[n]) * _UNIT_DAYS[m.group(3).lower()]


def _span(text: str, m: re.Match) -> Span:
    """What a duration match claims: its range of days, the kind of days, and whether it is hedged."""
    days, unit = _days(m), m.group(3).lower()
    before, after = text[max(0, m.start() - 80):m.start()], text[m.end():m.end() + 20]
    bound = next((name for name, rx in _BOUND_RES if rx.search(before)), None)
    if bound is None and _APPROX_AFTER_RE.match(after):
        bound = "approx"
    tol = _TOLERANCE_DAYS.get(unit, 0)
    lo, hi = {"lt": (0, days - 1), "le": (0, days), "gt": (days + 1, None), "ge": (days, None),
              "approx": (max(0, days - tol), days + tol)}.get(bound, (days, days))
    breaks = list(_CLAUSE_BREAK_RE.finditer(before))
    clause = before[breaks[-1].end():] if breaks else before
    hedged = bool(_HEDGE_RE.search(clause) or _HEDGE_AFTER_RE.match(after))
    return Span(lo, hi, business=bool(m.group(2)) and m.group(2).lower().strip() != "calendar", hedged=hedged)


def _purpose(text: str, start: int, end: int) -> str | None:
    if _DEADLINE_AFTER_RE.match(text, end):
        return "deadline"
    window = text[max(0, start - 70):start]
    best, best_end = None, -1
    for name, rx in DURATION_PURPOSES.items():
        for m in rx.finditer(window):
            if m.end() > best_end:
                best, best_end = name, m.end()
    return best


def _sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.?!])\s+|\n+", text) if p.strip()]


def _clause_before(sentence: str, pos: int, width: int = 100) -> str:
    """Text leading up to pos, cut at the previous amount so each amount gets its own label.

    Parenthesised explanations are dropped: in "Allowed maximum (the most the insurer would
    pay): $800" the label is "Allowed maximum", not the "pay" in the explanation.
    """
    start = max(0, pos - width)
    prev = [m.end() for m in _MONEY_RE.finditer(sentence, start, pos)]
    return re.sub(r"\([^()]*\)", " ", sentence[(prev[-1] if prev else start):pos])


def _roles(text: str, table: dict) -> list[tuple]:
    return [fields for rx, fields in table.values() if rx.search(text)]


def _amount_roles(clause: str) -> list[tuple]:
    """Roles of the label nearest the amount, plus words joined to it ("allowed maximum
    reimbursement", "receive up to"). An earlier, separate phrase in the clause doesn't count:
    in "the maximum didn't matter; your reimbursement is $X" only "reimbursement" labels $X.
    """
    hits = [(m.start(), m.end(), fields) for rx, fields in AMOUNT_ROLES.values() for m in rx.finditer(clause)]
    if not hits:
        return []
    nearest_start = max(hits, key=lambda h: h[1])[0]
    return list({fields for start, end, fields in hits if end >= nearest_start - 25})


def _tokens(text: str) -> set[str]:
    words = {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}
    return words - _STOPWORDS


def _affirms(sentence: str) -> bool:
    """True when the sentence agrees with something ("that's right", "I can confirm"), not "I can't confirm"."""
    def agrees(m: re.Match) -> bool:
        before = sentence[max(0, m.start() - 25):m.start()]
        return not (_NEGATION_RE.search(before) or re.search(r"\b(whether|if)\b", before, re.I))
    return any(agrees(m) for m in _AFFIRMATION_RE.finditer(sentence))


def _case_at(sentence: str, pos: int, current: str | None) -> str | None:
    """The claim a value at pos is about: nearest claim id before it, else after it, else the current one."""
    ids = list(CASE_ID_RE.finditer(sentence))
    before = [m for m in ids if m.end() <= pos]
    after = [m for m in ids if m.start() >= pos]
    hit = before[-1] if before else (after[0] if after else None)
    return claims_tool.normalize_case_id(hit.group(0)) if hit else current


def _candidates(ev: Evidence, case_id: str | None) -> list[dict]:
    """The claim records a value may come from: the bound claim only, or all when none is bound.

    A bound claim with no record in the evidence gives no candidates: the model has no data
    for it, so any value it states about it is unsupported.
    """
    if case_id is None:
        return list(ev.claims.values())
    return [ev.claims[case_id]] if case_id in ev.claims else []


def _ymd(iso: str) -> tuple[int, int, int]:
    y, mo, d = map(int, iso.split("-"))
    return y, mo, d


# ---------------------------------------------------------------- checks
# Each check gets the sentence, the evidence and the claim the reply was last talking about.

def _check_amounts(sentence: str, ev: Evidence, current: str | None) -> list[str]:
    problems = []
    caller_money = money_values(ev.caller)
    for m in _MONEY_RE.finditer(sentence):
        amount = _decimal(next(g for g in m.groups() if g))
        if amount is None:
            continue
        case_id = _case_at(sentence, m.start(), current)
        roles = _amount_roles(_clause_before(sentence, m.start()))
        fields = {f for r in roles for f in r} or set(AMOUNT_FIELDS)
        allowed = {_decimal(c[f]) for c in _candidates(ev, case_id) for f in fields if c.get(f)}
        if amount in allowed:
            continue
        # The caller's own figure may be mentioned or disputed ("there's no $9,999 on file"),
        # but not agreed with or presented as a claim field ("your reimbursement is $9,999").
        if amount in caller_money and (not roles or _CALLER_ATTRIBUTION_RE.search(sentence)) and not _affirms(sentence):
            continue
        label = " / ".join(sorted({k for k, (_, f) in AMOUNT_ROLES.items() if f in roles})) or "an amount"
        problems.append(f"amount {amount} stated as {label}" + (f" for {case_id}" if case_id else ""))
    return problems


_DATE_PATTERNS = (_ISO_DATE_RE, _US_DATE_RE, _MONTH_DAY_RE, _DAY_MONTH_RE)


def _dates_with_position(sentence: str) -> list[tuple[int, tuple]]:
    found = {}
    for rx in _DATE_PATTERNS:
        for m in rx.finditer(sentence):
            for v in date_values(m.group(0)):
                found.setdefault(m.start(), v)
    return sorted(found.items())


def _check_dates(sentence: str, ev: Evidence, current: str | None) -> list[str]:
    problems = []
    dated = _dates_with_position(sentence)
    today = (ev.today.year, ev.today.month, ev.today.day)
    for i, (pos, v) in enumerate(dated):
        case_id = _case_at(sentence, pos, current)
        # The words just before a date say what it is; stop at the previous date.
        start = max(dated[i - 1][0] + 1 if i else 0, pos - 50)
        roles = _roles(sentence[start:pos], DATE_ROLES)
        end = pos + len(next(m.group(0) for rx in _DATE_PATTERNS if (m := rx.match(sentence, pos))))
        if _DEADLINE_NOUN_AFTER_RE.match(sentence, end):  # "before the March 18 deadline"
            roles = [DATE_ROLES["deadline"][1]]
        fields = {f for r in roles for f in r} or {"created_at", "appeal_deadline"}
        known = {_ymd(c[f]) for c in _candidates(ev, case_id) for f in fields if c.get(f)}
        if not roles:
            known.add(today)
        if any(k[1:] == v[1:] and (v[0] is None or k[0] == v[0]) for k in known):
            continue
        label = "-".join(str(x) for x in v if x is not None)
        problems.append(f"date {label}" + (f" for {case_id}" if case_id else ""))
    return problems


def _check_durations(sentence: str, ev: Evidence, current: str | None) -> list[str]:
    trusted = ev.durations_by_purpose()
    problems = []
    for m in _DURATION_RE.finditer(sentence):
        said, purpose = _span(sentence, m), _purpose(sentence, m.start(), m.end())
        deadline_days = ev.days_to_deadline(ev.claims.get(_case_at(sentence, m.start(), current) or ""))
        if purpose == "deadline":
            allowed = deadline_days
        elif purpose:
            allowed = trusted.get(purpose, set()) | trusted.get(None, set())
        else:
            allowed = set().union(*trusted.values(), deadline_days)
        # Supported when a trusted duration implies it: same kind of days, a range inside the
        # reply's range, and the reply keeps any hedge. "less than a week" supports "within a
        # week", not "7 days" or "7 business days"; "usually less than a week" not "less than a week".
        # The upper end may not be stretched past the data ("submit within ten days" when the
        # guidance says within a week), beyond rounding to the reply's unit.
        slack = _TOLERANCE_DAYS.get(m.group(3).lower(), 1)
        if any(t.business == said.business and t.within(said) and (said.hedged or not t.hedged)
               and (said.hi is None) == (t.hi is None) and (said.hi is None or said.hi - t.hi <= slack)
               for t in allowed):
            continue
        problems.append(f"time span '{m.group(0)}'" + (f" as {purpose} time" if purpose else ""))
    return problems


def _check_status(sentence: str, ev: Evidence, current: str | None) -> list[str]:
    if sentence.rstrip().endswith("?"):
        return []
    problems = []
    for m in _STATUS_RE.finditer(sentence):
        if _HYPOTHETICAL_RE.search(sentence[:m.start()]):
            continue  # "once it's approved", "if the claim is reopened"
        case_id = _case_at(sentence, m.start(), current)
        claim = ev.claims.get(case_id) if case_id else None
        said = normalize_status(m.group(1)) or ("open" if "review" in m.group(1).lower() else None)
        if claim and claim.get("status") and said and said != claim["status"]:
            problems.append(f"status '{m.group(1)}' for {case_id} (recorded: {claim['status']})")
    return problems


# Denial reasons are checked as a controlled fact, not by shared words: "the pathology report
# showed the treatment isn't covered" shares words with "the pathology report is missing" but
# states a different reason.
_ABSENCE_RE = re.compile(
    r"\b(?:missing|lack\w*|without|absent|incomplete|outstanding|needed|required|"
    r"no (?:\w+ )?(?:report|note|documents?|records?|paperwork))\b"
    r"|(?:\bnot|n't) (?:been )?(?:include\w*|submit\w*|receiv\w*|provid\w*|sent|attach\w*|"
    r"in the (?:\w+ )?(?:file|record|submission)|on file)\b", re.I)
_OTHER_REASON_RE = re.compile(
    r"\b(?:not|isn't|wasn't|aren't|weren't|never|n't) (?:be |been )?covered\b|\bno coverage\b|"
    r"\bcoverage (?:was |has been )?(?:denied|excluded|terminated|ended)\b|\bexclu(?:ded|sion)\b|"
    r"\bnot medically necessary\b|\bmedical(?:ly)? necess\w*\b|\bpre-?existing\b|\bout[- ]of[- ]network\b|"
    r"\b(?:policy|plan|coverage) (?:had |has )?(?:lapsed|expired|was inactive|was cancell?ed|ended)\b|"
    r"\bfraud\w*\b|\bduplicate\b|\b(?:limit|maximum) (?:was |has been )?(?:exceeded|reached)\b|"
    r"\bnot eligible\b|\bineligible\b", re.I)


# "not missing", "nothing was missing", "was submitted": the opposite of an absence.
_ABSENCE_NEGATED_RE = re.compile(r"(?:\b(?:not|never|nothing|none|no|no longer|neither|nor)\b|n't)\W+(?:\w+\W+){0,3}$", re.I)
_PRESENCE_RE = re.compile(
    r"\b(?:was|were|has been|have been|had been)\s+(?:already\s+|actually\s+|in fact\s+)?"
    r"(?:submitted|included|received|provided|attached|sent|on file|in the (?:\w+ )?(?:file|record))\b", re.I)
# Words naming a specific document; the words before one ("police report") say which document.
# Generic words ("documents", "paperwork") name none and are always fine.
_DOC_HEAD_RE = re.compile(r"\b(?:reports?|notes?|records?|forms?|estimates?|photos?|photographs?|statements?|"
                          r"bills?|invoices?|receipts?|letters?|certificates?|scans?|x-rays?|prescriptions?|"
                          r"referrals?|authorizations?|summar(?:y|ies)|results?)\b", re.I)
_NOT_DOC_MODIFIERS = _STOPWORDS | set("""missing lacking needed required requested outstanding incomplete include
included including submit submitted provide provided receive received send sent attach attached without both
either neither no nor new copy copies signed two second other those these""".split())


def _recorded_reason(claim: dict) -> tuple[str, set[str]]:
    """(recorded reason text, documents the record says are missing)."""
    text = f"{claim.get('denial_reason', '')} {claim.get('summary', '')}".lower()
    docs = {d.lower() for d in claim.get("documents_needed", [])}
    docs |= {w for w in _claim_vocabulary() if w in text}
    return text, docs


def _named_documents(clause: str) -> set[str]:
    """Specific documents a clause names ("the police report" -> "police report"); bare heads are generic."""
    names = set()
    for m in _DOC_HEAD_RE.finditer(clause):
        words = re.findall(r"[a-z][a-z'-]*", clause[:m.start()].lower())
        modifiers = []
        for w in reversed(words[-4:]):
            if w in _NOT_DOC_MODIFIERS:
                break
            modifiers.insert(0, w)
        if modifiers:
            names.add(" ".join([*modifiers, m.group(0).lower()]))
    names |= {w for w in _claim_vocabulary() if w in clause.lower()}
    return names


def _says_absent(clause: str) -> bool:
    """The clause says documents were missing, and never that one was present after all."""
    if _PRESENCE_RE.search(clause):
        return False
    absences = list(_ABSENCE_RE.finditer(clause))
    for m in absences:
        before = clause[max(0, m.start() - 40):m.start()]
        breaks = list(_CLAUSE_BREAK_RE.finditer(before))
        before = before[breaks[-1].end():] if breaks else before
        # "not in the file" carries its own negation; "was not missing" negates the absence
        if not m.group(0).lower().startswith(("not", "n't")) and _ABSENCE_NEGATED_RE.search(before):
            return False
    return bool(absences)


def _reason_matches(clause: str, claim: dict) -> bool:
    if _OTHER_REASON_RE.search(clause):
        return False
    text, docs = _recorded_reason(claim)
    if docs or _ABSENCE_RE.search(text):
        # every document named must be one the record lists: "police report" and unknown names fail
        named_ok = all(any(d in w or w in d for d in docs) for w in _named_documents(clause))
        return _says_absent(clause) and named_ok
    return bool(_tokens(clause) & _tokens(text))  # a record that isn't about missing documents


def _check_reason(sentence: str, ev: Evidence, current: str | None) -> list[str]:
    if sentence.rstrip().endswith("?"):
        return []
    problems = []
    for m in _REASON_RE.finditer(sentence):
        if _HYPOTHETICAL_RE.search(sentence[:m.start()]):
            continue
        clause = m.group("a") or m.group("b") or ""
        denied = [c for c in _candidates(ev, _case_at(sentence, m.start(), current))
                  if c.get("denial_reason") or c.get("status") == "denied"]
        if not any(_reason_matches(clause, c) for c in denied):
            problems.append(f"denial reason '{clause.strip()[:60]}'")
    return problems


def _check_other_reasons(sentence: str, ev: Evidence, current: str | None) -> list[str]:
    """Coverage-type reasons the records never give, wherever they appear ("this isn't covered")."""
    m = _OTHER_REASON_RE.search(sentence)
    if not m or _HYPOTHETICAL_RE.search(sentence[:m.start()]) or m.group(0).lower() in ev.text.lower():
        return []
    return [f"unsupported reason '{m.group(0)}'"]


def _check_promises(sentence: str, ev: Evidence, current: str | None) -> list[str]:
    problems = []
    low = ev.text.lower()
    for m in _PROMISE_RE.finditer(sentence):
        # "I can't guarantee it will be approved" is the honest answer, not a promise.
        if _NEGATION_RE.search(sentence[max(0, m.start() - 30):m.start()]):
            continue
        if m.group(0).lower() not in low:
            problems.append(f"promise '{m.group(0)}'")
    return problems


CHECKS = (_check_amounts, _check_dates, _check_durations, _check_status, _check_reason,
          _check_other_reasons, _check_promises)


def _walk(reply: str, ev: Evidence):
    """Yield (line index, sentence, problems), tracking which claim the reply is talking about."""
    current = ev.primary_case_id
    for i, line in enumerate(reply.split("\n")):
        for sentence in re.split(r"(?<=[.?!])\s+", line):
            if not sentence.strip():
                yield i, sentence, []
                continue
            yield i, sentence, [p for check in CHECKS for p in check(sentence, ev, current)]
            ids = CASE_ID_RE.findall(sentence)
            if ids:
                current = claims_tool.normalize_case_id(ids[-1])


def violations(reply: str, ev: Evidence) -> list[str]:
    """Every statement in the reply that the trusted evidence does not support."""
    return list(dict.fromkeys(p for _, _, problems in _walk(reply, ev) for p in problems))


def prune(reply: str, ev: Evidence) -> tuple[str, int]:
    """Drop the sentences the evidence doesn't support, keeping the rest and the layout.

    Returns (pruned reply, number of sentences removed).
    """
    lines: dict[int, list[str]] = {}
    removed = 0
    for i, sentence, problems in _walk(reply, ev):
        if problems:
            removed += 1
        else:
            lines.setdefault(i, []).append(sentence)
    n = len(reply.split("\n"))
    text = "\n".join(" ".join(lines.get(i, [])).rstrip() for i in range(n))
    return re.sub(r"\n{3,}", "\n\n", text).strip(), removed

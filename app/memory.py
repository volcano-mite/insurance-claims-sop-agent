"""Session memory, per-turn extraction schema and the controller's turn plan."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator

from app.tools.consent import ConsentRequest
from app.tools.verification import COUNTED_FIELDS, FIELD_LABELS

INTENTS = ("status_inquiry", "denial_question", "document_submission", "next_steps", "general_claim_question")
EMOTIONS = ("none", "frustration", "anxiety", "anger", "confusion", "refusal")
MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    ESCALATED = "ESCALATED"
    ENDED = "ENDED"


# ---------------------------------------------------------------- extraction

def _none_if_blank(v: Any) -> Any:
    if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a", "unknown"):
        return None
    return v


class Identity(BaseModel):
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    id_last4: Optional[str] = None
    policy_number: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _blank(cls, v):
        return None if v is None else _none_if_blank(str(v))


class CaseHints(BaseModel):
    case_type: Optional[str] = None
    status: Optional[str] = None
    month: Optional[int] = None
    year: Optional[int] = None
    case_id: Optional[str] = None

    @field_validator("case_type", "status", "case_id", mode="before")
    @classmethod
    def _str(cls, v):
        return None if v is None else _none_if_blank(str(v))

    @field_validator("month", mode="before")
    @classmethod
    def _month(cls, v):
        v = _none_if_blank(v)
        if isinstance(v, str):
            low = v.strip().lower()
            for i, name in enumerate(MONTHS, 1):
                if name.startswith(low[:3]) and len(low) >= 3:
                    return i
            return int(low) if low.isdigit() else None
        return v if v is None or 1 <= int(v) <= 12 else None

    @field_validator("year", mode="before")
    @classmethod
    def _year(cls, v):
        v = _none_if_blank(v)
        try:
            return None if v is None else int(v)
        except (TypeError, ValueError):
            return None

    def present(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v not in (None, "")}


class Extraction(BaseModel):
    """What the extractor LLM found in one caller message. It never decides anything."""
    identity: Identity = Identity()
    caller_role: Optional[Literal["policyholder", "representative"]] = None
    representative_name: Optional[str] = None
    relationship: Optional[str] = None
    intent: Optional[str] = None
    case_hints: CaseHints = CaseHints()
    selected_case_id: Optional[str] = None
    switch_case: bool = False
    in_scope: bool = True
    off_topic_summary: Optional[str] = None
    emotion: str = "none"
    emotion_intensity: Literal["low", "medium", "high"] = "low"
    refuses_verification: bool = False
    declined_fields: list[str] = []
    wants_human: bool = False
    accepts_human_offer: Optional[bool] = None
    wants_to_end: bool = False
    email_decision: Optional[Literal["send", "skip"]] = None

    @field_validator("caller_role", "representative_name", "relationship", "selected_case_id",
                     "off_topic_summary", "email_decision", mode="before")
    @classmethod
    def _blank(cls, v):
        return _none_if_blank(v)

    @field_validator("intent", mode="before")
    @classmethod
    def _intent(cls, v):
        v = _none_if_blank(v)
        return v if v in INTENTS else None

    @field_validator("emotion", mode="before")
    @classmethod
    def _emotion(cls, v):
        v = (_none_if_blank(v) or "none")
        return v if v in EMOTIONS else "none"

    @field_validator("emotion_intensity", mode="before")
    @classmethod
    def _intensity(cls, v):
        return v if v in ("low", "medium", "high") else "low"

    @field_validator("declined_fields", mode="before")
    @classmethod
    def _declined(cls, v):
        return [f for f in (v or []) if f in COUNTED_FIELDS]

    @field_validator("identity", "case_hints", mode="before")
    @classmethod
    def _obj(cls, v):
        return v or {}


# ---------------------------------------------------------------- turn plan

@dataclass
class TurnPlan:
    """What the controller decided for this turn. The responder only phrases it."""
    directives: list[dict] = field(default_factory=list)
    tone: str = ""
    actions: list[str] = field(default_factory=list)
    tools_enabled: bool = False

    def add(self, kind: str, **data: Any) -> None:
        self.directives.append({"kind": kind, **data})

    def has(self, kind: str) -> bool:
        return any(d["kind"] == kind for d in self.directives)


# ---------------------------------------------------------------- session

@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    api_key: Optional[str] = None
    consent_scenario: str = "default"
    phase: Phase = Phase.VERIFY_ID
    turn: int = 0
    history: list[dict] = field(default_factory=list)

    # VERIFY_ID memory
    identity: dict[str, str] = field(default_factory=dict)
    caller_role: Optional[str] = None
    rep_name: Optional[str] = None
    relationship: Optional[str] = None
    declined_fields: list[str] = field(default_factory=list)
    verified_party_id: Optional[str] = None
    verify_failures: int = 0
    last_verification: dict = field(default_factory=dict)
    consent: Optional[ConsentRequest] = None

    # Cross-phase memory: captured whenever the caller says it
    pending_intent: Optional[str] = None
    case_hints: dict = field(default_factory=dict)
    candidate_case_ids: list[str] = field(default_factory=list)
    selected_case_id: Optional[str] = None
    topics_discussed: list[str] = field(default_factory=list)

    # Guardrails
    off_topic_streak: int = 0
    human_offered: bool = False
    gate_refusals: int = 0
    emotion_history: list[dict] = field(default_factory=list)
    escalation: Optional[dict] = None

    # POST_PROCESS
    email_decision: Optional[str] = None
    outbox: list[dict] = field(default_factory=list)

    trace: list[dict] = field(default_factory=list)

    def log(self, event: str, **data: Any) -> None:
        self.trace.append({"turn": self.turn, "event": event, **data})

    def identity_labels(self) -> list[str]:
        return [FIELD_LABELS[k] for k in COUNTED_FIELDS if self.identity.get(k)]

    def snapshot(self) -> dict:
        """State for the debug panel. A testing aid, not something a caller would see."""
        return {
            "session_id": self.id,
            "phase": self.phase.value,
            "turn": self.turn,
            "verification": {
                "verified_party_id": self.verified_party_id,
                "caller_role": self.caller_role,
                "representative": self.rep_name,
                "fields_provided": {k: self.identity.get(k) for k in (*COUNTED_FIELDS, "policy_number")
                                    if self.identity.get(k)},
                "last_check": self.last_verification,
                "declined_fields": self.declined_fields,
                "failures": self.verify_failures,
                "consent": self.consent.to_dict() if self.consent else None,
            },
            "memory": {
                "pending_intent": self.pending_intent,
                "case_hints": self.case_hints,
                "candidate_case_ids": self.candidate_case_ids,
                "selected_case_id": self.selected_case_id,
                "topics_discussed": self.topics_discussed,
            },
            "guardrails": {
                "off_topic_streak": self.off_topic_streak,
                "human_offered": self.human_offered,
                "gate_refusals": self.gate_refusals,
                "emotion": self.emotion_history[-1] if self.emotion_history else None,
                "escalation": self.escalation,
            },
            "email_decision": self.email_decision,
            "outbox": self.outbox,
            "trace": self.trace[-200:],
        }

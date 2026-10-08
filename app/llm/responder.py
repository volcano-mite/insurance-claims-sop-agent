"""Turns the controller's plan into a natural reply. Tools are only exposed in PROCESS_CASE."""
from __future__ import annotations

import json
from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app import data
from app.config import settings
from app.llm import prompts
from app.llm.client import LLM, LLMAuthError, LLMError
from app.memory import Session, TurnPlan
from app.sop import gates, grounding
from app.sop.grounding import Evidence
from app.tools import claims, guidance

MAX_TOOL_ROUNDS = 4

TOOLS = [
    {"type": "function", "function": {
        "name": "get_claim",
        "description": "Get the full record of one of the verified caller's claims.",
        "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"]},
    }},
    {"type": "function", "function": {
        "name": "list_claims",
        "description": "List all claims of the verified caller (id, type, date, status, summary).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_document_guidance",
        "description": ("Requirements for each document a claim still needs, what to do if a document is "
                        "unavailable, how to submit, and processing time after submission."),
        "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_followup_guidance",
        "description": ("Official answers to follow-up questions about a claim: submission timing, submission "
                        "method, processing time, file formats, receipt confirmation, missing material "
                        "alternatives. Pass the caller's question; optionally a topic."),
        "parameters": {"type": "object", "properties": {
            "case_id": {"type": "string"},
            "question": {"type": "string", "description": "The caller's question, in their words."},
            "topic": {"type": "string", "enum": guidance.followup_topics()},
        }, "required": ["case_id", "question"]},
    }},
]


class _Args(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _CaseArgs(_Args):
    case_id: Optional[str] = None

    @field_validator("case_id", mode="before")
    @classmethod
    def _id(cls, v):
        return None if v in (None, "") else str(v)[:40]


class _FollowupArgs(_CaseArgs):
    question: str = ""
    topic: Optional[str] = None

    @field_validator("question", mode="before")
    @classmethod
    def _question(cls, v):
        return "" if v is None else str(v)[:500]

    @field_validator("topic", mode="before")
    @classmethod
    def _topic(cls, v):
        return v if v in guidance.followup_topics() else None


TOOL_ARGS: dict[str, type[_Args]] = {
    "get_claim": _CaseArgs,
    "list_claims": _Args,
    "get_document_guidance": _CaseArgs,
    "get_followup_guidance": _FollowupArgs,
}


def parse_tool_args(name: str, raw: str | None) -> tuple[dict, str | None]:
    """Validate a model-written tool call. Returns (args, error); the error goes back to the model."""
    if name not in TOOL_ARGS:
        return {}, f"unknown tool {name}"
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}, "arguments are not valid JSON"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    try:
        return TOOL_ARGS[name].model_validate(parsed).model_dump(), None
    except ValidationError as e:
        return {}, f"invalid arguments: {e.errors()[0]['msg']}"


def run_tool(s: Session, name: str, args: dict) -> dict:
    """Execute a validated tool call, scoped to the verified party. Unknown or foreign claims return an error."""
    party = s.verified_party_id
    if party is None:
        return {"error": "caller not verified"}
    if name == "list_claims":
        return {"claims": [claims.claim_summary(c) for c in claims.list_claims(party)]}
    claim = claims.get_claim(party, args.get("case_id") or s.selected_case_id or "")
    if claim is None:
        return {"error": "no such claim for this caller"}
    if name == "get_claim":
        return claims.claim_facts(claim, settings.today)
    if name == "get_document_guidance":
        return guidance.get_document_guidance(claim)
    if name == "get_followup_guidance":
        return guidance.get_followup_guidance(claim, args.get("question") or "", args.get("topic"))
    return {"error": f"unknown tool {name}"}


def _assistant_tool_message(msg) -> dict:
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ],
    }


def prefetch_guidance(s: Session, log: bool = True) -> dict:
    """Look up the guidance a PROCESS_CASE answer usually needs, before calling the model.

    Same tool functions and data as the model's tools; doing it in code saves a full
    model round trip per tool call. Tools stay available for anything not covered here.
    """
    claim = claims.get_claim(s.verified_party_id, s.selected_case_id)
    out: dict = {}
    if claim.get("documents_needed"):
        out["document_guidance"] = guidance.get_document_guidance(claim)
        if log:
            s.log("tool_call", tool="get_document_guidance", args={"case_id": claim["case_id"]}, ok=True, prefetched=True)
    question = next((m["content"] for m in reversed(s.history) if m["role"] == "user"), "")
    followup = guidance.get_followup_guidance(claim, question)
    if followup.get("matched_topics"):
        out["followup_guidance_for_this_question"] = followup["matched_topics"]
        if log:
            s.log("tool_call", tool="get_followup_guidance", args={"case_id": claim["case_id"]}, ok=True, prefetched=True)
    return out


def respond(llm: LLM, s: Session, plan: TurnPlan, correction: str | None = None) -> tuple[str, Evidence]:
    """Write the reply. correction is set when a previous draft was rejected by a check.

    Returns (reply, evidence): evidence holds the facts and tool results the model was given
    (trusted) and, separately, what the caller said, so each stated fact can be checked.
    """
    facts = gates.visible_facts(s)
    answering = plan.tools_enabled and s.verified_party_id is not None and s.selected_case_id
    if answering:
        facts.update(prefetch_guidance(s, log=correction is None))
    system = prompts.responder_system(prompts.prompt_phase(s.phase, plan), plan, facts,
                                      verified=s.verified_party_id is not None)
    if correction:
        system += (f"\n\nCORRECTION: your previous draft was rejected because it {correction}. "
                   "Write the reply again, following the DIRECTIVES, VERIFICATION STATUS and FACTS exactly.")
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in s.history[-24:]]
    tools = TOOLS if answering else None
    evidence = Evidence(caller=gates.caller_text(s), primary_case_id=s.selected_case_id)
    evidence.add(facts)

    def done(text: str | None) -> tuple[str, Evidence]:
        return (text or "").strip() or prompts.fallback_reply(plan), evidence

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            msg = llm.chat(settings.model_responder, messages, tools=tools)
            if not getattr(msg, "tool_calls", None):
                return done(msg.content)
            messages.append(_assistant_tool_message(msg))
            for tc in msg.tool_calls:
                args, error = parse_tool_args(tc.function.name, tc.function.arguments)
                result = {"error": error} if error else run_tool(s, tc.function.name, args)
                evidence.add(result)
                s.log("tool_call", tool=tc.function.name, args=args, ok="error" not in result,
                      **({"error": error} if error else {}))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
        msg = llm.chat(settings.model_responder, messages, tools=None)
        return done(msg.content)
    except LLMAuthError:
        raise
    except LLMError as e:
        s.log("responder_error", error=str(e)[:300])
        return done(None)


TOPIC_PHRASES = {
    "status_inquiry": "the current status of the claim",
    "denial_question": "why the claim was denied and how to respond",
    "document_submission": "which documents to send and how",
    "next_steps": "what to do next and how long it takes",
    "general_claim_question": "your questions about the claim",
}


def write_email_summary(llm: LLM, s: Session, turn_evidence: Evidence | None = None) -> tuple[str, str]:
    """Write the summary email, check it, and fall back to a structured template if needed.

    Runs after this turn's reply, so the summary covers the last answer too. The model gets
    the facts of every claim discussed and the conversation from the verification turn on,
    with identity details masked. The draft is rejected if it contains identity details or
    a fact the claim records and guidance don't support (caller statements don't count).
    """
    holder = data.policyholder_by_party(s.verified_party_id)
    start = s.verified_at_history or 0
    transcript = "\n".join(f"{m['role'].upper()}: {gates.redact_identity(m['content'], holder)}"
                           for m in s.history[start:])
    email_facts = {
        "caller": gates.visible_facts(s).get("caller"),
        "today": settings.today.isoformat(),
        "topics_discussed": s.topics_discussed,
        "claims_discussed": [
            {"claim": claims.claim_facts(c, settings.today),
             "document_guidance": guidance.get_document_guidance(c) if c.get("documents_needed") else None}
            for c in _discussed_claims(s)
        ],
    }
    user = ("FACTS (json):\n" + json.dumps(email_facts, ensure_ascii=False)
            + "\n\nTRANSCRIPT:\n" + transcript + "\n\nReturn the JSON.")
    try:
        out = llm.complete_json(settings.model_responder, prompts.EMAIL_SYSTEM, user)
        subject, body = out.get("subject"), out.get("body")
        if isinstance(subject, str) and isinstance(body, str) and subject.strip() and body.strip():
            text = subject + "\n" + body
            evidence = grounding.for_session(s, include_guidance=True)
            if turn_evidence:
                evidence.merge(turn_evidence)
            problems = gates.email_pii_violations(text, holder) + grounding.violations(text, evidence)
            if not problems:
                return subject.strip(), body.strip()
            s.log("email_check_failed", problems=problems, rejected_reply=text[:600])
    except LLMAuthError:
        raise
    except LLMError as e:
        s.log("email_summary_error", error=str(e)[:300])
    return template_email(s)


def _discussed_claims(s: Session) -> list[dict]:
    ids = dict.fromkeys([*s.cases_discussed, *([s.selected_case_id] if s.selected_case_id else [])])
    return [c for cid in ids if (c := claims.get_claim(s.verified_party_id, cid))]


def _long_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def template_email(s: Session) -> tuple[str, str]:
    """A complete summary built only from structured data: what was discussed, outcome, next steps."""
    holder = data.policyholder_by_party(s.verified_party_id)
    first = holder["name"].split()[0]
    discussed = _discussed_claims(s)
    topics = [TOPIC_PHRASES[t] for t in s.topics_discussed if t in TOPIC_PHRASES]

    what, status, steps = [], [], []
    for c in discussed:
        what.append(f"- Claim {c['case_id']} ({c['case_type']}, filed {_long_date(c['created_at'])})"
                    + (f": {', '.join(topics)}." if topics else "."))
        if c.get("denial_reason"):
            status.append(f"- {c['case_id']} is {c['status']}. Reason for the denial: {c['denial_reason']}.")
        else:
            status.append(f"- {c['case_id']} is {c['status']}. {c['summary']}.")
        if c.get("appeal_deadline"):
            days = (date.fromisoformat(c["appeal_deadline"]) - settings.today).days
            status.append(f"- Appeal deadline: {_long_date(c['appeal_deadline'])}"
                          + (f", which is {days} days away." if days >= 0 else
                             ". This deadline has passed; a claims representative can review your options."))
        if c.get("documents_needed"):
            g = guidance.get_document_guidance(c)
            for doc in g["documents"]:
                steps.append(f"- Send the {doc['document']}. {doc['requirements'] or g['general_submission_guidance']}")
                steps.append(f"  If it is not available: {doc['if_unavailable']}")
            steps.append(f"- How to send: {g['general_submission_guidance']}")
            steps.append(f"- After the documents are received, review is {g['average_processing_time_after_submission']}.")
            if c.get("appeal_deadline") and (date.fromisoformat(c["appeal_deadline"]) - settings.today).days >= 0:
                steps.append(f"- Send everything before {_long_date(c['appeal_deadline'])}.")
        else:
            steps.append(f"- {c['case_id']}: no documents are currently requested.")
    if not discussed:
        what.append("- We verified your identity; no specific claim was reviewed.")
    steps.append("- If you have questions, reply to this email or contact claims support.")

    subject = f"Summary of your {prompts.COMPANY} claims conversation" + (
        f" - {', '.join(c['case_id'] for c in discussed)}" if discussed else "")
    body = "\n".join([
        f"Hello {first},", "",
        f"Thank you for contacting {prompts.COMPANY} claims support. Here is a summary of our conversation.", "",
        "What we discussed", *what, "",
        "Claim status", *(status or ["- No claim status was reviewed."]), "",
        "Next steps", *steps, "",
        f"{prompts.AGENT_NAME}, {prompts.COMPANY} Claims Support",
    ])
    return subject, body

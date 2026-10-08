"""Turns the controller's plan into a natural reply. Tools are only exposed in PROCESS_CASE."""
from __future__ import annotations

import json

from app.config import settings
from app.llm import prompts
from app.llm.client import LLM, LLMAuthError, LLMError
from app.memory import Phase, Session, TurnPlan
from app.sop import gates
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


def run_tool(s: Session, name: str, args: dict) -> dict:
    """Execute a tool call, scoped to the verified party. Unknown or foreign claims return an error."""
    party = s.verified_party_id
    if party is None:
        return {"error": "caller not verified"}
    if name == "list_claims":
        return {"claims": [claims.claim_summary(c) for c in claims.list_claims(party)]}
    claim = claims.get_claim(party, args.get("case_id", "") or s.selected_case_id or "")
    if claim is None:
        return {"error": "no such claim for this caller"}
    if name == "get_claim":
        return claims.claim_facts(claim, settings.today)
    if name == "get_document_guidance":
        return guidance.get_document_guidance(claim)
    if name == "get_followup_guidance":
        return guidance.get_followup_guidance(claim, args.get("question", ""), args.get("topic"))
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


def respond(llm: LLM, s: Session, plan: TurnPlan, correction: str | None = None) -> str:
    """Write the reply. correction is set when a previous draft contradicted the SOP state."""
    facts = gates.visible_facts(s)
    if plan.tools_enabled and s.phase == Phase.PROCESS_CASE and s.selected_case_id:
        facts.update(prefetch_guidance(s, log=correction is None))
    system = prompts.responder_system(s.phase, plan, facts, verified=s.verified_party_id is not None)
    if correction:
        system += (f"\n\nCORRECTION: your previous draft was rejected because the {correction}. "
                   "Write the reply again, following the DIRECTIVES and VERIFICATION STATUS exactly.")
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in s.history[-24:]]
    tools = TOOLS if plan.tools_enabled and s.phase == Phase.PROCESS_CASE else None

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            msg = llm.chat(settings.model_responder, messages, tools=tools)
            if not getattr(msg, "tool_calls", None):
                return (msg.content or "").strip() or prompts.fallback_reply(plan)
            messages.append(_assistant_tool_message(msg))
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(s, tc.function.name, args)
                s.log("tool_call", tool=tc.function.name, args=args, ok="error" not in result)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
        msg = llm.chat(settings.model_responder, messages, tools=None)
        return (msg.content or "").strip() or prompts.fallback_reply(plan)
    except LLMAuthError:
        raise
    except LLMError as e:
        s.log("responder_error", error=str(e)[:300])
        return prompts.fallback_reply(plan)


def write_email_summary(llm: LLM, s: Session) -> tuple[str, str]:
    facts = gates.visible_facts(s)
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in s.history)
    user = ("FACTS (json):\n" + json.dumps({
        "caller": facts.get("caller"), "selected_claim": facts.get("selected_claim"),
        "topics_discussed": s.topics_discussed,
        "document_guidance": run_tool(s, "get_document_guidance", {"case_id": s.selected_case_id})
        if s.selected_case_id else None,
    }, ensure_ascii=False) + "\n\nTRANSCRIPT:\n" + transcript + "\n\nReturn the JSON.")
    try:
        out = llm.complete_json(settings.model_responder, prompts.EMAIL_SYSTEM, user)
        if out.get("subject") and out.get("body"):
            return out["subject"], out["body"]
    except LLMAuthError:
        raise
    except LLMError as e:
        s.log("email_summary_error", error=str(e)[:300])
    claim = facts.get("selected_claim") or {}
    subject = f"Summary of your claims support conversation{' - ' + claim['case_id'] if claim else ''}"
    body = (f"Hello {facts['caller']['policyholder_name']},\n\nThank you for contacting us today."
            + (f"\n\nClaim {claim['case_id']} ({claim['case_type']}) status: {claim['status']}."
               f" {claim.get('summary', '')}" if claim else "")
            + "\n\nIf you have questions, reply to this email or contact claims support.\n\nClaims Support")
    return subject, body

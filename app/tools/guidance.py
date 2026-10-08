"""Grounded document and follow-up guidance from required_document_guideline.json."""
from __future__ import annotations

import re

from app import data


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def resolve_document_key(name: str, section: str = "document_guidance") -> str | None:
    """Map a claim's document name to a guideline key.

    claims.json says "pathology report" while the guideline key is
    "original pathology report". A key matches when it contains every word of
    the claim's document name.
    """
    keys = [k for k in data.document_guideline().get(section, {}) if k != "default"]
    if name in keys:
        return name
    wanted = _tokens(name)
    hits = [k for k in keys if wanted and wanted <= _tokens(k)]
    return min(hits, key=len) if hits else None


def _en(entry: dict | None) -> str | None:
    return entry.get("en") if entry else None


def join_documents(docs: list[str]) -> str:
    docs = [f"the {d}" for d in docs]
    if len(docs) <= 1:
        return "".join(docs)
    return ", ".join(docs[:-1]) + " and " + docs[-1]


def get_document_guidance(claim: dict) -> dict:
    g = data.document_guideline()
    docs = []
    for name in claim.get("documents_needed", []):
        key = resolve_document_key(name)
        alt_key = resolve_document_key(name, "document_alternative_guidance")
        docs.append({
            "document": name,
            "requirements": _en(g["document_guidance"].get(key)) if key else None,
            "if_unavailable": _en(g["document_alternative_guidance"].get(alt_key))
            if alt_key else _en(g["document_alternative_guidance"]["default"]),
            "specific_guidance_available": key is not None,
        })
    return {
        "case_id": claim["case_id"],
        "general_submission_guidance": _en(g["default_guidance"]),
        "case_type_guidance": _en(g["case_type_guidance"].get(claim["case_type"])),
        "documents": docs,
        "average_processing_time_after_submission": _en(
            g["claim_followup_settings"]["average_processing_time_after_submission"]),
        "if_all_alternatives_exhausted": _en(
            g["claim_followup_settings"]["human_review_after_document_alternatives_exhausted"]),
    }


def followup_topics() -> list[str]:
    return [t["topic"] for t in data.document_guideline()["claim_followup_guidance"]]


def _fill(template: str, claim: dict) -> str:
    g = data.document_guideline()
    return template.format(
        case_id=claim["case_id"],
        documents=join_documents(claim.get("documents_needed", [])) or "the requested documents",
        average_processing_time_after_submission=_en(
            g["claim_followup_settings"]["average_processing_time_after_submission"]),
    )


def get_followup_guidance(claim: dict, question: str = "", topic: str | None = None) -> dict:
    """Return the follow-up rules that apply to a question about this claim.

    The model may pass an explicit topic; otherwise the rules' match_any
    phrases are matched against the caller's question.
    """
    g = data.document_guideline()
    has_docs = bool(claim.get("documents_needed"))
    rules = [r for r in g["claim_followup_guidance"] if has_docs or not r.get("requires_documents")]
    q = question.lower()
    if topic:
        hits = [r for r in rules if r["topic"] == topic]
    else:
        hits = [r for r in rules if any(p in q for p in r.get("match_any", []))]
    if not hits:
        return {
            "case_id": claim["case_id"],
            "matched_topics": [],
            "fallback": _en(g["claim_followup_fallback"]),
            "available_topics": [r["topic"] for r in rules],
        }
    return {
        "case_id": claim["case_id"],
        "matched_topics": [{"topic": r["topic"], "guidance": _fill(r["en"], claim)} for r in hits],
    }

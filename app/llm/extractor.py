"""Per-turn extraction: an LLM call in JSON mode plus a small regex safety net."""
from __future__ import annotations

import re

from pydantic import ValidationError

from app.config import settings
from app.llm import prompts
from app.llm.client import LLM, LLMAuthError, LLMError
from app.memory import Extraction, Phase, Session
from app.tools import claims

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
POLICY_RE = re.compile(r"\bPOL-?\d{3,}\b", re.I)
CASE_RE = re.compile(r"\bCL-?\d{3,}\b", re.I)
ISO_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


def _context(s: Session) -> dict:
    ctx: dict = {"phase": s.phase.value, "caller_verified": s.verified_party_id is not None}
    if s.verified_party_id and s.candidate_case_ids:
        ctx["candidate_claims"] = [
            claims.claim_summary(claims.get_claim(s.verified_party_id, cid)) for cid in s.candidate_case_ids
        ]
    if s.selected_case_id:
        ctx["current_claim"] = s.selected_case_id
    ctx["email_summary_offered"] = s.phase == Phase.POST_PROCESS
    ctx["human_transfer_offered"] = s.human_offered
    return ctx


def regex_fill(ex: Extraction, text: str) -> Extraction:
    """Fill obvious structured values the model may have missed. Never overrides the model."""
    ident = ex.identity
    if not ident.email and (m := EMAIL_RE.search(text)):
        ident.email = m.group(0)
    if not ident.policy_number and (m := POLICY_RE.search(text)):
        ident.policy_number = m.group(0).upper()
    if not ex.case_hints.case_id and (m := CASE_RE.search(text)):
        ex.case_hints.case_id = claims.normalize_case_id(m.group(0))
    if not ident.dob and re.search(r"\b(dob|birth|born)\b", text, re.I) and (m := ISO_DATE_RE.search(text)):
        ident.dob = m.group(0)
    return ex


def extract(llm: LLM, s: Session, text: str) -> Extraction:
    transcript = [m for m in s.history if m["role"] in ("user", "assistant")][-6:]
    user = prompts.extractor_user_prompt(_context(s), transcript, text)
    try:
        raw = llm.complete_json(settings.model_extractor, prompts.EXTRACTOR_SYSTEM, user)
        ex = Extraction.model_validate(raw)
    except LLMAuthError:
        raise
    except (LLMError, ValidationError) as e:
        s.log("extraction_error", error=str(e)[:300])
        ex = Extraction()
    ex = regex_fill(ex, text)
    s.log("extraction", **ex.model_dump(exclude_defaults=True))
    return ex

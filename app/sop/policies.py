"""Emotion-aware tone and escalation policies."""
from __future__ import annotations

from app.config import Settings
from app.memory import Extraction, Session

# Emotions that, without new identity info, count as pushback against a gate.
PUSHBACK_EMOTIONS = {"frustration", "anger", "refusal"}

TONE = {
    "none": "Warm, professional and concise.",
    "frustration": (
        "The caller is frustrated. Open by acknowledging it in one specific, sincere sentence "
        "(not a generic 'I understand'). Then be brief and concrete about the fastest way forward."
    ),
    "anger": (
        "The caller is angry. Stay calm and non-defensive. Apologize for the experience, not for the "
        "rules. Keep sentences short, avoid jargon, and give one clear next step."
    ),
    "anxiety": (
        "The caller is anxious or worried. Reassure them that there is a path forward, explain what "
        "happens next in simple steps, and avoid alarming wording."
    ),
    "confusion": (
        "The caller is confused. Explain plainly, one step at a time, and check that it makes sense."
    ),
    "refusal": (
        "The caller is reluctant to do what is asked. Respect that, explain briefly why it matters, "
        "and offer acceptable alternatives without pressure."
    ),
}


def tone_for(emotion: str, intensity: str) -> str:
    tone = TONE.get(emotion, TONE["none"])
    if emotion != "none" and intensity == "high":
        tone += " The emotion is strong: de-escalate first, before any workflow request."
    return tone


def is_pushback(ex: Extraction) -> bool:
    return ex.refuses_verification or ex.emotion in PUSHBACK_EMOTIONS


def persuasion_exhausted(session: Session, ex: Extraction, cfg: Settings) -> bool:
    """Stop persuading after repeated refusals, or sooner if anger is strong."""
    if session.gate_refusals >= cfg.persuasion_budget:
        return True
    return ex.emotion == "anger" and ex.emotion_intensity == "high" and session.gate_refusals >= 2

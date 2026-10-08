"""Runtime configuration, read from environment variables (and .env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _today() -> date:
    raw = os.getenv("SOP_TODAY", "").strip()
    if raw:
        return date.fromisoformat(raw)
    return date.today()


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    model_extractor: str = field(default_factory=lambda: os.getenv("LLM_MODEL_EXTRACTOR", "deepseek-chat"))
    model_responder: str = field(default_factory=lambda: os.getenv("LLM_MODEL_RESPONDER", "deepseek-chat"))
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 60))

    verify_min_fields: int = field(default_factory=lambda: _int("VERIFY_MIN_FIELDS", 3))
    max_verify_failures: int = field(default_factory=lambda: _int("MAX_VERIFY_FAILURES", 3))
    max_off_topic: int = field(default_factory=lambda: _int("MAX_OFF_TOPIC", 3))
    persuasion_budget: int = field(default_factory=lambda: _int("PERSUASION_BUDGET", 3))

    consent_scenario: str = field(default_factory=lambda: os.getenv("CONSENT_SCENARIO", "default"))
    fixtures_dir: Path = field(
        default_factory=lambda: Path(os.getenv("FIXTURES_DIR", ROOT_DIR / "insurance_claims" / "fixtures"))
    )

    @property
    def today(self) -> date:
        return _today()


settings = Settings()

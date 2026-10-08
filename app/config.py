"""Runtime configuration, read from environment variables (and .env if present).

Invalid values fail at startup with a clear message instead of surfacing mid-conversation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# "Today" for deadline checks unless SOP_TODAY says otherwise. Fixed so every way of
# starting the demo gives the same answers; CL-2048's appeal window (2026-03-18) is open.
DEMO_TODAY = date(2026, 3, 1)
REAL_DATE_VALUES = ("real", "today", "system")


class ConfigError(ValueError):
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None


def _bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL") or "https://api.deepseek.com")
    model_extractor: str = field(default_factory=lambda: _env("LLM_MODEL_EXTRACTOR") or "deepseek-chat")
    model_responder: str = field(default_factory=lambda: _env("LLM_MODEL_RESPONDER") or "deepseek-chat")
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 60))

    # SOP_TODAY: empty or "demo" -> DEMO_TODAY, "real" -> the system date, YYYY-MM-DD -> that date
    sop_today: str = field(default_factory=lambda: _env("SOP_TODAY"))

    verify_min_fields: int = field(default_factory=lambda: _int("VERIFY_MIN_FIELDS", 3))
    max_verify_failures: int = field(default_factory=lambda: _int("MAX_VERIFY_FAILURES", 3))
    max_off_topic: int = field(default_factory=lambda: _int("MAX_OFF_TOPIC", 3))
    persuasion_budget: int = field(default_factory=lambda: _int("PERSUASION_BUDGET", 3))

    consent_scenario: str = field(default_factory=lambda: _env("CONSENT_SCENARIO") or "default")
    fixtures_dir: Path = field(
        default_factory=lambda: Path(_env("FIXTURES_DIR") or ROOT_DIR / "insurance_claims" / "fixtures")
    )

    # Inspector: per-field identity match results are hidden from API responses unless enabled
    debug_inspector: bool = field(default_factory=lambda: _bool("DEBUG_INSPECTOR"))

    # Public deployment limits
    access_token: str = field(default_factory=lambda: _env("ACCESS_TOKEN"))
    rate_limit_per_minute: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MINUTE", 30))
    session_ttl_minutes: int = field(default_factory=lambda: _int("SESSION_TTL_MINUTES", 30))
    max_sessions: int = field(default_factory=lambda: _int("MAX_SESSIONS", 200))
    max_message_chars: int = field(default_factory=lambda: _int("MAX_MESSAGE_CHARS", 2000))
    max_turns: int = field(default_factory=lambda: _int("MAX_TURNS", 60))

    def __post_init__(self) -> None:
        errors = []
        # The requirement is "at least 3 PII fields"; 5 is every counted field.
        if not 3 <= self.verify_min_fields <= 5:
            errors.append(f"VERIFY_MIN_FIELDS must be between 3 and 5, got {self.verify_min_fields}")
        for name in ("llm_timeout", "max_verify_failures", "max_off_topic", "persuasion_budget",
                     "rate_limit_per_minute", "session_ttl_minutes", "max_sessions",
                     "max_message_chars", "max_turns"):
            if getattr(self, name) < 1:
                errors.append(f"{name.upper()} must be at least 1, got {getattr(self, name)}")
        if self.consent_scenario not in ("default", "timeout"):
            errors.append(f"CONSENT_SCENARIO must be default or timeout, got {self.consent_scenario!r}")
        try:
            self._fixed_today = self._parse_today(self.sop_today)
        except ValueError:
            errors.append(f"SOP_TODAY must be empty, 'demo', 'real' or a YYYY-MM-DD date, got {self.sop_today!r}")
        if errors:
            raise ConfigError("Invalid configuration: " + "; ".join(errors))

    @staticmethod
    def _parse_today(raw: str) -> date | None:
        low = raw.lower()
        if low in ("", "demo"):
            return DEMO_TODAY
        if low in REAL_DATE_VALUES:
            return None
        return date.fromisoformat(raw)

    @property
    def today(self) -> date:
        return self._fixed_today or date.today()

    @property
    def today_mode(self) -> str:
        """demo | real | fixed, shown in the UI so a deadline answer can be read in context."""
        if self._fixed_today is None:
            return "real"
        return "demo" if self._fixed_today == DEMO_TODAY and self.sop_today.lower() in ("", "demo") else "fixed"


settings = Settings()

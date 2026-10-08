"""Read-only access to the JSON fixtures."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings


def _load(name: str, fixtures_dir: Path | None = None) -> Any:
    path = (fixtures_dir or settings.fixtures_dir) / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def policyholders() -> list[dict]:
    return _load("policyholders.json")


@lru_cache(maxsize=1)
def claims() -> list[dict]:
    return _load("claims.json")


@lru_cache(maxsize=1)
def representatives() -> list[dict]:
    return _load("representatives.json")


@lru_cache(maxsize=1)
def consent_scenarios() -> dict:
    return _load("consent_scenarios.json")


@lru_cache(maxsize=1)
def document_guideline() -> dict:
    return _load("required_document_guideline.json")


@lru_cache(maxsize=1)
def claim_schema() -> dict:
    return _load("claim_schema.json")


def policyholder_by_party(party_id: str) -> dict | None:
    return next((p for p in policyholders() if p["party_id"] == party_id), None)

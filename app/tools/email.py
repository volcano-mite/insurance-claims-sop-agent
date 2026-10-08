"""Email summary delivery. Default mode writes to an in-app outbox."""
from __future__ import annotations

from datetime import datetime, timezone


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if len(local) <= 2:
        return f"{local[:1]}***@{domain}"
    return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"


def send_summary(outbox: list[dict], to: str, subject: str, body: str) -> dict:
    record = {
        "to": to,
        "subject": subject,
        "body": body,
        "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "transport": "outbox",
    }
    outbox.append(record)
    return record

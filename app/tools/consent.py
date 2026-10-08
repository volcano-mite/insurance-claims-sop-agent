"""Simulated policyholder consent for representative callers (consent_scenarios.json)."""
from __future__ import annotations

from dataclasses import dataclass, field

from app import data


@dataclass
class ConsentRequest:
    party_id: str
    rep_name: str
    scenario: str
    sequence: list[str] = field(default_factory=list)
    polls: int = 0
    status: str = "pending"  # pending | approved | timeout

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "polls": self.polls, "status": self.status,
                "rep_name": self.rep_name, "max_polls": len(self.sequence)}


def request_consent(party_id: str, rep_name: str, scenario: str = "default") -> ConsentRequest:
    scenarios = data.consent_scenarios()
    seq = scenarios.get(scenario, scenarios["default"])["status_sequence"]
    return ConsentRequest(party_id=party_id, rep_name=rep_name, scenario=scenario, sequence=list(seq))


def poll(req: ConsentRequest) -> str:
    """Advance the simulated consent status by one poll."""
    if req.status != "pending":
        return req.status
    status = req.sequence[req.polls] if req.polls < len(req.sequence) else "pending"
    req.polls += 1
    if status == "approved":
        req.status = "approved"
    elif req.polls >= len(req.sequence):
        req.status = "timeout"
    return req.status

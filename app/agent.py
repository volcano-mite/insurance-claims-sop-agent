"""One conversational turn: extract -> remember -> SOP controller -> respond -> guard."""
from __future__ import annotations

import copy
from typing import Callable

from app import data
from app.config import settings
from app.llm import prompts
from app.llm.client import LLM, LLMAuthError, make_llm
from app.llm.extractor import extract
from app.llm.responder import respond, write_email_summary
from app.memory import Phase, Session
from app.sop import gates
from app.sop.state_machine import Controller
from app.tools.email import send_summary


class Agent:
    def __init__(self, llm_factory: Callable[[str | None], LLM] = make_llm, controller: Controller | None = None):
        self.llm_factory = llm_factory
        self.controller = controller or Controller(settings)

    def new_session(self, api_key: str | None = None, consent_scenario: str | None = None) -> Session:
        s = Session(api_key=api_key or None, consent_scenario=consent_scenario or settings.consent_scenario)
        s.history.append({"role": "assistant", "content": prompts.GREETING})
        return s

    def handle(self, s: Session, text: str) -> str:
        text = text.strip()
        if s.phase in (Phase.ESCALATED, Phase.ENDED):
            reply = self._closed_reply(s)
            s.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            return reply

        before = copy.deepcopy(s.__dict__)
        try:
            return self._turn(s, text)
        except LLMAuthError:
            s.__dict__.update(before)  # a rejected key must not leave a half-applied turn
            raise

    def _turn(self, s: Session, text: str) -> str:
        s.turn += 1
        llm = self.llm_factory(s.api_key)
        ex = extract(llm, s, text)
        s.history.append({"role": "user", "content": text})
        plan = self.controller.step(s, ex)

        if "SEND_EMAIL" in plan.actions:
            holder = data.policyholder_by_party(s.verified_party_id)
            subject, body = write_email_summary(llm, s)
            send_summary(s.outbox, holder["email"], subject, body)
            s.log("send_email", to=holder["email"], subject=subject)

        reply = self._checked_reply(llm, s, plan)
        s.history.append({"role": "assistant", "content": reply})
        return reply

    @staticmethod
    def _checked_reply(llm: LLM, s: Session, plan) -> str:
        """Generate the reply and apply the output guards.

        A data leak falls back to a fixed reply immediately. A reply that contradicts the
        SOP state (e.g. "details don't match" after verification) is regenerated once with a
        correction note, and falls back to a fixed reply if it is still inconsistent.
        """
        correction = None
        for attempt in range(2):
            reply = respond(llm, s, plan, correction=correction)
            _, leak = gates.guard_reply(s, reply)
            if leak:
                s.log("guard_blocked", reason=leak, blocked_reply=reply[:300])
                return prompts.fallback_reply(plan)
            correction = gates.state_consistency(s, reply)
            if not correction:
                return reply
            s.log("state_mismatch", reason=correction, attempt=attempt + 1, rejected_reply=reply[:300])
        return prompts.fallback_reply(plan)

    @staticmethod
    def _closed_reply(s: Session) -> str:
        if s.phase == Phase.ESCALATED:
            return ("You're being connected to a human representative. Your reference number is "
                    f"{s.escalation['reference']}. Please hold on, or start a new chat if you'd like.")
        return "This conversation has ended. Please start a new chat if there's anything else I can help with."

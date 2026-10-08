"""One conversational turn: extract -> remember -> SOP controller -> respond -> guard."""
from __future__ import annotations

import copy
import logging
from typing import Callable

from app import data
from app.config import settings
from app.llm import prompts
from app.llm.client import LLM, LLMError, make_llm
from app.llm.extractor import extract
from app.llm.responder import respond, write_email_summary
from app.memory import Phase, Session
from app.sop import gates, grounding
from app.sop.grounding import Evidence
from app.sop.state_machine import Controller
from app.tools.email import send_summary

log = logging.getLogger(__name__)


class TurnError(RuntimeError):
    """An unexpected failure inside a turn. The session was rolled back to before the turn."""


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

        # Whatever fails, the session must not keep a half-applied turn (e.g. phase advanced
        # but no reply). The caller can simply send the message again.
        before = copy.deepcopy(s.__dict__)
        try:
            return self._turn(s, text)
        except LLMError:  # includes a rejected API key; reported to the caller as such
            s.__dict__.update(before)
            raise
        except Exception as e:
            s.__dict__.update(before)
            s.log("turn_error", error=type(e).__name__)
            log.exception("turn failed in session %s", s.id)
            raise TurnError("Something went wrong on our side. Your message was not applied; please try again.") from e

    def _turn(self, s: Session, text: str) -> str:
        s.turn += 1
        llm = self.llm_factory(s.api_key)
        ex = extract(llm, s, text)
        s.history.append({"role": "user", "content": text})
        plan = self.controller.step(s, ex)

        reply, evidence = self._checked_reply(llm, s, plan)
        s.history.append({"role": "assistant", "content": reply})

        # After the reply, so "tell me why it was denied, then email me the summary" gets a
        # summary that includes that answer and the tool results behind it. Sending cannot
        # fail short of a rejected key (which rolls the turn back), so the reply's
        # "I've sent it" stays true.
        if "SEND_EMAIL" in plan.actions:
            holder = data.policyholder_by_party(s.verified_party_id)
            subject, body = write_email_summary(llm, s, turn_evidence=evidence)
            send_summary(s.outbox, holder["email"], subject, body)
            s.log("send_email", to=holder["email"], subject=subject)
        return reply

    @staticmethod
    def _checked_reply(llm: LLM, s: Session, plan) -> tuple[str, Evidence | None]:
        """Generate the reply and apply the output checks, in order:

        1. disclosure guard: no claim information before verification, no other party's claims after;
        2. grounding: after verification, each stated amount, date, time span, status, denial reason
           and promise must match the trusted claim data and tool results (not the caller's words);
        3. state consistency: e.g. no "details don't match" after verification succeeded;
        4. required directives: e.g. the email offer or the transfer reference number is not left out.

        A rejected draft never reaches the caller. It is regenerated once with a correction
        note, and if the second draft also fails, a fixed reply built from the plan is used.
        """
        correction = None
        for attempt in (1, 2):
            reply, evidence = respond(llm, s, plan, correction=correction)
            problem = Agent._first_problem(s, plan, reply, evidence, attempt)
            if problem is None:
                return reply, evidence
            kind, correction = problem
        # Grounding is checked per sentence: rather than discard a mostly correct answer,
        # drop the unsupported sentences and use the rest if it now passes every check.
        if kind == "ungrounded":
            pruned, removed = grounding.prune(reply, evidence)
            if len(pruned) >= 40 and Agent._first_problem(s, plan, pruned, evidence, None) is None:
                s.log("pruned_reply", sentences_removed=removed)
                return pruned, evidence
        return prompts.fallback_reply(plan), None

    @staticmethod
    def _first_problem(s: Session, plan, reply: str, evidence: Evidence, attempt: int | None):
        """(kind, correction note) for the first check the reply fails, or None. Logs when attempt is set."""
        def fail(event: str, kind: str, note: str, **data):
            if attempt is not None:
                s.log(event, attempt=attempt, rejected_reply=reply[:300], **data)
            return kind, note

        if leak := gates.guard_reply(s, reply):
            return fail("guard_blocked", "leak", f"{leak.removeprefix('unverified reply ').removeprefix('reply ')}, "
                        "which is not allowed here", reason=leak)
        if s.verified_party_id and (ungrounded := grounding.violations(reply, evidence)):
            return fail("ungrounded_reply", "ungrounded",
                        "stated things that FACTS and tool results do not support: " + "; ".join(ungrounded)
                        + ". Use each value only for what its field means", problems=ungrounded)
        if mismatch := gates.state_consistency(s, reply):
            return fail("state_mismatch", "state", f"contradicted the SOP state: the {mismatch}", reason=mismatch)
        if missing := gates.missing_directives(plan, reply):
            return fail("directive_missing", "directive", missing, reason=missing)
        return None

    @staticmethod
    def _closed_reply(s: Session) -> str:
        if s.phase == Phase.ESCALATED:
            return ("You're being connected to a human representative. Your reference number is "
                    f"{s.escalation['reference']}. Please hold on, or start a new chat if you'd like.")
        return "This conversation has ended. Please start a new chat if there's anything else I can help with."

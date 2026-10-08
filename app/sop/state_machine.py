"""The SOP controller: a deterministic state machine.

Each turn it receives the extractor's output, updates session memory, applies
the gates and decides the phase and the directives for the responder. The LLM
never changes the phase or passes a gate by itself.

    VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS -> ENDED
        any phase -> ESCALATED (human handoff)
"""
from __future__ import annotations

import random

from app import data
from app.config import Settings, settings
from app.memory import Extraction, Phase, Session, TurnPlan
from app.sop import policies
from app.tools import claims, consent, verification
from app.tools.email import mask_email
from app.tools.verification import COUNTED_FIELDS, FIELD_LABELS

MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")


def describe_hints(hints: dict, intent: str | None = None) -> str:
    """Human-readable version of what the caller said they are calling about."""
    parts = []
    if hints.get("status"):
        parts.append(hints["status"])
    if hints.get("case_type"):
        parts.append(hints["case_type"])
    parts.append("claim")
    text = " ".join(parts)
    if hints.get("case_id"):
        text += f" {hints['case_id']}"
    if hints.get("month"):
        text += f" from {MONTH_NAMES[int(hints['month']) - 1]}"
        if hints.get("year"):
            text += f" {hints['year']}"
    elif hints.get("year"):
        text += f" from {hints['year']}"
    if intent:
        text += f" ({intent.replace('_', ' ')})"
    return text


class Controller:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg

    # ------------------------------------------------------------ entry point

    def step(self, s: Session, ex: Extraction) -> TurnPlan:
        plan = TurnPlan(tone=policies.tone_for(ex.emotion, ex.emotion_intensity))
        self._remember(s, ex)

        if s.phase in (Phase.ESCALATED, Phase.ENDED):
            plan.add("CONVERSATION_CLOSED", phase=s.phase.value)
            return plan

        if ex.wants_human or (s.human_offered and ex.accepts_human_offer):
            self._escalate(s, plan, "caller_requested_human")
            return plan
        if s.human_offered and ex.accepts_human_offer is False:
            s.human_offered = False
            s.off_topic_streak = 0

        if not ex.in_scope:
            s.off_topic_streak += 1
            plan.add("DECLINE_OFF_TOPIC", topic=ex.off_topic_summary)
            if s.off_topic_streak >= self.cfg.max_off_topic:
                s.human_offered = True
                plan.add("OFFER_HUMAN", reason="repeated_off_topic")
        else:
            s.off_topic_streak = 0
            if ex.off_topic_summary:
                plan.add("DECLINE_OFF_TOPIC", topic=ex.off_topic_summary)

        handler = {
            Phase.VERIFY_ID: self._verify_id,
            Phase.RESOLVE_INTENT: self._resolve_intent,
            Phase.PROCESS_CASE: self._process_case,
            Phase.POST_PROCESS: self._post_process,
        }[s.phase]
        handler(s, ex, plan)
        s.log("plan", phase=s.phase.value, selected_case=s.selected_case_id,
              directives=[d["kind"] for d in plan.directives], actions=plan.actions)
        return plan

    # ------------------------------------------------------------ memory

    def _remember(self, s: Session, ex: Extraction) -> None:
        """Store anything useful the caller said, whatever the current phase."""
        hints = ex.case_hints.present()
        if hints.get("case_id"):
            hints["case_id"] = claims.normalize_case_id(hints["case_id"])
        if hints.get("case_type"):
            hints["case_type"] = claims.normalize_case_type(hints["case_type"]) or hints["case_type"]
        if hints.get("status"):
            hints["status"] = claims.normalize_status(hints["status"]) or hints["status"]
        if hints:
            s.case_hints.update(hints)
        if ex.intent:
            s.pending_intent = ex.intent
        if ex.caller_role:
            s.caller_role = ex.caller_role
        if ex.representative_name:
            s.rep_name = ex.representative_name
            s.caller_role = s.caller_role or "representative"
        if ex.relationship:
            s.relationship = ex.relationship
        for f in ex.declined_fields:
            if f not in s.declined_fields:
                s.declined_fields.append(f)
        if ex.emotion != "none":
            s.emotion_history.append({"turn": s.turn, "emotion": ex.emotion, "intensity": ex.emotion_intensity})

    # ------------------------------------------------------------ VERIFY_ID

    def _id_request(self, s: Session) -> dict:
        have = [k for k in COUNTED_FIELDS if s.identity.get(k)]
        options = [FIELD_LABELS[k] for k in COUNTED_FIELDS if k not in have and k not in s.declined_fields]
        return {
            "already_provided": [FIELD_LABELS[k] for k in have],
            "still_needed": max(0, self.cfg.verify_min_fields - len(have)),
            "options": options,
            "declined": [FIELD_LABELS[k] for k in s.declined_fields],
        }

    def _verify_id(self, s: Session, ex: Extraction, plan: TurnPlan) -> None:
        if s.consent and s.consent.status == "pending":
            self._await_consent(s, plan)
            return

        new = {k: v for k, v in ex.identity.model_dump().items() if v and s.identity.get(k) != v}
        s.identity.update(new)
        if new:
            result = verification.evaluate(s.identity, self.cfg.verify_min_fields)
            s.last_verification = {"matched": result.matched, "mismatched": result.mismatched,
                                   "verified": result.verified}
            s.log("verify_identity", fields=sorted(new), matched=len(result.matched),
                  mismatched=len(result.mismatched), verified=result.verified)
            if result.verified:
                self._on_identity_match(s, ex, plan, result.party_id)
                return
            if result.failed:
                s.verify_failures += 1
                # Clear everything so the response cannot reveal which field was wrong.
                s.identity = {}
                if s.verify_failures >= self.cfg.max_verify_failures:
                    self._escalate(s, plan, "verification_failed")
                    return
                plan.add("VERIFICATION_FAILED",
                         attempts_left=self.cfg.max_verify_failures - s.verify_failures)
        elif policies.is_pushback(ex) and not plan.has("DECLINE_OFF_TOPIC"):
            s.gate_refusals += 1
            if policies.persuasion_exhausted(s, ex, self.cfg):
                self._escalate(s, plan, "verification_refused")
                return
            plan.add("EXPLAIN_WHY_VERIFICATION", refusals=s.gate_refusals)

        # Anything the caller wants help with is remembered (in _remember) and acknowledged,
        # but nothing about it is looked up until verification passes.
        if ex.intent or ex.case_hints.present():
            plan.add("ACK_REMEMBERED_REQUEST", request=describe_hints(s.case_hints, s.pending_intent))
        plan.add("ASK_FOR_IDENTITY", **self._id_request(s))

    def _on_identity_match(self, s: Session, ex: Extraction, plan: TurnPlan, party_id: str) -> None:
        if s.caller_role == "representative":
            rep = verification.is_authorized_representative(s.rep_name or "", party_id)
            if not rep:
                s.human_offered = True
                plan.add("REPRESENTATIVE_NOT_AUTHORIZED")
                plan.add("OFFER_HUMAN", reason="representative_not_on_file")
                return
            s.relationship = s.relationship or rep["relationship"]
            s.consent = consent.request_consent(party_id, rep["rep_name"], s.consent_scenario)
            s.log("request_consent", scenario=s.consent_scenario)
            self._await_consent(s, plan, first=True)
            return
        s.verified_party_id = party_id
        self._after_verified(s, ex, plan)

    def _await_consent(self, s: Session, plan: TurnPlan, first: bool = False) -> None:
        status = consent.poll(s.consent)
        s.log("poll_consent", status=status, polls=s.consent.polls)
        if status == "approved":
            s.verified_party_id = s.consent.party_id
            plan.add("CONSENT_APPROVED")
            self._after_verified(s, Extraction(), plan)
        elif status == "timeout":
            self._escalate(s, plan, "consent_timeout")
        else:
            plan.add("CONSENT_PENDING", first_request=first,
                     checks_left=len(s.consent.sequence) - s.consent.polls)

    def _after_verified(self, s: Session, ex: Extraction, plan: TurnPlan) -> None:
        holder = data.policyholder_by_party(s.verified_party_id)
        plan.add("VERIFIED", policyholder_first_name=holder["name"].split()[0],
                 caller_role=s.caller_role or "policyholder")
        s.gate_refusals = 0
        self._resolve_intent(s, ex, plan)

    # ------------------------------------------------------------ RESOLVE_INTENT

    def _resolve_intent(self, s: Session, ex: Extraction, plan: TurnPlan) -> None:
        s.phase = Phase.RESOLVE_INTENT
        party = s.verified_party_id
        on_file = claims.list_claims(party)
        if not on_file:
            if ex.wants_to_end:
                self._enter_post_process(s, ex, plan)
            else:
                plan.add("NO_CLAIMS_ON_FILE")
            return

        chosen = claims.get_claim(party, ex.selected_case_id) if ex.selected_case_id else None
        if chosen is None and s.case_hints:
            found, exact = claims.match_claims(party, s.case_hints)
            s.log("match_claims", hints=s.case_hints, candidates=[c["case_id"] for c in found], exact=exact)
            if len(found) == 1 and exact:
                chosen = found[0]
            elif found:
                s.candidate_case_ids = [c["case_id"] for c in found]
                plan.add("DISAMBIGUATE", request=describe_hints(s.case_hints),
                         candidates=[claims.claim_summary(c) for c in found], exact_match=exact)
                return
            else:
                plan.add("NO_MATCHING_CLAIM", request=describe_hints(s.case_hints))

        if chosen is None and len(on_file) == 1 and not plan.has("NO_MATCHING_CLAIM"):
            chosen = on_file[0]
        if chosen:
            s.selected_case_id = chosen["case_id"]
            s.candidate_case_ids = []
            plan.add("CASE_SELECTED", claim=claims.claim_summary(chosen),
                     matched_from=describe_hints(s.case_hints) if s.case_hints else None)
            self._process_case(s, ex, plan, entering=True)
            return
        if ex.wants_to_end:
            self._enter_post_process(s, ex, plan)
            return
        plan.add("ASK_WHICH_CLAIM_AND_NEED", has_remembered_intent=bool(s.pending_intent))

    # ------------------------------------------------------------ PROCESS_CASE

    def _process_case(self, s: Session, ex: Extraction, plan: TurnPlan, entering: bool = False) -> None:
        s.phase = Phase.PROCESS_CASE
        if not entering:
            new_id = claims.normalize_case_id(ex.case_hints.case_id or ex.selected_case_id)
            if ex.switch_case or (new_id and new_id != s.selected_case_id):
                s.log("switch_case", from_case=s.selected_case_id, to_case=new_id)
                s.selected_case_id = None
                s.case_hints = {k: v for k, v in s.case_hints.items() if k in ex.case_hints.present()}
                self._resolve_intent(s, ex, plan)
                return
            if ex.wants_to_end or ex.email_decision:
                self._enter_post_process(s, ex, plan)
                return
        intent = ex.intent or s.pending_intent or "general_claim_question"
        if intent not in s.topics_discussed:
            s.topics_discussed.append(intent)
        plan.tools_enabled = True
        plan.add("ANSWER_CASE_QUESTION", case_id=s.selected_case_id, intent=intent)

    # ------------------------------------------------------------ POST_PROCESS

    def _enter_post_process(self, s: Session, ex: Extraction, plan: TurnPlan) -> None:
        s.phase = Phase.POST_PROCESS
        if ex.email_decision:
            self._post_process(s, ex, plan)
            return
        plan.add("OFFER_EMAIL_SUMMARY", masked_email=self._masked_email(s))

    def _post_process(self, s: Session, ex: Extraction, plan: TurnPlan) -> None:
        if ex.email_decision == "send":
            s.email_decision = "send"
            plan.actions.append("SEND_EMAIL")
            plan.add("EMAIL_SENT", masked_email=self._masked_email(s))
            plan.add("GOODBYE")
            s.phase = Phase.ENDED
        elif ex.email_decision == "skip":
            s.email_decision = "skip"
            plan.add("EMAIL_SKIPPED")
            plan.add("GOODBYE")
            s.phase = Phase.ENDED
        elif s.selected_case_id and (ex.intent or ex.switch_case or ex.case_hints.present()) and not ex.wants_to_end:
            self._process_case(s, ex, plan)
        else:
            plan.add("OFFER_EMAIL_SUMMARY", masked_email=self._masked_email(s), repeat=True)

    def _masked_email(self, s: Session) -> str:
        return mask_email(data.policyholder_by_party(s.verified_party_id)["email"])

    # ------------------------------------------------------------ escalation

    def _escalate(self, s: Session, plan: TurnPlan, reason: str) -> None:
        ref = f"HX-{random.randint(10000, 99999)}"
        s.escalation = {"reason": reason, "reference": ref, "from_phase": s.phase.value}
        s.phase = Phase.ESCALATED
        s.human_offered = False
        s.log("escalate", reason=reason, reference=ref)
        plan.add("ESCALATE_TO_HUMAN", reason=reason, reference=ref,
                 caller_verified=s.verified_party_id is not None)

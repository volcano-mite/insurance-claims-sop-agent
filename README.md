# Insurance Claims SOP Agent

A conversational insurance-claims support agent that follows a fixed business workflow without sounding like a form:

```
VERIFY_ID  →  RESOLVE_INTENT  →  PROCESS_CASE  →  POST_PROCESS
 (strict)       (flexible)         (grounded)       (caller's choice)
```

The SOP is enforced in code by a deterministic state machine. The LLM (DeepSeek) is used to understand messy language, extract information and write natural replies. It never decides on its own whether a gate has been passed.

See [PLAN.md](PLAN.md) for the full design rationale.

---

## Quick start

### Option A: Docker

```bash
cp .env.example .env          # optional: put your DeepSeek key in .env
docker compose up --build
```

Open http://localhost:8000.

### Option B: Local Python

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...      # optional, see below
uvicorn app.server:app --port 8000
```

### Providing the API token

Use either of these:

- **Environment variable:** set `DEEPSEEK_API_KEY` in `.env` or in your shell.
- **In the UI:** paste the key into the *DeepSeek API key* field at the top of the page. The server keeps it in memory for that chat session only and never writes it to disk. The browser remembers it for the current tab only.

An invalid key is reported in the chat instead of failing silently, and the turn is not applied.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key. Can also be entered in the UI. |
| `LLM_BASE_URL` | `https://api.deepseek.com` | Any OpenAI-compatible endpoint works. |
| `LLM_MODEL_EXTRACTOR` | `deepseek-chat` | Model for per-turn structured extraction (JSON mode). |
| `LLM_MODEL_RESPONDER` | `deepseek-chat` | Model for replies, tool calls and the email summary. |
| `LLM_TIMEOUT` | `60` | Seconds per LLM request. |
| `SOP_TODAY` | `2026-03-01` in `.env.example` and Docker Compose; real current date if unset | "Today" used for deadline checks. Leave empty to use the real date. |
| `VERIFY_MIN_FIELDS` | `3` | Number of matching identity fields required to verify. |
| `MAX_VERIFY_FAILURES` | `3` | Failed verification attempts before human handoff. |
| `MAX_OFF_TOPIC` | `3` | Consecutive out-of-scope questions before offering a human. |
| `PERSUASION_BUDGET` | `3` | Turns of pushback on verification before handing off to a human. |
| `CONSENT_SCENARIO` | `default` | Default consent simulation for representative callers (`default` or `timeout`). Can be changed per chat in the UI. |

---

## Using the demo

The page has two panes:

- **Chat** (left): talk to the agent in natural language. The *Try a scenario* chips send sample messages (hover to see the full text). Small markers between messages show what the SOP did on that turn: gate held, identity verified, claim selected, tools looked up, guard blocked a reply, transferred to a human, email sent.
- **SOP Inspector** (right): the workflow stepper, the identity gate (which fields matched or did not), remembered memory slots, guardrail counters, the detected emotion, the email outbox, and a collapsible trace of every extraction, plan, tool call and guard decision. It is a testing aid; a real caller would not see it.

The *Consent* switch in the top bar chooses how the simulated policyholder responds when a representative calls (*Approves* or *Times out*); switching starts a new chat. The key button shows whether an API key is available and lets you enter one. Reloading the page keeps the current conversation. The page follows the system light or dark theme.

### Test personas

All data comes from [`insurance_claims/fixtures/`](insurance_claims/fixtures/).

| Name | Policy | DOB | ID last 4 | Phone | Email | Notes |
|---|---|---|---|---|---|---|
| Margaret Chen | POL-9921 | 1985-03-15 | 4472 (SSN) | +1 650 521 2836 | margaret@email.com | 4 claims, incl. denied CL-2048 |
| Ava Lopez | POL-1044 | 1990-08-21 | 9180 (SSN) | +1 650 388 2920 | ava.lopez@email.com | no claims |
| Ma Tian | POL-8836 | 1964-09-10 | 6688 (national ID) | +1 650 208 8799 | matian@example.com | denied CL-3001 |
| Ya Wen Li | POL-7742 | 1989-12-03 | 5317 (national ID) | +1 650 521 2830 | yawen.li@gmail.com | alias "Yaven Li", alt email yawen.li@example.com |
| David Chen | — | — | — | — | — | Margaret's son; representative flow, needs her consent |

### Scripted walkthrough

1. **Verify with a remembered intent**
   > I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

   The agent verifies Margaret and does not ask what the call is about. It uses the remembered hint to pick **CL-2048** instead of CL-2011, the other January healthcare claim, and explains the denial.

2. **Ask grounded follow-ups**
   > How soon do I need to submit the documents, and how long does review take?
   > I can't get the original pathology report. What can I do?

3. **Test the guardrails** (start a new chat)
   - *Before verifying:* "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
   - *Out of scope:* "What is RL?" Ask unrelated questions three times in a row to trigger the human-handoff offer.

4. **Wrap up.** Say "That's all", then accept or decline the email summary. If you accept, the email appears in the outbox panel.

---

## How it works

On each turn:

1. **Extractor (LLM, JSON mode):** pulls identity fields, intent and case hints, emotion, scope, and email or handoff decisions out of the message, whatever the current phase is. A small regex pass fills obvious values (emails, policy and claim numbers) the model may miss.
2. **Session memory:** stores everything it extracted, so details mentioned early (e.g. "my denied claim from January") are used later.
3. **SOP controller (code):** checks identity against the fixtures, moves between phases, counts failures, refusals and off-topic turns, and decides when to escalate. It outputs a list of directives for the reply.
4. **Responder (LLM):** turns the directives into a natural reply. It only sees the data the current phase allows. Before verification, no claim data is in its context at all.
5. **Output guards (code):**
   - *Leak guard:* blocks claim IDs, amounts or denial reasons before verification, and other callers' claims after it. A blocked reply is replaced by a safe fixed message.
   - *State-consistency check:* catches replies that contradict the SOP state, such as saying "details don't match" after verification succeeded, or "you're verified" before it. The reply is regenerated once with a correction note, then replaced by a fixed message if it is still wrong. The prompt also states the current verification status explicitly, so earlier "didn't match" messages in the conversation don't mislead the model.

| Phase | Decided in code | LLM freedom | LLM tools |
|---|---|---|---|
| VERIFY_ID | identity match (≥3 of name, DOB, phone, email, ID last 4), failure lockout, representative authorization and consent | Phrasing, empathy | none |
| RESOLVE_INTENT | claim matching from remembered hints | Interpreting which claim and what need; disambiguation | none |
| PROCESS_CASE | claim ownership checks on every tool call | Multi-step reasoning grounded in tool results | `get_claim`, `list_claims`, `get_document_guidance`, `get_followup_guidance` |
| POST_PROCESS | send only after the caller agrees, only to the email on file | Writing the summary | none |

Escalation to a human is available in every phase: on request, after repeated failed verification or refusal, after repeated off-topic questions (offered), or when a representative's consent times out.

### Identity rules

- Verification needs at least 3 matching fields and no mismatches. The policy number helps find the account but does not count.
- Names match aliases and reordered or joined spellings. Phones and dates of birth are matched across common formats.
- On a mismatch, the agent never says which field was wrong. All provided fields are cleared and the caller is asked again.
- A representative must be on file for the policyholder and the policyholder must approve a consent request before anything is shared.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/unit                       # deterministic, no API key needed
pytest tests/scenarios -m llm           # end-to-end conversations, needs DEEPSEEK_API_KEY
SCENARIO_RUNS=3 pytest tests/scenarios  # repeat each scenario to check stability
```

Unit tests cover verification, claim matching, guidance mapping, the state machine, the output guard and the HTTP API, using a scripted fake LLM where needed.

Scenario tests in [`tests/scenarios/`](tests/scenarios/) are YAML conversations. They check **state**, not exact wording: the current phase, whether data leaked, which case was selected, and whether the conversation escalated. Covered: the official demo, partial or refused ID fields, national-ID callers, name and email aliases, verification lockout, representative consent (approved and timeout), an angry caller, repeated off-topic questions, claim disambiguation, prompt injection, and accepting or declining the email.

---

## Project layout

```
app/
  sop/          state_machine.py (controller), gates.py (data visibility and output guard), policies.py (tone, escalation)
  llm/          client.py (DeepSeek), extractor.py, responder.py (replies, tool loop, email summary), prompts.py
  tools/        verification.py, claims.py, guidance.py, consent.py, email.py
  agent.py      one conversational turn
  memory.py     session memory, extraction schema, turn plan
  server.py     FastAPI app
web/            single-page chat UI and debug panel
insurance_claims/fixtures/   sample data
tests/          unit/ and scenarios/
```

## Known limitations

- All data is read from local JSON fixtures. There is no real claims system or authentication backend.
- Sessions live in server memory and are lost on restart.
- Emails go to an in-app outbox; nothing is sent over SMTP.
- The consent flow is simulated from `consent_scenarios.json`: each caller message after the request checks the status once.
- The demo date defaults to `SOP_TODAY=2026-03-01` so CL-2048's appeal window (deadline 2026-03-18) is still open. With the real date, the deadline has passed and the agent says so and suggests a human representative.

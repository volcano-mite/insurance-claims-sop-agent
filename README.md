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
| `SOP_TODAY` | empty (demo date `2026-03-01`) | "Today" used for deadline checks: empty or `demo` = 2026-03-01, `real` = the system date, or a fixed `YYYY-MM-DD`. The same default applies however the app is started, and the UI header shows the date in use. |
| `VERIFY_MIN_FIELDS` | `3` | Number of matching identity fields required to verify. Must be 3 to 5; the requirement is at least 3. |
| `MAX_VERIFY_FAILURES` | `3` | Failed verification attempts before human handoff. |
| `MAX_OFF_TOPIC` | `3` | Consecutive out-of-scope questions before offering a human. |
| `PERSUASION_BUDGET` | `3` | Turns of pushback on verification before handing off to a human. |
| `CONSENT_SCENARIO` | `default` | Default consent simulation for representative callers (`default` or `timeout`). Can be changed per chat in the UI. |
| `DEBUG_INSPECTOR` | `false` | Show which identity field matched or failed in the inspector. For local testing only. |
| `ACCESS_TOKEN` | — | If set, the UI asks for this code and every session and chat request must carry it. |
| `RATE_LIMIT_PER_MINUTE` | `30` | Session and chat requests per client IP per minute. |
| `SESSION_TTL_MINUTES` | `30` | Idle sessions are deleted after this long. |
| `MAX_SESSIONS` | `200` | Sessions kept in memory; the least recently used are dropped first. |
| `MAX_MESSAGE_CHARS` | `2000` | Longest accepted caller message. |
| `MAX_TURNS` | `60` | Messages per conversation before a new chat is required. |

Invalid values (for example `VERIFY_MIN_FIELDS=2` or a malformed `SOP_TODAY`) stop the server at startup with a message naming the setting.

---

## Using the demo

The page has two panes:

- **Chat** (left): talk to the agent in natural language. The *Try a scenario* chips send sample messages (hover to see the full text). Small markers between messages show what the SOP did on that turn: gate held, identity verified, claim selected, tools looked up, guard blocked a reply, transferred to a human, email sent.
- **SOP Inspector** (right): the workflow stepper, the identity gate (which fields matched or did not), remembered memory slots, guardrail counters, the detected emotion, the email outbox, and a collapsible trace of every extraction, plan, tool call and guard decision. It is a testing aid; a real caller would not see it.

The *Consent* switch in the top bar chooses how the simulated policyholder responds when a representative calls (*Approves* or *Times out*); switching starts a new chat. The header shows the date used for deadline checks.

Three parts of the workflow are simulated, and the UI labels them that way: the human handoff (it ends the chat with a reference number, but no live agent is connected), the email summary (written to the *Test outbox* panel, never emailed), and the policyholder's consent for representative callers.

Before verification, the inspector shows which identity fields were provided but not which ones matched, because that would let anyone probe the records. Set `DEBUG_INSPECTOR=true` locally to see per-field results. The key button shows whether an API key is available and lets you enter one. Reloading the page keeps the current conversation. The page follows the system light or dark theme.

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
5. **Output checks (code):** every draft passes four checks before the caller sees it. A draft that fails is regenerated once with a note saying what was wrong. If the second draft only fails grounding, its unsupported sentences are removed and the rest is used if it passes every check; otherwise a fixed reply built from the controller's directives is used. Rejected drafts are never shown or returned by the API.
   - *Disclosure guard:* before verification, rejects anything that reads as claim information: claim IDs, money amounts, dates from claim records, document names, "missing documents", explanations of a decision ("denied because…") and status statements, including confirming the caller's own guess ("Is CL-2048 denied?" → "Yes" or "CL-2048 was denied" is rejected; "I can check that once you're verified" is fine). After verification, rejects other callers' claims.
   - *Grounding check* ([grounding.py](app/sop/grounding.py)): after verification, each statement must match the trusted data **for what it claims to be**, not just contain a number seen somewhere. An amount called a reimbursement must equal a reimbursement field (so the $1,450 allowed maximum can't be presented as a payout); a time span must match a duration in the guidance or the days until a deadline; a date called a deadline must be an appeal deadline; a stated status must match the record; a denial explanation must match the recorded reason; outcome promises are never grounded. Trusted data is claim records, guidance and tool results. What the caller said is kept apart: the agent may mention or dispute it ("there's no $9,999 on this claim") but not agree with it.
   - *State-consistency check:* catches replies that contradict the SOP state, such as "details don't match" after verification succeeded, or "you're verified" before it. The prompt also states the current verification status explicitly.
   - *Required steps:* a reply must not drop a must-say element of its directives: the email-summary offer, the transfer reference number, or the confirmation that the email was sent.
6. **Recovery:** if anything fails unexpectedly during a turn, the session is restored to its state before the turn and the caller is asked to send the message again. Tool calls from the model are validated against a schema; malformed arguments are returned to the model as an error instead of failing the turn.

| Phase | Decided in code | LLM freedom | LLM tools |
|---|---|---|---|
| VERIFY_ID | identity match (≥3 of name, DOB, phone, email, ID last 4), failure lockout, representative authorization and consent | Phrasing, empathy | none |
| RESOLVE_INTENT | claim matching from remembered hints | Interpreting which claim and what need; disambiguation | none |
| PROCESS_CASE | claim ownership checks on every tool call | Multi-step reasoning grounded in tool results | `get_claim`, `list_claims`, `get_document_guidance`, `get_followup_guidance` |
| POST_PROCESS | send only after the caller agrees, only to the email on file; summary checked for identity details and ungrounded facts | Writing the summary | none |

Escalation to a human is available in every phase: on request, after repeated failed verification or refusal, after repeated off-topic questions (offered), or when a representative's consent times out.

### Identity rules

- Verification needs at least 3 matching fields and no mismatches. The policy number helps find the account but does not count.
- Names match aliases and reordered or joined spellings. Phones and dates of birth are matched across common formats.
- On a mismatch, the agent never says which field was wrong, and neither does the API. All provided fields are cleared and the caller is asked again.

### Email summary

The summary is written after the turn's reply, so a request like "tell me what to send, then email me the summary" gets a summary that includes that answer. The summary model gets the facts of every claim discussed and the conversation from the verification turn onward, with dates of birth, ID digits, phone numbers and email addresses masked. Its draft is rejected if it contains any of the caller's identity details or fails the grounding check against the claim records and guidance. A rejected draft is replaced by a template built from the same structured data: what was discussed, each claim's status and denial reason, the appeal deadline, and next steps (documents and their requirements, what to do if one is unavailable, how to submit, processing time).
- A representative must be on file for the policyholder and the policyholder must approve a consent request before anything is shared.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/unit                       # deterministic, no API key needed
pytest tests/scenarios -m llm           # end-to-end conversations, needs DEEPSEEK_API_KEY
SCENARIO_RUNS=3 pytest tests/scenarios  # repeat each scenario to check stability
```

Unit tests cover verification, claim matching, guidance mapping, the state machine, the output checks, email privacy, tool-argument validation, turn rollback, configuration validation and the HTTP API limits, using a scripted fake LLM where needed.

Scenario tests in [`tests/scenarios/`](tests/scenarios/) are YAML conversations. They check state and content, not exact wording. Expected facts are written in each YAML file and checked without the app's own guards: which dollar amounts may appear, required words (documents, deadlines), forbidden phrasings as regular expressions (e.g. $1,450 described as a reimbursement, "Yes" to a status guess before verification), and what the email must and must not contain. Every verified reply is additionally run through the grounding check. Covered: the official demo, partial or refused ID fields, national-ID callers, name and email aliases, verification lockout, representative consent (approved and timeout), an angry caller, repeated off-topic questions, claim disambiguation, prompt injection, accepting or declining the email, asking a question and ending in the same message (right after verification, mid-conversation, and together with the email request), a status guess before verification, and a caller asserting an amount.

---

## Project layout

```
app/
  sop/          state_machine.py (controller), gates.py (data visibility, disclosure guard), grounding.py (field-aware fact checks), policies.py (tone, escalation)
  llm/          client.py (DeepSeek), extractor.py, responder.py (replies, tool loop, email summary), prompts.py
  tools/        verification.py, claims.py, guidance.py, consent.py, email.py
  agent.py      one conversational turn
  memory.py     session memory, extraction schema, turn plan
  server.py     FastAPI app
web/            single-page chat UI and debug panel
insurance_claims/fixtures/   sample data
tests/          unit/ and scenarios/
```

## Deploying publicly

The defaults suit a local demo. Before putting it on the internet:

- Set `ACCESS_TOKEN` so strangers can't spend the server's API key, and share the code only with testers. Callers can also be asked to bring their own key by leaving `DEEPSEEK_API_KEY` unset.
- Keep `DEBUG_INSPECTOR=false`.
- Behind a reverse proxy, set `FORWARDED_ALLOW_IPS` (read by uvicorn) to the proxy's address so rate limiting sees real client IPs instead of the proxy's.
- Sessions, rate-limit counters and the outbox live in one process's memory. Run a single instance, or move them to a shared store first.

## Known limitations

- All data is read from local JSON fixtures. There is no real claims system or authentication backend.
- Sessions live in server memory, expire after `SESSION_TTL_MINUTES` of inactivity and are lost on restart.
- The human handoff is simulated: the conversation ends with a reference number, but no agent is connected.
- Emails go to an in-app test outbox; nothing is sent over SMTP.
- The consent flow is simulated from `consent_scenarios.json`: each caller message after the request checks the status once.
- The output checks are pattern-based (English wording, numbers written as digits). They catch the common ways a reply can leak, misstate or invent facts, but they are a second line of defense; the first is that the model never receives data the phase doesn't allow.
- The demo date defaults to 2026-03-01 so CL-2048's appeal window (deadline 2026-03-18) is still open. With `SOP_TODAY=real`, the deadline has passed and the agent says so and suggests a human representative.

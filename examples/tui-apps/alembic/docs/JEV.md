# Jev in alembic: question packs, gates, receipts

## What Jev is (and is not)

Jev is TypeSafe AI's "System One" model. It does not write text. You send it a
piece of *state* (a diff, a task record, a log, a message) plus a map of typed
questions, and it returns typed answers with calibrated probabilities, in one
call, all questions evaluated in parallel, typically in 70–500 ms.

Three question types cover every decision alembic needs:

| type | you give | you get |
|---|---|---|
| `noul` | a yes/no statement (`instructions`, optional `criteria.true/false`) | `noul`: P(true) in 0..1 |
| `choice` | `criteria`: map of option → description (2..255) | `choice` (winner), `probabilities` per option, `confidence` |
| `score` | `criteria`: ordered list of 2..10 level names | `score` (continuous position), `legend`, `probabilities` per level, `confidence` |

Wire format: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`,
body `{"model":"jev-latest","state":…,"questions":{…}}`. alembic reads the key from
`TYPESAFE_API_KEY`; without it every run is a **MOCK** — deterministic in (state,
question), clearly labelled in the UI and in receipts, useful for demos and tests
and worthless as a judgment.

Use Jev for listable decisions where a wrong answer is cheap to detect: routing,
triage, gating, "is this stuck", "who should review". Keep prose with an LLM and keep
decisions that are expensive *and hard to detect when wrong* with a human.

## Question packs

A pack is one JSON file: a named set of questions designed for one core task, plus
the gate that turns answers into a decision. Packs live in `packs/` (or
`ALEMBIC_PACKS`); alembic reloads them with `r` on the Jev tab.

```json
{
  "id": "commit-safety",
  "name": "Commit safety",
  "state_source": "staged_diff",
  "order": ["secrets", "irreversible", "area", "blast"],
  "questions": {
    "secrets":      {"type": "noul",   "instructions": "The diff adds credentials, tokens or keys.",
                     "criteria": {"true": "Literal secret material in added lines", "false": "None added"}},
    "area":         {"type": "choice", "instructions": "Which area does the diff mainly touch?",
                     "criteria": {"ui": "…", "api": "…", "infra": "…", "docs": "…", "tests": "…"}},
    "blast":        {"type": "score",  "instructions": "Blast radius if this change is wrong?",
                     "criteria": ["Cosmetic", "Single feature", "Multiple features", "Whole service", "Multiple services"]}
  },
  "gate": {
    "action": "commit.create",
    "favorable": {"secrets": "no", "irreversible": "no", "blast": "<=Multiple features"},
    "minProbability": 0.75, "autoConfidence": 0.9, "refuseBelow": 0.35
  }
}
```

`state_source` tells alembic where the state comes from:

| value | state sent to Jev |
|---|---|
| `task` | the selected task record as JSON |
| `events` | task record + its last 30 events |
| `staged_diff` / `working_diff` | `git diff --cached` / `git diff` of the selected worktree |
| `text` | whatever you type in the State pane |
| `file` | the contents of a path you enter |

### Designing a pack

1. Write the decision the pack serves as one action id (`pr.automerge`, `task.autocontinue`).
   If you cannot name the action, it is a report, not a pack — leave `gate` out.
2. Ask the fewest questions that make the decision. Four is typical; the reference
   harness uses exactly four for proposal review.
3. Phrase each `instructions` as a statement Jev can check against the state, not a
   request for an opinion. Give `criteria` for noul so the yes/no boundary is explicit.
4. Put the safety-critical questions first in `order`; the Result pane reads top-down.
5. Choose thresholds and *write them down as chosen*, not derived: `minProbability`
   (favorable-side mass the weakest answer must reach), `autoConfidence` (calibrated
   confidence the weakest answer must reach to auto-act), `refuseBelow` (favorable
   mass below which the pack refuses rather than escalates). Tune them against
   receipts, not intuition.

### Gate semantics (aurum-gate vocabulary)

For each question in `favorable`: noul favorable `yes` means P ≥ 0.5 (mass = P),
`no` means P < 0.5 (mass = 1−P); choice favorable is an option (mass = its
probability); score favorable is a level, or `<=Level` / `>=Level` (mass = summed
probability on the favorable side).

```
any unfavorable and its mass < refuseBelow  → REFUSE
any unfavorable                             → ESCALATE
weakest favorable mass < minProbability     → ESCALATE
weakest confidence   < autoConfidence       → ESCALATE
otherwise                                   → AUTO
```

alembic never acts on AUTO by itself. It shows the decision, saves the receipt, and
(with `s`) sends the receipt to the harness through the outbox. The harness — and a
human at the irreversible boundary — decides.

## Receipts

Every run writes `~/.alembic/receipts/<time>-<pack>.json`: pack, state source and
SHA-256 of the state, model, mock flag, latency, token usage, every answer, decision,
reason, and a short digest of the record. Receipts are written once and never edited.
The Jev tab's history (`h`) and compare (`c`) views read them back, so you can watch a
pack's decisions drift as you tune thresholds, and `s` ships one to the harness as a
`jev.receipt` command.

## Triage: the observability tick

Question packs answer "should this one action proceed?". Triage answers "which task
needs a human next?" — every minute, cheaply, for every task.

Two layers, in this order:

1. **Deterministic** (no model, no cost, every tick): a score in 0..1 per task from
   state (`blocked` 0.75 · `failed` 0.70 · `review` 0.55 · `running` 0.35 · `queued` 0.20 ·
   `done` 0), plus priority (+0.10 per level), staleness (running and untouched for
   10 m +0.15, 30 m +0.25), recent errors (+0.05 each, cap 0.15), unanswered pings
   (+0.10) and no progress since the previous tick (+0.10). Each addend leaves a
   reason string, so the ranking is explainable without Jev.
2. **Jev, only for the ambiguous** — tasks that are blocked, in review, failed, stale,
   erroring or ignoring pings. Their compact states (id, title, state, agent,
   progress, status line, minutes idle, last five events) go in **one** call with
   three namespaced questions each: `stuck` (noul), `needs_human` (noul), `next`
   (choice: wait / ping / review / cancel / retry). Results add up to +0.15·P(stuck)
   and +0.15·P(needs human) and override the default next action when Jev's
   confidence is ≥ 0.60.

Budget is declared before anything is spent: `MaxTasksPerTick` (8), `MaxCallsPerHour`
(60), `ChunkSize` (6 tasks = 18 questions per call). At the ceiling the tick still
runs, deterministically, and says so (`skipped: "budget: 60 calls/hour reached"`).
Cost is recorded per tick from `usage.input_tokens` at the list price ($0.042 / M):
a typical tick with eight ambiguous tasks is on the order of $0.0001.

Every tick is appended to `~/.alembic/triage.jsonl`. When the ranking's top task or any
recommended action changes, the tick is also sent to the harness as a `triage` outbox
command, so the harness can route "what next" on the same evidence the operator sees.

Run it from the TUI (`t` now, `--triage-interval 60s` on a timer, `s` to sort by it,
`N` to jump to the top task with its action pre-armed) or from a real cron:

```
* * * * *  alembic triage --once --json >> /var/log/alembic-triage.jsonl
```

Without `TYPESAFE_API_KEY` (or without `--live`) the tick is deterministic-only or uses
the mock, and every record says so (`"deterministic": true` / `"mock": true`).

### Live evidence (2026-09-28)

One triage tick over the demo snapshot (seven tasks, one made 35 minutes stale, one
with an unanswered ping), against the real model:

```
jev_calls=1  jev_tasks=4  input_tokens=1753  cost=$0.000074  latency=567ms
#1 1.00 ping    T-1042 blocked   [blocked, priority 2, 1 ping unanswered, jev: needs human 0.95]
#2 0.89 retry   T-0990 failed    [failed]                     stuck 0.60  needs human 0.67
#3 0.73 wait    T-1041 running   [running, priority 1, stale 35m]  stuck 0.17  needs human 0.06
#4 0.70 review  T-0977 review    [review, jev: needs human 0.89]
#5 0.35 wait    T-0981 running   deterministic only
#6 0.20 wait    T-2202 queued    deterministic only
#7 0.00 wait    T-2201 done
```

Worth noticing: the stale running task was *not* escalated — Jev read the events and
judged it progressing (stuck 0.17), so staleness alone did not produce a false ping.
At this cost, a tick every minute is about $0.10 a day for a harness of this size.

# Porting alembic into the Ormus agent harness — handoff

Audience: whoever (person or agent) opens `Ormus-Solutions/ormus-agent-harness` next.
Goal: the harness gets alembic as its operator TUI and its observability tick, with the
smallest change to the harness itself. Everything alembic needs from the harness is two
append-only JSONL files; everything the harness gets back arrives the same way.

## The shape of the port

```
ormus-agent-harness (TypeScript)                      alembic (Go binary, this repo)
┌──────────────────────────────┐   feed.jsonl   ┌──────────────────────────────┐
│ task runner / agents         │ ─────────────▶ │ Tasks · Worktrees · Jev · … │
│  + contrib/ormus-adapter.ts  │ ◀───────────── │ pings · cancel/retry · triage│
│  + contrib/triage.ts (cron)  │  outbox.jsonl  └──────────────────────────────┘
└──────────────────────────────┘
```

alembic is **not rewritten in TypeScript**. It stays a static Go binary (fast, 92 tests,
PTY-verified) that the harness launches (`ormus tui`) or the operator runs directly. The
harness side is ~150 lines: emit feed records where task state changes, consume outbox
commands. The deterministic half of triage is additionally ported to TypeScript
(`contrib/triage.ts`) so the harness can rank tasks itself on a timer, with or without
the TUI running — the Jev half is one HTTP call the harness already knows how to make.

## Step 0 — answered 2026-09-30 against `Ormus-Solutions/ormus-agent-harness` @ 5df85bc

The harness is a TypeScript monorepo (`packages/{kernel,gateway,agent,tools,channels,graph,harness}`,
Node ≥ 22.18, erasable-syntax TS, no build, vitest, **zero runtime npm deps by law**). Its laws
(`CONTRIBUTING.md`) reshape this port: no second TUI, no config flag, propose-never-act, every safety
claim tested, docs + a signed `LOGS.md` entry per change, and no push or PR unless Diego asks.

1. **Task state lives in** the dashboard roster: sessions (`~/.oah/profiles/<p>/sessions/*.jsonl`, `GET /api/sessions`)
   and subagents (`subagents.json`, in `/api/status`), painted by `packages/harness/src/tui/dashboard.ts` as
   `DashboardEntry`. There is no separate "task" entity and no feed file — **the JSONL feed/outbox contract is not needed
   inside the harness**; it stays the contract for external tools (the Go alembic) if ever wanted.
2. **States:** `AgentLife = idle | working | blocked | done` (`blocked` = an exec approval is pending, from `GET /api/life`).
   Mapping used: blocked→blocked, working→running, idle→(quiet, base 0.10), done→done.
3. **Workflow** = a profile (`~/.oah/profiles/<name>`: workspace, ledger, handbrake, schedules). Pantry goal atoms are plans, not live work.
4. **Ping** = `POST /api/chat {profile, session:"web:<name>", text}` — synchronous; the reply is the ack. **Cancel** = `POST /api/interrupt`
   or the handbrake. No retry primitive exists.
5. **Worktrees:** the harness does not run per-task worktrees. Not ported.
6. **Elements:** sessions have transcripts (`GET /api/session`), subagents a `task` line; no PR/URL list. `Enter` on a dashboard row already opens it.
7. **Jev:** mentioned in the pantry (H5 gates are "Jev-adjacent") but **not in code**; `api.typesafe.ai` would be new egress → security lane, separate change.
8. **Timer:** `OperatorCron` wakes an agent turn (costs a model call) — wrong for triage. A plain interval belongs in `serve.ts`; for now
   `ormus triage --every 60s` runs client-side, and cron can call it.

## What landed (PR 1, branch `feature/tasks-triage` in the harness, local, not pushed)

`packages/harness/src/tui/triage.ts` (pure scoring, same record shape as `jev/triage.go` / `contrib/triage.ts`),
ranked Agent Dashboard with `#n ▰▰▱▱ next` chips and reason lines, `/triage`, `ormus triage [--json] [--every]`,
plus a real bug fix (the EVENTS pane filtered on event names nothing emits). 12 tests; verify green apart from the
pre-existing Bubblewrap cases. Signed entry `CLAUDE-20260930-TRIAGE` in `LOGS.md`.

## Next (PR 2, needs Tomas's review): Jev advice on gates

`adviseGate(verdict, advice)` in `kernel/gate.ts` that may only downgrade (`run→ask`, `ask→hold`), never mint;
`JevClient` with the key at `<state>/typesafe-key` (0600); spend through the kernel `BudgetLedger`.
The question packs and the batched triage call from `contrib/triage.ts` are the reference.

## Original Step 0 questions (kept for the record)

1. Where does task state live and change? (class/store/DB; the single place to hook `feed.task(...)`)
2. What is a "task" called there, and what are its states? Map them onto
   `queued | running | blocked | review | done | failed`. Anything that doesn't map → nearest, and note it.
3. What is a "workflow" / "production workflow" there (pipeline? project? env?) → `workflow.upsert`.
4. How are agents addressed for a message? (queue, channel, method) → what `ping` delivery calls.
5. Does the harness already run git worktrees per task? If yes, set `task.worktree` and alembic links them.
6. Where do logs/PRs/URLs for a task live? → `elements[]`.
7. Is there already a Jev integration (aurum-gate / quicksilver-judge in the deps)? If yes, reuse its client for
   `contrib/triage.ts`; if not, the raw `fetch` in that file is enough.
8. Where can a 60 s timer live (existing scheduler? cron? `setInterval` in the daemon)?

## Step 1 — plug in (day 1)

1. Copy `contrib/ormus-adapter.ts` into the harness (no dependencies, Node 18+).
2. At the one place task state changes, call `feed.task({...})`; on notable log lines `feed.event(id, level, text)`;
   on agent lifecycle `feed.agent(...)`; once at boot `feed.workflow(...)` per workflow.
3. Start `outbox.watch(cmd => ...)` in the daemon: `ping` → deliver to the agent, then `feed.ack(cmd.id, agent, reply)`;
   `task.cancel` / `task.retry` → the harness's own cancel/retry, then report through the feed;
   `jev.receipt` and `triage` → store or ignore (they are evidence, not instructions).
4. Build alembic (`go build -o alembic .`) and run it against the real feed: `alembic --feed ~/.ormus/feed.jsonl`.
   Acceptance: every real task shows with a status line; `p` on a task produces an ack toast within a second;
   `x` on a running task cancels it in the harness.

## Step 2 — triage on a timer (day 1–2)

Option A (no TUI needed): `contrib/triage.ts` — deterministic ranking + one batched Jev call for the ambiguous
tasks, identical scoring to `jev/triage.go`. Run it from the harness scheduler every 60 s; it appends
`triage.jsonl` and returns the ranking for routing.
Option B: run the Go binary headless from cron: `alembic triage --once --json`.
Both write the same record shape, so the TUI's history reads either.

Acceptance: the harness picks its next task from `triage.tasks[0]` when idle; a blocked task with an
unanswered ping ranks first; the spend line in the log stays around $0.0001 per tick.

## Step 3 — brand and ship (day 2)

- The binary already carries Ormus chrome (`BRAND.md`). Add an `ormus tui` subcommand (or npm `bin`) that
  execs the binary with the harness's feed/outbox paths.
- Release: `goreleaser`-style matrix (linux/darwin × amd64/arm64) is enough; the binary has no runtime deps.

## What to copy, verbatim

| from this repo | to the harness | why |
|---|---|---|
| `contrib/ormus-adapter.ts` | `src/alembic/adapter.ts` | feed/outbox writer + watcher |
| `contrib/triage.ts` (+ `.test.ts`) | `src/alembic/triage.ts` | deterministic scorer + Jev batch |
| `docs/HARNESS-CONTRACT.md` | `docs/alembic-contract.md` | the wire format, v1 |
| `docs/JEV.md` | `docs/jev.md` | packs, gates, receipts, triage |
| `packs/*.json` | `packs/` | question packs for core tasks |
| the `alembic` binary (or this module) | `bin/` or a release asset | the TUI |

## What must NOT change without bumping `v`

Feed/outbox record types and field names in `docs/HARNESS-CONTRACT.md`. Add fields freely (both
readers ignore unknown ones); rename or remove nothing under `v: 1`.

## Known gaps to close once the repo is visible

- State-name mapping (Step 0.2) may need a `blocked` heuristic if the harness has no such state
  (e.g. "waiting on approval" events → `blocked`).
- If the harness already has a Jev client with retries/telemetry, `contrib/triage.ts` should call it
  instead of `fetch` so spend is metered in one place.
- If tasks are not 1:1 with git worktrees, the Worktrees tab still works (it reads `git worktree list`)
  but the task ↔ worktree link will be empty until `task.worktree` is set.

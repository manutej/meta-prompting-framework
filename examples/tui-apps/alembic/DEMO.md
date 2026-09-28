# alembic — demo script (≈2–3 minutes)

**What it is:** an operator console for an agent harness. The harness (Ormus, or
any Hermes-style runtime — see `docs/HARNESS-CONTRACT.md`) appends JSONL to a
*feed*; alembic tails it, shows every task with a live status line, and writes
*commands* (ping, cancel, retry, receipts) to an append-only *outbox*. Decisions
go through **Jev** question packs (TypeSafe AI's System One model, `docs/JEV.md`):
typed answers with calibrated probabilities, a gate, and a receipt on disk.

Start in a 120×40 terminal (80×24 works; the layout reflows):

```bash
cd examples/tui-apps/alembic && go build -o alembic . && ./alembic --demo
```

`--demo` seeds a believable harness under `~/.alembic/demo/`, advances it every
2.5 s, and acknowledges your pings. Gold is action and focus, navy is structure.

## Talk track

| Time | Press | Say / point at |
|------|-------|----------------|
| 0:00 | — | Header: **ALEMBIC · live 1s ago · 3 workflows · 7 tasks · 4 agents · jev: MOCK · demo**. "Everything on screen came from one JSONL file the harness appends to. Nothing is polled from an API." |
| 0:10 | `j` `j` | Tasks are grouped by workflow (prod/staging/dev badge, health dot) and sorted: urgent `!!` first, then blocked, failed, running, review, queued, done. The second line under each task is the agent's own status line — watch one change as the demo ticks. |
| 0:25 | `tab` `j` `enter` | Detail pane: state gauge, worktree with ↑↓ and dirty count, **elements** — `enter` opens a file in `$EDITOR` at the line, a PR in the browser (and copies the URL via OSC 52). `tab` back. |
| 0:40 | `p`, type `status?`, `enter` | Composer at the bottom (`tab` cycles canned pings). Toast **ping sent → Coder**. Within ~3 s the harness acks: toast **Coder: ack — on it** and an `operator:` event on the task. "That went through the outbox and came back through the feed." |
| 0:55 | `J` | "Now the decision layer." Spinner in the `─ jev ─` line, then **task-readiness: AUTO / ESCALATE / REFUSE** with the unfavourable questions. A receipt was written to `~/.alembic/receipts/`. |
| 1:05 | `3` | Jev tab, three columns: **Packs** (5 JSON packs, invalid ones in red), **State** (what will be sent — here the task + its last events, byte count at the bottom), **Result** (the questions with `NOUL` / `CHOICE` / `SCORE` badges). |
| 1:15 | `enter` | Result: a bipolar `no ◀ ░░░████ ▶ yes` bar per yes/no question (green ≥ 0.8 conf, amber, red), a ranked bar chart for the choice, a level ladder with `◀` on the nearest level for the score. Then the banner: **▌REFUSE task.autocontinue** with the reason and thresholds `p≥0.70 · conf≥0.85 · refuse<0.30`. Point at the amber **MOCK — deterministic, not a model** line: "same state, same answer, every time — set `TYPESAFE_API_KEY` and this line disappears." |
| 1:35 | `T`, pick a task, `enter` | Different state, different verdict. `h` opens the history: `c` marks one, `j`/`k` onto another shows a side-by-side of headline probabilities with green/red deltas. `enter` loads an old receipt back into Result. `esc`. |
| 1:50 | `s` | **receipt sent → harness** — the receipt goes into the outbox as `jev.receipt`, so the harness can act on the gate. (`y` copies the receipt path; `S` attaches a note.) |
| 2:00 | `2` | Worktrees: `git worktree list` enriched with ahead/behind, dirty count, HEAD, and which tasks live in each. `n` opens a form to add one; `d` removes (dirty trees need two confirmations); `J` runs the commit-safety / PR-triage pack on the staged or working diff. |
| 2:15 | `4` | Agents: state, model, current task, last seen, **pings sent / acks received** — the ping from earlier shows `1 / 1`. `enter` jumps to the agent's task. |
| 2:25 | `ctrl+k`, type `read` | Command palette — every action is a command, fuzzy-matched with gold highlights. `enter` reloads the packs. `?` shows every key per tab (`j`/`k` scroll on short terminals), `?` again closes. `q`. |

## If something goes sideways

- Header says **no feed**? You started without `--demo` and nothing has written
  `~/.ormus/feed.jsonl` yet — that is the empty state, not a crash. `--feed PATH` points elsewhere.
- **state is empty** on the Jev tab: the pack reads a diff and the selected worktree is
  clean; `W` picks another worktree, or `T` picks a task for the task/events packs.
- A pack shows in red: the JSON did not validate; the error is right under its name. `r` reloads.
- Worktrees says **not a git repo**: pass `--repo DIR` or start inside a repository.

## Known limits (honest)

- `--demo` is a **simulated harness** (`harness/demo.go`): scripted tasks, rotating status
  lines, acks for your pings and cancels. A real harness implements the contract in
  `docs/HARNESS-CONTRACT.md` (`contrib/ormus-adapter.ts` is a starting point).
- Jev is **MOCK** without `TYPESAFE_API_KEY`, and always in `--demo` unless you pass
  `--live`: deterministic in (state, question), labelled in the header, the result pane and
  every receipt. Mock verdicts are repeatable, not judgments.
- Receipts carry a digest, not a signature: they detect accidents, not tampering.
- The state preview and result rows are truncated to the pane width rather than wrapped.
- Pings sent counts this session only; acks are read from the feed and persist.

## Live verification (recorded 2026-09-28)

One real call through `jev/client.go` against `api.typesafe.ai/v1/systemone` with the
`ping-priority` pack and the state *"URGENT: prod checkout is returning 500s after the
idempotency deploy — stop the rollout now and roll back."*:

```
model=jev-1.13.0  latency=586ms  usage={input_tokens:475 output_tokens:78}
urgent     noul   0.99
blocking   noul   0.86
intent     choice stop 0.99  {answer:0 change:0.01 continue:0 stop:0.99}  conf 0.98
```

Wire format, parsing of all three answer types, retry and error mapping are covered by
`jev/jev_test.go` against a local HTTP stub; this run confirms the same code against the
real service. `--demo` deliberately uses the mock so rehearsals never spend calls on
simulated tasks; pass `--live` to gate the demo tasks with the real model.

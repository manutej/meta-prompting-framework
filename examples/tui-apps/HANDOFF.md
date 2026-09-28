# Handoff — state of examples/tui-apps (2026-09-28)

Branch `claude/nextgen-tui-startup-plan-012e7wH9eujTxec1P6qwteE9`. Everything below is committed and pushed
unless marked OPEN.

## Shipped

| App | Lines | Tests | Verified |
|---|---|---|---|
| nexus-command | ~1.2k | 18 | build · vet · tests · PTY smoke · frames captured |
| gitscope | ~3.5k | 50 (4 red-team bugs fixed) | same |
| docscope | ~3.5k | 45 (6 red-team bugs fixed) | same |
| alembic | ~5.7k + core | 41 UI + 13 core | same, plus ONE live Jev call (jev-1.13.0, 586 ms) |
| progress-timer, file-browser, system-monitor | small | 16 / 24 / 22 | reference apps, smoke passes |

Gates: `make test` (all six + alembic core) and `make smoke` (`scripts/smoke.py`, one spec table, real PTY).
Frames: `scripts/screenshot.py` → HTML; `scripts/preview_page.py` assembles the four-app preview.
Published preview (three apps, pre-alembic): https://claude.ai/artifact/FqYgDgfxNzGBi9g5dompJY

## alembic — how it plugs in

- Contract: `alembic/docs/HARNESS-CONTRACT.md` (v1). Two append-only JSONL files: harness → `~/.ormus/feed.jsonl`
  (`task.upsert`, `task.event`, `agent.upsert`, `workflow.upsert`, `ping.ack`), alembic → `~/.ormus/outbox.jsonl`
  (`ping`, `task.cancel`, `task.retry`, `jev.receipt`). Drop-in adapter: `alembic/contrib/ormus-adapter.ts`.
- Jev: `alembic/docs/JEV.md`. `POST api.typesafe.ai/v1/systemone`, `TYPESAFE_API_KEY`; noul/choice/score; packs in
  `alembic/packs/*.json` with aurum-gate thresholds (`minProbability`/`autoConfidence`/`refuseBelow` → auto/escalate/refuse);
  receipts in `~/.alembic/receipts/`. Mock when no key; `--demo` is mock unless `--live`.

## OPEN

1. **Ormus-Solutions/ormus-agent-harness is not readable from this session** (proxy: "GitHub access to this repository is
   not enabled for this session. Use add_repo"). Once attached, the work is: read its task/agent model, map it onto the
   feed contract (probably 100–200 lines in their TS), and adjust `harness/types.go` only if a field is genuinely missing.
   Everything else already runs against the demo harness.
2. alembic red-team pass was in flight at the time of writing; its fixes (if any) land as `alembic/qa_test.go` + minimal
   source edits, then `make test && make smoke`, re-capture alembic frames, regenerate + republish the preview page.
3. Research findings not in the repo yet: Hermes TUI docs (the bar): transcript-first, Ctrl+T live dock, Ctrl+X session
   switcher, `/agents` `/tasks` overlay, sessions in `~/.hermes/state.db` (SQLite), `jev-typesafe` plugin exposes
   `jev_check/route/score/evaluate`. Ormus public kits (all TypeScript, Jev-based): aurum-gate (gate router),
   quicksilver-judge (PR pre-filter, PASS/HOLD/FAIL), karat-filter, molten-cascade, gold-assay; index: liquid-gold.

## Demo tomorrow

`DEMO.md` (runbook, four apps, ~10 min) and each app's `DEMO.md` (talk track + known limits).

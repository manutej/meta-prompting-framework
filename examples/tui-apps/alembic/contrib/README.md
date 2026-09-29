# contrib — the harness side of alembic, in TypeScript

Drop-in files for `Ormus-Solutions/ormus-agent-harness` (Node 18+, zero dependencies).

| file | role |
|---|---|
| `ormus-adapter.ts` | `AlembicFeed` (write task/event/agent/workflow/ack records) and `AlembicOutbox` (`watch` for ping / cancel / retry / receipts / triage) |
| `triage.ts` | `Triager` — deterministic ranking every tick, one batched Jev call for ambiguous tasks under a budget; `loadFeed` folds a feed into a snapshot; `JevClient` for TypeSafe |
| `triage.test.ts` | 7 tests, `node --experimental-strip-types --test contrib/triage.test.ts` (or Vitest with no changes) |

Parity: `triage.ts` produces the same ranks, scores, next actions and reasons as `jev/triage.go`
for the same feed (`PARITY OK` in the port notes); both append the same JSONL record shape, so
alembic's history view reads ticks from either.

Typecheck: `tsc --noEmit --strict --module nodenext --moduleResolution nodenext --target es2022 --allowImportingTsExtensions`.
See `../PORT.md` for the step-by-step port.

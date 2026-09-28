# alembic ↔ harness contract (v1)

alembic is an operator console. It does not run agents; it observes a harness
and sends it short commands. The whole integration is two append-only JSONL
files. Both sides only ever append, so a crash on either side loses at most
one partial line, and the full history stays reconstructable.

```
harness ──appends──▶  ~/.ormus/feed.jsonl    ──tails──▶  alembic
harness ◀──tails───   ~/.ormus/outbox.jsonl  ◀─appends── alembic
```

Paths are configurable (`--feed`, `--outbox`, `ALEMBIC_FEED`, `ALEMBIC_OUTBOX`).
An HTTP or socket transport can be added later behind the same record types;
the file form is the reference because it needs no daemon and survives restarts.

## Feed records (harness → alembic)

One JSON object per line. `v` is the contract version, `ts` an RFC 3339 UTC
timestamp, `type` selects exactly one payload field. Unknown `type`s are ignored,
so a newer harness never breaks an older alembic.

### `task.upsert`
Full or partial task. Fields omitted on a later upsert keep their previous value
for `elements` and `created`; everything else is replaced.

```json
{"v":1,"ts":"2026-09-28T10:00:00Z","type":"task.upsert","task":{
  "id":"T-1041","workflow":"checkout","title":"Idempotency keys on /orders POST",
  "agent":"coder-1","state":"running","progress":0.55,
  "status_line":"writing dedupe middleware · 3 files touched",
  "worktree":"/home/me/src/checkout-idempotency","branch":"feat/idempotency-keys",
  "priority":1,
  "elements":[
    {"kind":"file","ref":"internal/orders/handler.go","line":88},
    {"kind":"pr","ref":"acme/checkout#412","label":"PR #412"},
    {"kind":"url","ref":"https://grafana.internal/d/orders","label":"dashboard"},
    {"kind":"log","ref":"runs/T-1041.log"}
  ]}}
```

| field | required | notes |
|---|---|---|
| `id` | yes | stable, unique across workflows |
| `workflow` | no | defaults to `default`; creates the workflow if unseen |
| `state` | yes | `queued` `running` `blocked` `review` `done` `failed` |
| `status_line` | yes | one line, present tense, what the agent is doing *now* — this is what the operator reads |
| `progress` | no | 0..1 |
| `worktree` | no | absolute path; alembic links it to `git worktree list` |
| `elements[].kind` | | `file` (ref = path relative to worktree, optional `line`), `pr` (`owner/repo#N`), `url`, `log`, `dir` |
| `priority` | no | 0 normal · 1 high (`!`) · 2 urgent (`!!`) |

### `task.event`
One line of history. `warn`/`error`/`ok` events also replace the task's status line.
```json
{"v":1,"ts":"…","type":"task.event","event":{"task_id":"T-1041","level":"warn","text":"CI red on push 3"}}
```

### `agent.upsert`
```json
{"v":1,"ts":"…","type":"agent.upsert","agent":{"id":"coder-1","name":"Coder","model":"claude-opus-5-5","state":"busy","current_task":"T-1041"}}
```

### `workflow.upsert`
```json
{"v":1,"ts":"…","type":"workflow.upsert","workflow":{"id":"checkout","name":"checkout-service","env":"prod","state":"healthy"}}
```

### `ping.ack`
The agent received a ping (see outbox). Shown as a toast and as a task event.
```json
{"v":1,"ts":"…","type":"ping.ack","ping_ack":{"ping_id":"20260928T100512-9f2a1c","agent":"coder-1","text":"ack — pausing after current step"}}
```

## Outbox commands (alembic → harness)

```json
{"v":1,"id":"20260928T100512-9f2a1c","ts":"…","type":"ping","agent":"coder-1","task_id":"T-1041","text":"stop after current step"}
{"v":1,"id":"…","ts":"…","type":"task.cancel","task_id":"T-0990"}
{"v":1,"id":"…","ts":"…","type":"task.retry","task_id":"T-0990"}
{"v":1,"id":"…","ts":"…","type":"jev.receipt","task_id":"T-1041","data":{ …receipt… }}
```

`id` is unique per command; a harness that replays the file must treat repeated
ids as one command (one intent, one effect). The harness should answer a `ping`
with a `ping.ack` carrying the same `ping_id`. `task.cancel` and `task.retry`
are requests: the harness decides, then reports the outcome through the feed.
Nothing in the outbox is executed by alembic itself.

## What alembic writes for Jev

Receipts (`~/.alembic/receipts/<time>-<pack>.json`) record every pack run:
pack id, state source and SHA-256 of the state, model, mock flag, latency,
token usage, every answer, the gate decision and reason, and a digest of the
record. A digest detects accidental edits; it is not a signature.

## Implementing the harness side

`contrib/ormus-adapter.ts` is a dependency-free TypeScript module that writes
feed records and tails the outbox with the exact types above. Drop it into the
harness, call `feed.task(...)` / `feed.event(...)` where task state changes,
and `outbox.watch(cmd => …)` to receive pings and cancel/retry requests.
Everything it emits is validated by the Go tests in `harness/`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { Triager, loadFeed, type Snapshot, type Asker } from "./triage.ts";
import { writeFileSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const T0 = Date.parse("2026-09-28T12:00:00Z");
const task = (id: string, state: any, priority = 0, ageMin = 0, progress = 0.5) =>
  ({ id, title: "t " + id, state, priority, progress, status_line: "", updated: new Date(T0 - ageMin * 60_000).toISOString() });
const snap = (...tasks: any[]): Snapshot => ({ tasks: Object.fromEntries(tasks.map(t => [t.id, t])), events: {} });

test("deterministic ordering matches the Go implementation", async () => {
  const tr = new Triager(); tr.now = () => T0;
  const s = snap(task("done", "done", 2), task("queued", "queued"), task("running-fresh", "running", 0, 1),
    task("running-stale", "running", 0, 40), task("blocked", "blocked", 0, 1), task("failed-urgent", "failed", 2, 1));
  s.events["running-fresh"] = [{ task_id: "running-fresh", level: "error", text: "boom" }];
  const out = await tr.run(s, { blocked: 2 });
  assert.deepEqual(out.tasks.map(t => t.id), ["failed-urgent", "blocked", "running-stale", "running-fresh", "queued", "done"]);
  assert.equal(out.skipped, "jev disabled"); assert.equal(out.deterministic, true);
  const b = out.tasks.find(t => t.id === "blocked")!;
  assert.ok(b.reasons.includes("2 pings unanswered")); assert.equal(b.next, "ping");
  assert.equal(out.tasks.find(t => t.id === "done")!.score, 0);
  assert.equal(out.tasks.find(t => t.id === "failed-urgent")!.next, "retry");
  assert.equal(out.cost_usd, 0);
});

test("stall baseline survives one-minute ticks and fires at five", async () => {
  const tr = new Triager(); let now = T0; tr.now = () => now;
  const r = task("r", "running", 0, 1); const s = snap(r);
  const first = await tr.run(s);
  for (let i = 1; i <= 4; i++) { now = T0 + i * 60_000; r.updated = new Date(now - 60_000).toISOString();
    const got = await tr.run(s); assert.ok(!got.tasks[0].reasons.some(x => x.startsWith("no progress")), `too early at ${i}`); }
  now = T0 + 5 * 60_000; r.updated = new Date(now - 60_000).toISOString();
  const fifth = await tr.run(s);
  assert.ok(fifth.tasks[0].score > first.tasks[0].score); assert.ok(fifth.tasks[0].reasons.includes("no progress for 5m"));
  r.progress = 0.6; now = T0 + 6 * 60_000;
  assert.ok(!(await tr.run(s)).tasks[0].reasons.some(x => x.startsWith("no progress")));
});

test("jev is consulted only for ambiguous tasks and merged", async () => {
  let calls = 0; let lastQ: Record<string, unknown> = {};
  const fake: Asker = { async ask(_s, q) { calls++; lastQ = q; const answers: any = {};
    for (const id of Object.keys(q)) {
      if (id.endsWith("__stuck")) answers[id] = { type: "noul", noul: 0.9 };
      else if (id.endsWith("__needs_human")) answers[id] = { type: "noul", noul: 0.1 };
      else answers[id] = { type: "choice", choice: "cancel", probabilities: { cancel: 0.8, wait: 0.2 }, confidence: 0.85 }; }
    return { answers, usage: { input_tokens: 1000, output_tokens: 10 }, mock: true }; } };
  const tr = new Triager({ client: fake }); tr.now = () => T0;
  const out = await tr.run(snap(task("blocked", "blocked", 0, 1), task("fresh", "running", 0, 1), task("queued", "queued")));
  assert.equal(calls, 1); assert.equal(out.jev_calls, 1); assert.equal(out.jev_tasks, 1); assert.equal(Object.keys(lastQ).length, 3);
  const b = out.tasks.find(t => t.id === "blocked")!;
  assert.equal(b.jev_used, true); assert.equal(b.stuck, 0.9); assert.equal(b.next, "cancel"); assert.equal(b.next_conf, 0.85);
  assert.ok(b.score > b.base); assert.ok(b.reasons.includes("jev: stuck 0.90"));
  assert.equal(out.tasks.find(t => t.id === "fresh")!.jev_used, false);
  assert.equal(out.cost_usd, 0); assert.equal(out.mock, true);
});

test("budget and chunking", async () => {
  let calls = 0;
  const fake: Asker = { async ask() { calls++; return { answers: {}, usage: { input_tokens: 1, output_tokens: 0 }, mock: false }; } };
  const tr = new Triager({ client: fake, budget: { maxTasksPerTick: 8, maxCallsPerHour: 3, chunkSize: 3 } });
  let now = T0; tr.now = () => now;
  const s = snap(...Array.from({ length: 10 }, (_, i) => task("b" + String.fromCharCode(97 + i), "blocked", 0, 1)));
  await tr.run(s); assert.equal(calls, 3);
  const second = await tr.run(s); assert.equal(calls, 3); assert.ok(second.skipped!.startsWith("budget"));
  now = T0 + 61 * 60_000; await tr.run(s); assert.equal(calls, 6);
});

test("jev error falls back to deterministic", async () => {
  const fake: Asker = { async ask() { throw new Error("429"); } };
  const tr = new Triager({ client: fake }); tr.now = () => T0;
  const out = await tr.run(snap(task("b", "blocked", 0, 1)));
  assert.equal(out.deterministic, true); assert.ok(out.skipped!.startsWith("jev error"));
});

test("question ids are safe and wording matches Go", () => {
  const qs = Triager.questions([{ id: "T-1041/x", title: 'q"uote', state: "running", status_line: "" }]);
  for (const id of Object.keys(qs)) { assert.ok(id.startsWith("T_1041_x__")); assert.ok(!/[-\/ ]/.test(id)); }
  assert.equal(qs["T_1041_x__stuck"].instructions, 'Task T-1041/x ("q\\"uote"): the agent is looping or has stopped making progress.');
});

test("loadFeed folds a feed like the Go snapshot", () => {
  const dir = mkdtempSync(join(tmpdir(), "triage-")); const p = join(dir, "feed.jsonl");
  writeFileSync(p, [
    JSON.stringify({ v: 1, ts: "2026-09-28T10:00:00Z", type: "task.upsert", task: { id: "T1", title: "a", state: "running", status_line: "x", elements: [{ kind: "file", ref: "a.go" }] } }),
    "not json",
    JSON.stringify({ v: 1, ts: "2026-09-28T10:01:00Z", type: "task.upsert", task: { id: "T1", title: "a", state: "review", status_line: "x" } }),
    JSON.stringify({ v: 1, ts: "2026-09-28T10:02:00Z", type: "task.event", event: { task_id: "T1", level: "warn", text: "hmm" } }),
    JSON.stringify({ v: 1, ts: "2026-09-28T10:02:00Z", type: "bogus" }),
  ].join("\n") + "\n");
  const s = loadFeed(p);
  assert.equal(s.tasks.T1.state, "review"); assert.equal(s.tasks.T1.elements!.length, 1); assert.equal(s.tasks.T1.workflow, "default");
  assert.equal(s.tasks.T1.status_line, "hmm"); assert.equal(s.events.T1.length, 1);
});
